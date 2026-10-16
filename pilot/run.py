"""Run an auditable exploratory pilot; procedural runs cannot reach a scientific gate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .core import (
    ExactCosineBank, alpha_mask, assert_unchanged, calibration_threshold, check_unique_files,
    classify, densest_mask_crop, paired_summary, paste, patch_mask, pixel_mask,
    production_map, regions, resized_mask, resized_rgb, select_split, sha256, write_json,
)
from .model import Encoder, repo_revision, verify_reference
from .report import save_effects, save_example, save_gallery, thumbnail


BASE = Path(__file__).resolve().parent


def chosen_donors(root: Path, category: str, split: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    selected, excluded = [], []
    size = cfg["edit_patches"] * cfg["patch_size"]
    support = alpha_mask(size, "feather", cfg["feather_pixels"]) > 0.5
    for path in split["defect_candidates"]:
        image = resized_rgb(path, cfg["resolution"], cfg["patch_size"])
        mask_path = root / category / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
        mask = resized_mask(mask_path, image.shape[:2])
        if not mask.any():
            excluded.append({"image": str(path.relative_to(root)), "reason": "empty resized GT"})
            continue
        top, left = densest_mask_crop(mask, size)
        crop_mask = mask[top : top + size, left : left + size]
        if (crop_mask & support).sum() < cfg["minimum_visible_defect_pixels"]:
            excluded.append({"image": str(path.relative_to(root)), "reason": "insufficient GT in feathered center"})
            continue
        selected.append({"path": path, "mask_path": mask_path, "top": top, "left": left,
                         "image": image, "mask": crop_mask})
        if len(selected) == len(split["receivers"]):
            break
    if len(selected) != len(split["receivers"]):
        raise ValueError(f"{category}: cannot assign one unique qualifying donor per receiver; no silent reuse")
    return selected, excluded


def run(args) -> int:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {args.output}")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    cfg = json.loads(args.config.read_text())
    assets = json.loads((BASE / "assets.json").read_text())
    status_path = args.output / "status.json"

    def status(state, **details):
        write_json(status_path, {"state": state, "kind": args.kind, "elapsed_seconds": time.monotonic() - start,
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(), **details})

    def budget():
        if time.monotonic() - start > cfg["max_runtime_seconds"]:
            raise TimeoutError("Prespecified accelerator-resident wall-time budget exhausted")

    status("checking_assets")
    try:
        if not args.data_root.is_dir():
            status("blocked", reason="Approved dataset directory is absent; no scientific evaluation performed")
            return 2
        marker = args.data_root / "_procedural.json"
        if marker.exists() and args.kind != "procedural":
            raise ValueError("Procedural fixtures must not be labeled MVTec")
        if args.kind == "procedural" and not marker.is_file():
            raise ValueError("Procedural provenance marker is required")
        if not args.checkpoint.is_file():
            status("blocked", reason="Checkpoint is absent")
            return 2
        # Fixed study geometry; non-square inputs or alternative modes need a new protocol.
        if cfg["resolution"] % cfg["patch_size"] or cfg["postprocess_sigma_pixels"] != 4:
            raise ValueError("Unsupported protocol geometry or departure from released postprocessing")
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        splits, donors, input_manifest = {}, {}, {}
        for category in cfg["categories"]:
            split = select_split(args.data_root, category, cfg)
            chosen, excluded = chosen_donors(args.data_root, category, split, cfg)
            splits[category], donors[category] = split, chosen
            roles = {k: split[k] for k in ("bank", "calibration", "clean_donors", "receivers")}
            roles["defect_donors"] = [d["path"] for d in chosen]
            hashes = check_unique_files(roles)
            input_manifest[category] = {
                "roles": {k: [str(p.relative_to(args.data_root)) for p in paths] for k, paths in roles.items()},
                "image_sha256": {str(Path(p).relative_to(args.data_root)): h for p, h in hashes.items()},
                "mask_sha256": {str(d["mask_path"].relative_to(args.data_root)): sha256(d["mask_path"]) for d in chosen},
                "donor_crops": [{"top": d["top"], "left": d["left"]} for d in chosen],
                "excluded_by_prescore_geometry": excluded,
            }
        manifest = {
            "kind": args.kind, "purpose": "engineering only" if args.kind == "procedural" else "exploratory real-data pilot",
            "protocol": cfg, "protocol_sha256": sha256(args.config), "assets": assets,
            "checkpoint_sha256": sha256(args.checkpoint), "repository_revision": repo_revision(BASE.parent),
            "code_sha256": {p.name: sha256(p) for p in sorted(BASE.glob("*.py"))},
            "inputs": input_manifest, "data_root": str(args.data_root.resolve()),
            "environment": {p: version(p) for p in ("torch", "torchvision", "numpy", "Pillow", "scipy",
                                                   "opencv-python-headless", "faiss-cpu", "matplotlib", "scikit-learn")},
            "python": sys.version, "platform": platform.platform(), "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "precision": "float32; TF32 off; no autocast; one image per forward",
            "inferential_limit": "CIs conditional on the fixed nominal bank, calibration images and donor assignment",
        }
        write_json(args.output / "manifest.json", manifest)
        fingerprint = sha256(args.output / "manifest.json")
        status("loading_model", manifest_sha256=fingerprint)
        encoder = Encoder(BASE / "external/dinov2", args.checkpoint, cfg, assets, args.device)
        records, summaries, reference_checks, calibration, integrity = [], {}, {}, {}, []
        inspection_files = []
        (args.output / "raw").mkdir()
        with (args.output / "pairs.jsonl").open("w", encoding="utf-8") as stream:
            for category in cfg["categories"]:
                split = splits[category]
                budget()
                status("building_bank", category=category)
                features = []
                for path in split["bank"]:
                    image = resized_rgb(path, cfg["resolution"], cfg["patch_size"])
                    features.append(encoder.features(image))
                bank = ExactCosineBank(torch.cat(features))
                del features

                def score(image):
                    budget()
                    if image.shape != (cfg["resolution"], cfg["resolution"], 3):
                        raise ValueError("This pilot requires square, fixed-resolution stimuli")
                    return bank.score(encoder.features(image)).reshape(
                        cfg["resolution"] // cfg["patch_size"], cfg["resolution"] // cfg["patch_size"]
                    )

                status("checking_released_pipeline", category=category)
                reference_checks[category] = verify_reference(
                    BASE / "external/AnomalyDINO", encoder, bank, split, donors[category][0]["path"],
                    category, cfg, assets, args.output / category,
                )
                raw_max, processed_max = [], []
                status("calibrating", category=category)
                for path in split["calibration"]:
                    image = resized_rgb(path, cfg["resolution"], cfg["patch_size"])
                    raw = score(image)
                    raw_max.append(float(raw.max()))
                    processed_max.append(float(production_map(raw, image.shape[:2], cfg).max()))
                thresholds = {"raw": calibration_threshold(raw_max, cfg["calibration_image_fpr"]),
                              "processed": calibration_threshold(processed_max, cfg["calibration_image_fpr"])}
                calibration[category] = {"thresholds": thresholds, "raw_image_maxima": raw_max,
                                         "processed_image_maxima": processed_max,
                                         "rule": "ceil((n+1)*(1-alpha))-th order statistic of whole-image maxima"}
                write_json(args.output / "calibration.json", calibration)
                first_payload = peak_payload = None
                largest_drift = -np.inf
                gallery = []
                size, p = cfg["edit_patches"] * cfg["patch_size"], cfg["patch_size"]
                for index, receiver_path in enumerate(split["receivers"]):
                    status("paired_inference", category=category, receiver=index + 1,
                           receiver_total=len(split["receivers"]))
                    receiver = resized_rgb(receiver_path, cfg["resolution"], p)
                    raw_identity = score(receiver)
                    # Repeated identity forwards check inference/scoring noise per independent receiver.
                    identity_error = float(np.max(np.abs(score(receiver.copy()) - raw_identity)))
                    if identity_error > cfg["numeric_tolerance"]:
                        raise AssertionError(f"Identity repeatability failed: {identity_error}")
                    processed_identity = production_map(raw_identity, receiver.shape[:2], cfg)
                    donor = donors[category][index]
                    clean_path = split["clean_donors"][index]
                    clean = resized_rgb(clean_path, cfg["resolution"], p)
                    dt, dl = donor["top"], donor["left"]
                    defect_crop = donor["image"][dt : dt + size, dl : dl + size]
                    clean_crop = clean[dt : dt + size, dl : dl + size]
                    visual_row = [thumbnail(receiver, f"base {index:02d}")]
                    for construction in cfg["constructions"]:
                        alpha = alpha_mask(size, construction, cfg["feather_pixels"])
                        for location in cfg["locations"]:
                            row, col = location
                            edited_region, remote = regions(receiver.shape[:2], row, col, cfg)
                            top, left = row * p, col * p
                            images = {"identity": receiver,
                                      "sham": paste(receiver, clean_crop, top, left, alpha),
                                      "defect": paste(receiver, defect_crop, top, left, alpha)}
                            gt = np.zeros(receiver.shape[:2], dtype=bool)
                            gt[top : top + size, left : left + size] = donor["mask"] & (alpha > 0.5)
                            local = patch_mask(gt, p)
                            if gt.sum() < cfg["minimum_visible_defect_pixels"]:
                                raise AssertionError("Selected donor violates the prespecified visible-GT rule")
                            for arm in ("sham", "defect"):
                                assert_unchanged(receiver, images[arm], edited_region, p)
                            raw = {"identity": raw_identity, "sham": score(images["sham"]), "defect": score(images["defect"])}
                            processed = {"identity": processed_identity,
                                         **{arm: production_map(raw[arm], receiver.shape[:2], cfg) for arm in ("sham", "defect")}}
                            # A direct postprocessing-spread control: change only edit-region raw scores.
                            local_only = raw_identity.copy()
                            local_only[edited_region] = raw["defect"][edited_region]
                            remote_pixels = pixel_mask(remote, p)
                            leakage = float(np.max(np.abs(
                                production_map(local_only, receiver.shape[:2], cfg)[remote_pixels]
                                - processed_identity[remote_pixels]
                            )))
                            if leakage > cfg["numeric_tolerance"]:
                                raise AssertionError("Guard band does not exclude direct smoothing/interpolation spread")
                            pair_id = f"{category}-{index:03d}-{construction}-{row}-{col}"
                            record = {"pair_id": pair_id, "category": category,
                                      "receiver": str(receiver_path.relative_to(args.data_root)),
                                      "defect_donor": str(donor["path"].relative_to(args.data_root)),
                                      "sham_donor": str(clean_path.relative_to(args.data_root)),
                                      "construction": construction, "location": location,
                                      "crop_top_left": [dt, dl], "visible_defect_pixels": int(gt.sum()),
                                      "remote_patch_count": int(remote.sum()), "identity_max_error": identity_error,
                                      "postprocessing_only_remote_error": leakage, "outside_edit_pixels_identical": True}
                            for arm in ("identity", "sham", "defect"):
                                record[f"{arm}_local_alarm"] = bool(np.max(raw[arm][local]) > thresholds["raw"])
                                for stage, maps, mask in (("raw", raw, remote), ("processed", processed, remote_pixels)):
                                    scores = maps[arm][mask]
                                    record[f"{arm}_{stage}_remote_alarm"] = bool(np.max(scores) > thresholds[stage])
                                    record[f"{arm}_{stage}_remote_max"] = float(np.max(scores))
                                    record[f"{arm}_{stage}_remote_area"] = float(np.mean(scores > thresholds[stage]))
                            record["raw_remote_mean_absolute_drift"] = float(np.mean(np.abs(raw["defect"][remote] - raw_identity[remote])))
                            np.savez_compressed(args.output / "raw" / f"{pair_id}.npz",
                                                **raw, edited=edited_region, remote=remote, local=local)
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                            stream.flush()
                            records.append(record)
                            integrity.append({"pair_id": pair_id, "identity": identity_error, "direct_spread": leakage})
                            payload = {"category": category, "receiver": receiver_path.name, "construction": construction,
                                       "location": location, "images": images, "raw": raw, "processed": processed, "remote": remote}
                            if first_payload is None:
                                first_payload = payload
                            drift = record["defect_raw_remote_max"] - record["sham_raw_remote_max"]
                            if drift > largest_drift:
                                peak_payload, largest_drift = payload, drift
                            for arm in ("sham", "defect"):
                                visual_row.append(thumbnail(images[arm], f"{construction[:1]} {row},{col} {arm[:1]}"))
                    gallery.append(visual_row)
                    print(f"{category}: {index + 1}/{len(split['receivers'])} receivers complete", flush=True)
                for label, payload, selection in (
                    ("first", first_payload, "prespecified first receiver / hard / first location"),
                    ("largest", peak_payload, "EXPLORATORY largest defect-minus-sham remote score difference"),
                ):
                    name = f"{category}-{label}.png"
                    save_example(args.output / name, payload, cfg, thresholds, kind=args.kind, selection=selection)
                    inspection_files.append(name)
                gallery_name = f"{category}-all-pairs.png"
                save_gallery(args.output / gallery_name, gallery, category, args.kind)
                inspection_files.append(gallery_name)
                for construction in cfg["constructions"]:
                    selected = [r for r in records if r["category"] == category and r["construction"] == construction]
                    values = {"raw_remote": paired_summary(selected, "raw_remote_alarm", cfg),
                              "processed_remote": paired_summary(selected, "processed_remote_alarm", cfg),
                              "local": paired_summary(selected, "local_alarm", cfg)}
                    values["decision"] = classify(values["raw_remote"], values["local"], cfg, valid=True,
                                                  data_kind=args.kind, reviewed=False)
                    values["rates"] = {f"{arm}_{endpoint}": float(np.mean([r[f"{arm}_{endpoint}"] for r in selected]))
                                       for arm in ("identity", "sham", "defect")
                                       for endpoint in ("raw_remote_alarm", "processed_remote_alarm", "local_alarm")}
                    summaries[f"{category}/{construction}"] = values
                del bank

        if args.device == "cuda":
            torch.cuda.synchronize()
        summary = {"kind": args.kind, "manifest_sha256": fingerprint, "summaries": summaries,
                   "reference_checks": reference_checks, "integrity_passed": True,
                   "pair_count": len(records), "max_identity_error": max(x["identity"] for x in integrity),
                   "max_direct_postprocessing_spread": max(x["direct_spread"] for x in integrity),
                   "elapsed_seconds": time.monotonic() - start,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                   "scientific_claim": "none; procedural data are engineering fixtures, real-data results require review",
                   "additional_gates": ["independent evidence", "current backbone", "mechanism", "matched-recall remedy"]}
        write_json(args.output / "summary.json", summary)
        save_effects(args.output / "effects.png", summaries, cfg, args.kind)
        inspection_files.append("effects.png")
        write_json(args.output / "review-template.json", {
            "manifest_sha256": fingerprint, "reviewer": "", "inspected_files": [],
            "required_files": inspection_files, "data_provenance_verified": False,
            "all_edits_plausible": False, "largest_effects_and_controls_reviewed": False,
            "no_postscore_exclusions": False, "notes": "",
        })
        status("complete_engineering_only" if args.kind == "procedural" else "complete_pending_review",
               manifest_sha256=fingerprint, elapsed_seconds=time.monotonic() - start)
        return 0
    except TimeoutError as exc:
        status("budget_stop_inconclusive", reason=str(exc))
        return 3
    except Exception as exc:
        status("failed_no_scientific_conclusion", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("mvtec", "procedural"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=BASE / "protocol.json")
    parser.add_argument("--checkpoint", type=Path, default=BASE / "checkpoints/dinov2_vits14_pretrain.pth")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
