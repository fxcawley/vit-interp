"""Export a completed run and re-render figures from stored maps, without inference."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .core import alpha_mask, paste, production_map, resized_rgb, sha256, write_json
from .report import save_effects, save_example, save_gallery, thumbnail


def export(run: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite export: {output}")
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if summary["manifest_sha256"] != sha256(run / "manifest.json"):
        raise ValueError("Manifest fingerprint differs from the numerical result")
    cfg, kind = manifest["protocol"], manifest["kind"]
    calibration = json.loads((run / "calibration.json").read_text())
    records = [json.loads(line) for line in (run / "pairs.jsonl").read_text().splitlines()]
    root = Path(manifest["data_root"])
    cache = {}

    def image(relative: str):
        if relative not in cache:
            path = root / relative
            category = relative.split("/")[0]
            if sha256(path) != manifest["inputs"][category]["image_sha256"][relative]:
                raise ValueError(f"Input image changed since inference: {relative}")
            cache[relative] = resized_rgb(path, cfg["resolution"], cfg["patch_size"])
        return cache[relative]

    def payload(record):
        size, p = cfg["edit_patches"] * cfg["patch_size"], cfg["patch_size"]
        receiver = image(record["receiver"])
        top, left = record["crop_top_left"]
        alpha = alpha_mask(size, record["construction"], cfg["feather_pixels"])
        row, col = record["location"]
        images = {"identity": receiver}
        for arm, key in (("sham", "sham_donor"), ("defect", "defect_donor")):
            crop = image(record[key])[top:top + size, left:left + size]
            images[arm] = paste(receiver, crop, row * p, col * p, alpha)
        with np.load(run / "raw" / f"{record['pair_id']}.npz") as stored:
            raw = {arm: stored[arm] for arm in images}
            remote = stored["remote"]
        return {"category": record["category"], "receiver": record["receiver"],
                "construction": record["construction"], "location": record["location"],
                "images": images, "raw": raw, "remote": remote,
                "processed": {arm: production_map(scores, receiver.shape[:2], cfg) for arm, scores in raw.items()}}

    output.mkdir(parents=True)
    for name in ("manifest.json", "summary.json", "calibration.json", "pairs.jsonl", "status.json", "review-template.json"):
        shutil.copyfile(run / name, output / name)
    for category in cfg["categories"]:
        selected = [r for r in records if r["category"] == category]
        first = selected[0]
        largest = max(selected, key=lambda r: r["defect_raw_remote_max"] - r["sham_raw_remote_max"])
        thresholds = calibration[category]["thresholds"]
        for name, record, selection in (
            ("first", first, "prespecified first receiver / hard / first location"),
            ("largest", largest, "EXPLORATORY largest defect-minus-sham remote score difference"),
        ):
            save_example(output / f"{category}-{name}.png", payload(record), cfg, thresholds,
                         kind=kind, selection=selection)
        gallery = []
        for receiver in dict.fromkeys(r["receiver"] for r in selected):
            row = [thumbnail(image(receiver), Path(receiver).name)]
            for record in (r for r in selected if r["receiver"] == receiver):
                sample = payload(record)
                for arm in ("sham", "defect"):
                    loc = record["location"]
                    row.append(thumbnail(sample["images"][arm], f"{record['construction'][:1]} {loc[0]},{loc[1]} {arm[:1]}"))
            gallery.append(row)
        save_gallery(output / f"{category}-all-pairs.png", gallery, category, kind)
    save_effects(output / "effects.png", summary["summaries"], cfg, kind)
    write_json(output / "export.json", {
        "source_run": str(run.resolve()), "manifest_sha256": sha256(run / "manifest.json"),
        "numerical_results_unchanged": True, "encoder_rerun": False,
        "render_source_sha256": sha256(Path(__file__)),
        "plot_source_sha256": sha256(Path(__file__).with_name("report.py")),
        "raw_maps_location": str((run / "raw").resolve()),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.output)
