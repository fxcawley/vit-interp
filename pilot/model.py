"""Pinned upstream encoder and an independent check against the released detector."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import cv2
import faiss
import numpy as np
import torch

from .core import ExactCosineBank, normalized_tensor, production_map, resized_rgb, sha256


def repo_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def verify_checkout(path: Path, expected: str) -> None:
    if repo_revision(path) != expected:
        raise ValueError(f"Pinned source revision differs: {path}")
    changed = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if changed:
        raise ValueError(f"Upstream tracked source is modified: {path}")


class Encoder:
    def __init__(self, repo: Path, checkpoint: Path, cfg: dict, assets: dict, device: str):
        verify_checkout(repo, assets["dinov2"]["revision"])
        if assets["checkpoint"]["model"] != cfg["model"]:
            raise ValueError("Checkpoint model does not match the protocol")
        digest = sha256(checkpoint)
        if digest != assets["checkpoint"]["sha256"]:
            raise ValueError("Checkpoint SHA256 mismatch")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
        sys.path.insert(0, str(repo.resolve()))
        backbones = importlib.import_module("dinov2.hub.backbones")
        builder = getattr(backbones, cfg["model"])
        self.model = builder(pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.device, self.cfg = device, cfg
        self.checkpoint_sha256 = digest

    @torch.inference_mode()
    def features(self, image: np.ndarray) -> torch.Tensor:
        tensor = normalized_tensor(image).unsqueeze(0).to(self.device)
        # Upstream default n=1, norm=True: last block, post-final-LN, patch tokens only.
        return self.model.get_intermediate_layers(tensor)[0][0].float()


def verify_reference(
    repo: Path, encoder: Encoder, bank: ExactCosineBank, split: dict, donor: Path,
    category: str, cfg: dict, assets: dict, output: Path,
) -> dict:
    verify_checkout(repo, assets["anomalydino"]["revision"])
    sys.path.insert(0, str(repo.resolve()))
    backbones = importlib.import_module("src.backbones")
    detection = importlib.import_module("src.detection")
    utils = importlib.import_module("src.utils")
    with patch("torch.hub.load", return_value=encoder.model):
        reference = backbones.DINOv2Wrapper(cfg["model"], encoder.device, cfg["resolution"])

    # Run the UNMODIFIED released detector on a small, deterministic dataset subset.
    # Its bank, preprocessing, FAISS implementation, scoring and map conversion are independent checks.
    dataset = output / "reference_input"
    first_receiver = split["receivers"][0]
    paths = {f"{category}/train/good/{p.name}": p for p in split["bank"]}
    paths[f"{category}/test/good/{first_receiver.name}"] = first_receiver
    paths[f"{category}/test/{donor.parent.name}/{donor.name}"] = donor
    for relative, path in paths.items():
        destination = dataset / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    # The upstream runner synchronizes CUDA unconditionally, even in CPU mode.
    synchronize = patch("torch.cuda.synchronize", lambda: None) if encoder.device == "cpu" else nullcontext()
    with synchronize:
        detection.run_anomaly_detection(
            reference, category, str(dataset), cfg["reference_images"],
            {category: [donor.parent.name]}, str(output / "reference_output"),
            masking=False, mask_ref_images=False, rotation=False, knn_metric="L2_normalized",
            knn_neighbors=1, faiss_on_cpu=True, seed=0, save_patch_dists=True,
            save_examples=False, save_tiffs=False,
        )
    errors = {"preprocessing": 0.0, "raw_scores": 0.0, "production_map": 0.0, "feature_tensor": 0.0}
    for path in (first_receiver, donor):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        prepared = resized_rgb(path, cfg["resolution"], cfg["patch_size"])
        upstream_tensor, grid = reference.prepare_image(rgb)
        errors["preprocessing"] = max(errors["preprocessing"], float((upstream_tensor - normalized_tensor(prepared)).abs().max()))
        features = encoder.features(prepared)
        expected_features = reference.extract_features(upstream_tensor)
        errors["feature_tensor"] = max(errors["feature_tensor"], float(np.max(np.abs(features.cpu().numpy() - expected_features))))
        ours = bank.score(features).reshape(grid)
        relative = f"anomaly_maps/seed=0/{category}/test/{path.parent.name}/{path.stem}.npy"
        expected = np.load(output / "reference_output" / relative)
        errors["raw_scores"] = max(errors["raw_scores"], float(np.max(np.abs(ours - expected))))
        actual_map = production_map(ours, rgb.shape[:2], cfg)
        expected_map = utils.dists2map(expected, rgb.shape)
        errors["production_map"] = max(errors["production_map"], float(np.max(np.abs(actual_map - expected_map))))
    if max(errors.values()) > cfg["numeric_tolerance"]:
        raise AssertionError(f"Released-pipeline parity failed: {errors}")
    return {"passed": True, "max_absolute_errors": errors, "tolerance": cfg["numeric_tolerance"],
            "reference_revision": assets["anomalydino"]["revision"], "reference_mode": "informed; five-shot; seed=0"}
