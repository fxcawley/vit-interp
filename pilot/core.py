"""Experimental primitives; no model downloads or implicit dataset fallbacks."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def category_rng(seed: int, category: str) -> np.random.Generator:
    offset = int(hashlib.sha256(category.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed + offset)


def resized_rgb(path: Path, resolution: int, patch_size: int) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image = transforms.Resize(
            resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )(image)
        array = np.asarray(image).copy()
    h, w = array.shape[:2]
    return array[: h - h % patch_size, : w - w % patch_size].copy()


def normalized_tensor(image: np.ndarray) -> torch.Tensor:
    return transforms.functional.normalize(
        transforms.functional.to_tensor(image),
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )


def resized_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(
            source.convert("L").resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        ) > 0


def select_split(root: Path, category: str, cfg: dict) -> dict:
    """Reference selection matches upstream seed=0; remaining roles never overlap."""
    train = sorted((root / category / "train/good").glob("*.png"))
    receivers = sorted((root / category / "test/good").glob("*.png"))
    defects = sorted(p for p in (root / category / "test").glob("*/*.png") if p.parent.name != "good")
    n_ref, n_cal = cfg["reference_images"], cfg["calibration_images"]
    if not receivers or not defects or len(train) < n_ref + n_cal + len(receivers):
        raise ValueError(f"{category}: insufficient bank/calibration/clean-donor/evaluation images")
    bank = train[:n_ref]
    rng = category_rng(cfg["seed"], category)
    remaining = [train[n_ref:][i] for i in rng.permutation(len(train) - n_ref)]
    return {
        "bank": bank,
        "calibration": remaining[:n_cal],
        "clean_donors": remaining[n_cal : n_cal + len(receivers)],
        "receivers": receivers,
        "defect_candidates": [defects[i] for i in rng.permutation(len(defects))],
    }


def check_unique_files(groups: dict[str, list[Path]]) -> dict[str, str]:
    """Reject content duplicates as well as path leakage between sampling roles."""
    hashes, seen = {}, {}
    for role, paths in groups.items():
        for path in paths:
            digest = sha256(path)
            if digest in seen:
                raise ValueError(f"Duplicate image content: {seen[digest]} and {role}:{path}")
            seen[digest] = f"{role}:{path}"
            hashes[str(path)] = digest
    return hashes


def densest_mask_crop(mask: np.ndarray, size: int) -> tuple[int, int]:
    """Use GT only, before model scoring; tie-breaking is fixed row-major order."""
    if size > min(mask.shape) or not mask.any():
        raise ValueError("Cannot select a nonempty defect crop")
    integral = cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_64F)
    sums = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return tuple(int(x) for x in np.unravel_index(np.argmax(sums), sums.shape))


def alpha_mask(size: int, construction: str, feather: int) -> np.ndarray:
    if construction == "hard":
        return np.ones((size, size), dtype=np.float32)
    if construction != "feather" or feather <= 0 or 2 * feather >= size:
        raise ValueError("Invalid edit construction/feather width")
    distance = np.minimum(np.arange(size), np.arange(size)[::-1])
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.minimum(distance / feather, 1))
    return (ramp[:, None] * ramp[None, :]).astype(np.float32)


def paste(receiver: np.ndarray, crop: np.ndarray, top: int, left: int, alpha: np.ndarray) -> np.ndarray:
    h, w = alpha.shape
    if crop.shape != (h, w, 3) or top < 0 or left < 0 or top + h > receiver.shape[0] or left + w > receiver.shape[1]:
        raise ValueError("Edit exceeds the image or has incompatible shape")
    result = receiver.copy()
    original = receiver[top : top + h, left : left + w].astype(np.float32)
    result[top : top + h, left : left + w] = np.rint(
        original * (1 - alpha[..., None]) + crop.astype(np.float32) * alpha[..., None]
    ).clip(0, 255).astype(np.uint8)
    return result


def regions(shape: tuple[int, int], row: int, col: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    p, width, guard = cfg["patch_size"], cfg["edit_patches"], cfg["guard_patches"]
    grid = (shape[0] // p, shape[1] // p)
    if row < 0 or col < 0 or row + width > grid[0] or col + width > grid[1]:
        raise ValueError("Prespecified location does not fit the input")
    edited = np.zeros(grid, dtype=bool)
    edited[row : row + width, col : col + width] = True
    remote = np.ones(grid, dtype=bool)
    remote[max(0, row - guard) : min(grid[0], row + width + guard),
           max(0, col - guard) : min(grid[1], col + width + guard)] = False
    if not remote.any():
        raise ValueError("Empty remote region")
    return edited, remote


def pixel_mask(mask: np.ndarray, patch_size: int) -> np.ndarray:
    return mask.repeat(patch_size, axis=0).repeat(patch_size, axis=1)


def patch_mask(mask: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = mask.shape
    return mask.reshape(h // patch_size, patch_size, w // patch_size, patch_size).any(axis=(1, 3))


def assert_unchanged(receiver: np.ndarray, edited: np.ndarray, edit_region: np.ndarray, patch_size: int) -> None:
    outside = ~pixel_mask(edit_region, patch_size)
    if not np.array_equal(receiver[outside], edited[outside]):
        raise AssertionError("Pixels changed outside the declared edit support")
    a, b = normalized_tensor(receiver), normalized_tensor(edited)
    if not torch.equal(a[:, outside], b[:, outside]):
        raise AssertionError("Normalization changed the purportedly unchanged region")


class ExactCosineBank:
    def __init__(self, features: torch.Tensor, query_chunk: int = 256, bank_chunk: int = 4096):
        if features.ndim != 2 or not len(features) or not torch.isfinite(features).all() or (features.norm(dim=1) == 0).any():
            raise ValueError("Invalid reference features")
        self.features = torch.nn.functional.normalize(features.float(), dim=1)
        self.query_chunk, self.bank_chunk = query_chunk, bank_chunk

    @torch.inference_mode()
    def score(self, features: torch.Tensor) -> np.ndarray:
        if not torch.isfinite(features).all() or (features.norm(dim=1) == 0).any():
            raise ValueError("Invalid query features")
        query = torch.nn.functional.normalize(features.float().to(self.features.device), dim=1)
        scores = []
        for chunk in query.split(self.query_chunk):
            best = torch.full((len(chunk),), -torch.inf, device=chunk.device)
            for bank in self.features.split(self.bank_chunk):
                best = torch.maximum(best, (chunk @ bank.T).max(dim=1).values)
            scores.append((1 - best).clamp(0, 2).cpu())
        return torch.cat(scores).numpy()


def production_map(raw: np.ndarray, shape: tuple[int, int], cfg: dict) -> np.ndarray:
    # Same operations and border conventions as upstream src/utils.py:dists2map.
    resized = cv2.resize(raw.astype(np.float64), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(
        resized, sigma=cfg["postprocess_sigma_pixels"], truncate=cfg["postprocess_truncate"], mode="reflect"
    )


def calibration_threshold(maxima: list[float], alpha: float) -> float:
    """Finite-sample upper order statistic; calibration images are the units."""
    if not maxima or not 0 < alpha < 1 or not np.isfinite(maxima).all():
        raise ValueError("Invalid calibration observations")
    rank = math.ceil((len(maxima) + 1) * (1 - alpha))
    if rank > len(maxima):
        raise ValueError("Insufficient calibration images for requested nominal FPR")
    return float(np.sort(maxima)[rank - 1])


def paired_summary(rows: list[dict], key: str, cfg: dict) -> dict:
    """Average locations within receiver; bootstrap receivers, never patches/trials."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["receiver"]].append(float(row[f"defect_{key}"]) - float(row[f"sham_{key}"]))
    values = np.array([np.mean(grouped[k]) for k in sorted(grouped)], dtype=float)
    if not len(values):
        raise ValueError("No paired observations")
    rng = np.random.default_rng(cfg["seed"])
    indices = rng.integers(len(values), size=(cfg["bootstrap_samples"], len(values)))
    boot = values[indices].mean(axis=1)
    tail = (1 - cfg["confidence"]) / 2
    lo, hi = np.quantile(boot, [tail, 1 - tail])
    # The paired receiver value is bounded in [-1, 1]. A percentile bootstrap
    # falsely gives zero uncertainty when every observed pair has the same
    # outcome. Use a conservative bounded-mean interval for branch selection;
    # retain the bootstrap only as a descriptive, conditional estimate.
    radius = math.sqrt(2 * math.log(2 / (1 - cfg["confidence"])) / len(values))
    point = float(values.mean())
    return {"n_receivers": len(values), "n_trials": len(rows), "mean": float(values.mean()),
            "ci": [max(-1.0, point - radius), min(1.0, point + radius)],
            "ci_method": "Hoeffding bounded paired-receiver mean; conservative",
            "bootstrap_ci": [float(lo), float(hi)], "receiver_values": values.tolist(),
            "interval_scope": "conditional on the fixed bank, calibration threshold and sampled donor assignment"}


def classify(raw: dict, local: dict, cfg: dict, *, valid: bool, data_kind: str, reviewed: bool) -> str:
    if not valid:
        return "invalid_implementation_or_stimuli"
    if data_kind != "mvtec":
        return "engineering_only"
    if not reviewed:
        return "pending_visual_and_provenance_review"
    if raw["n_receivers"] < cfg["minimum_independent_receivers"]:
        return "inconclusive_sample_size"
    if local["ci"][0] <= 0:
        return "inconclusive_local_positive_control"
    margin = cfg["practical_remote_alarm_difference"]
    if raw["ci"][0] > margin:
        return "candidate_for_independent_replication"
    if raw["ci"][1] < margin:
        return "no_large_effect_in_this_setting"
    return "inconclusive_remote_effect"
