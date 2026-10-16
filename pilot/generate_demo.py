"""Generate engineering fixtures. These are never a substitute for MVTec evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .core import write_json


def texture(rng: np.random.Generator, category: str, size: int) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    dx, dy = rng.uniform(-10, 10, 2)
    if category == "grid":
        value = 150 + 27 * np.cos((x + dx) * 2 * np.pi / 28) * np.cos((y + dy) * 2 * np.pi / 28)
    else:
        bx, by = np.floor((x + dx) / 56), np.floor((y + dy) / 56)
        value = 145 + 15 * ((bx + by) % 2) + 5 * np.sin(x / 13) + 4 * np.cos(y / 19)
    value += rng.normal(0, 2, (size, size)) + rng.uniform(-5, 5)
    rgb = np.stack([value + 2, value, value - 2], axis=-1)
    return np.rint(rgb).clip(0, 255).astype(np.uint8)


def generate(root: Path, *, size: int = 448, receivers: int = 12, seed: int = 712) -> None:
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for category in ("grid", "tile"):
        for sub in ("train/good", "test/good", "test/scratch", "ground_truth/scratch"):
            (root / category / sub).mkdir(parents=True)
        for i in range(128):
            Image.fromarray(texture(rng, category, size)).save(root / category / "train/good" / f"{i:03d}.png")
        for i in range(receivers):
            Image.fromarray(texture(rng, category, size)).save(root / category / "test/good" / f"{i:03d}.png")
            image = texture(rng, category, size)
            mask = np.zeros((size, size), dtype=np.uint8)
            top, left = rng.integers(size // 4, size * 3 // 4 - 40, size=2)
            mask[top : top + 35, left : left + 7] = 255
            image[mask > 0] = rng.integers(15, 40)
            Image.fromarray(image).save(root / category / "test/scratch" / f"{i:03d}.png")
            Image.fromarray(mask).save(root / category / "ground_truth/scratch" / f"{i:03d}_mask.png")
    write_json(root / "_procedural.json", {"kind": "engineering_fixture", "seed": seed, "size": size,
                                          "receivers_per_category": receivers, "not_mvtec": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receivers", type=int, default=12)
    args = parser.parse_args()
    generate(args.output, receivers=args.receivers)
