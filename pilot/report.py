"""Inspection artifacts with common color scales and explicit selection labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw


def save_example(path: Path, payload: dict, cfg: dict, thresholds: dict, *, kind: str, selection: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    p, width = cfg["patch_size"], cfg["edit_patches"]
    row, col = payload["location"]
    for i, arm in enumerate(("identity", "sham", "defect")):
        axes[i, 0].imshow(payload["images"][arm])
        axes[i, 0].add_patch(Rectangle((col * p, row * p), width * p, width * p, fill=False, edgecolor="magenta"))
        axes[i, 0].set_title(arm)
        for j, stage in enumerate(("raw", "processed"), start=1):
            data = payload[stage][arm]
            maximum = 1.5 * thresholds[stage]
            im = axes[i, j].imshow(data, vmin=0, vmax=maximum, cmap="magma", interpolation="nearest")
            mask = payload["remote"] if stage == "raw" else payload["remote"].repeat(p, 0).repeat(p, 1)
            axes[i, j].contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
            fraction = float(np.mean(data[mask] > thresholds[stage]))
            axes[i, j].set_title(f"{stage}: remote fraction > threshold = {fraction:.3f}")
            fig.colorbar(im, ax=axes[i, j], fraction=0.04)
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(
        f"{kind.upper()} — {payload['category']} — {payload['receiver']}\n"
        f"{selection}; {payload['construction']}; cyan = remote boundary; magenta = edit support\n"
        "Common scale across arms; saturation does not change recorded numeric scores",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def thumbnail(image: np.ndarray, label: str) -> Image.Image:
    thumb = Image.fromarray(image).resize((112, 112), Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (116, 138), "white")
    tile.paste(thumb, (2, 20))
    ImageDraw.Draw(tile).text((2, 3), label, fill="black")
    return tile


def save_gallery(path: Path, rows: list[list[Image.Image]], category: str, kind: str) -> None:
    columns = max(len(row) for row in rows)
    canvas = Image.new("RGB", (116 * columns, 44 + 138 * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), f"{kind.upper()} / {category}: ALL receivers, locations and constructions", fill="black")
    draw.text((6, 22), "Each row: base, then sham/defect pairs in protocol order. No score-based omissions.", fill="black")
    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            canvas.paste(tile, (116 * x, 44 + 138 * y))
    canvas.save(path)


def save_effects(path: Path, summaries: dict, cfg: dict, kind: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = []
    for i, (name, values) in enumerate(summaries.items()):
        raw = values["raw_remote"]
        point = raw["mean"]
        ax.errorbar(point, i, xerr=[[max(0, point - raw["ci"][0])], [max(0, raw["ci"][1] - point)]],
                    fmt="o", capsize=4)
        labels.append(f"{name} (n={raw['n_receivers']} independent receivers)")
    ax.axvline(0, color="gray")
    ax.axvline(cfg["practical_remote_alarm_difference"], color="red", linestyle="--", label="pilot relevance margin")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Remote alarm probability: defect minus sham (95% bounded-mean gate interval)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(f"{kind.upper()} — raw endpoint; conditional on bank/calibration/donor assignment")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
