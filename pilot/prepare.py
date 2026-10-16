"""Validate and extract a locally supplied, approved MVTec category archive."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path, PurePosixPath

from .core import sha256, write_json


def validate_archive(path: Path) -> None:
    with path.open("rb") as stream:
        head = stream.read(32768)
    text = head.lower()
    if b"access blocked" in text or b"security policy" in text:
        raise PermissionError("Downloaded content is a security-policy block page, not a dataset. Obtain approved access.")
    if not head.startswith(b"\xfd7zXZ\x00"):
        raise ValueError("Expected an XZ archive; HTML, partial downloads and other formats are rejected")
    with tarfile.open(path, "r:xz") as archive:
        for member in archive:
            posix = PurePosixPath(member.name)
            if posix.is_absolute() or ".." in posix.parts or "\\" in member.name or ":" in member.name:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive links/devices are not allowed: {member.name}")


def extract(path: Path, output: Path) -> None:
    validate_archive(path)
    if output.exists():
        raise FileExistsError(f"Use a fresh extraction destination: {output}")
    output.mkdir(parents=True)
    with tarfile.open(path, "r:xz") as archive:
        archive.extractall(output, filter="data")
    write_json(output / "archive-provenance.json", {"archive": path.name, "sha256": sha256(path),
                                                    "source": "locally supplied; verify against assets.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    extract(args.archive, args.output)
