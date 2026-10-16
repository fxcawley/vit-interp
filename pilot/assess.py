"""Apply the prespecified pilot branch after documented inspection, without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import classify, sha256, write_json


def assess(run: Path, review: dict | None = None) -> dict:
    status = json.loads((run / "status.json").read_text())
    if not status["state"].startswith("complete_"):
        return {"decision": status["state"], "scientific_claim": "none", "reason": status.get("reason", status.get("error"))}
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    template = json.loads((run / "review-template.json").read_text())
    digest = sha256(run / "manifest.json")
    if summary["manifest_sha256"] != digest or template["manifest_sha256"] != digest:
        raise ValueError("Manifest mismatch; assessment does not apply to this run")
    reviewed = False
    inspection_recorded = False
    valid = summary["integrity_passed"]
    if review:
        if review.get("manifest_sha256") != digest:
            raise ValueError("Review belongs to a different run")
        inspection_recorded = (
            bool(review.get("reviewer", "").strip())
            and set(template["required_files"]).issubset(review.get("inspected_files", []))
        )
        reviewed = inspection_recorded and all(review.get(k) is True for k in (
                "data_provenance_verified", "all_edits_plausible",
                "largest_effects_and_controls_reviewed", "no_postscore_exclusions"
            ))
        if manifest["kind"] == "mvtec" and inspection_recorded and review.get("all_edits_plausible") is False:
            valid = False
    cfg = manifest["protocol"]
    decisions = {}
    for name, result in summary["summaries"].items():
        decisions[name] = classify(result["raw_remote"], result["local"], cfg,
                                   valid=valid, data_kind=manifest["kind"], reviewed=reviewed)
    primary = f"{cfg['primary_category']}/{cfg['primary_construction']}"
    return {"decision": decisions[primary], "primary": primary, "all_conditions": decisions,
            "visual_inspection_recorded": inspection_recorded, "review_gate_passed": reviewed,
            "scope": "pilot spending decision only",
            "unmet_publication_gates": summary["additional_gates"], "manifest_sha256": digest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    decision = assess(args.run, json.loads(args.review.read_text()) if args.review else None)
    if args.output:
        write_json(args.output, decision)
    print(json.dumps(decision, indent=2))
