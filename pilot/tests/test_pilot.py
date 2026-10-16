"""Validity tests that distinguish confounds, leakage, and false scientific gates."""

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image

from pilot.assess import assess
from pilot.core import (
    ExactCosineBank, alpha_mask, assert_unchanged, calibration_threshold, check_unique_files,
    classify, densest_mask_crop, paired_summary, paste, pixel_mask, production_map, regions, select_split, sha256, write_json,
)
from pilot.prepare import validate_archive


CFG = json.loads((Path(__file__).parents[1] / "protocol.json").read_text())


class ValidityTests(unittest.TestCase):
    def test_exact_cosine_matches_independent_faiss_l2_over_chunks(self):
        rng = np.random.default_rng(7)
        bank, queries = rng.normal(size=(73, 24)).astype("float32"), rng.normal(size=(51, 24)).astype("float32")
        native_bank, native_queries = bank.copy(), queries.copy()
        faiss.normalize_L2(native_bank)
        faiss.normalize_L2(native_queries)
        native = faiss.IndexFlatL2(24)
        native.add(native_bank)
        expected = native.search(native_queries, 1)[0][:, 0] / 2
        result = ExactCosineBank(torch.from_numpy(bank), query_chunk=7, bank_chunk=11).score(torch.from_numpy(queries))
        np.testing.assert_allclose(result, expected, atol=2e-6, rtol=0)
        self.assertLess(ExactCosineBank(torch.from_numpy(bank)).score(torch.from_numpy(bank)).max(), 1e-6)

    def test_zero_feature_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            ExactCosineBank(torch.zeros(3, 4))

    def test_paired_edits_preserve_outside_pixels_and_normalization(self):
        rng = np.random.default_rng(5)
        base = rng.integers(0, 256, (448, 448, 3), dtype=np.uint8)
        crop = rng.integers(0, 256, (56, 56, 3), dtype=np.uint8)
        for construction in CFG["constructions"]:
            support, remote = regions((448, 448), 8, 8, CFG)
            alpha = alpha_mask(56, construction, 7)
            result = paste(base, crop, 112, 112, alpha)
            assert_unchanged(base, result, support, 14)
            self.assertTrue(np.array_equal(result[pixel_mask(remote, 14)], base[pixel_mask(remote, 14)]))
            identity = paste(base, base[112:168, 112:168], 112, 112, alpha)
            np.testing.assert_array_equal(base, identity)
        result[0, 0] = 255 - base[0, 0]
        with self.assertRaises(AssertionError):
            assert_unchanged(base, result, support, 14)

    def test_guard_excludes_interpolation_and_gaussian_spread(self):
        support, remote = regions((448, 448), 8, 8, CFG)
        impulse = support.astype("float32")
        processed = production_map(impulse, (448, 448), CFG)
        self.assertEqual(float(processed[pixel_mask(remote, 14)].max()), 0.0)
        unsafe = dict(CFG, guard_patches=0)
        _, no_guard = regions((448, 448), 8, 8, unsafe)
        self.assertGreater(float(processed[pixel_mask(no_guard, 14)].max()), 0.0)

    def test_calibration_uses_image_maxima_finite_sample_order_statistic(self):
        self.assertEqual(calibration_threshold(list(range(64)), .05), 61)
        with self.assertRaises(ValueError):
            calibration_threshold([.1] * 10, .05)

    def test_bootstrap_clusters_locations_not_patches(self):
        base = [{"receiver": "a", "defect_alarm": 1, "sham_alarm": 0},
                {"receiver": "b", "defect_alarm": 0, "sham_alarm": 1}]
        first = paired_summary(base, "alarm", CFG)
        replicated = paired_summary(base * 20, "alarm", CFG)
        self.assertEqual(first["n_receivers"], 2)
        self.assertEqual(first["ci"], replicated["ci"])
        self.assertEqual(first["bootstrap_ci"], replicated["bootstrap_ci"])

    def test_identical_observations_do_not_create_zero_width_gate_interval(self):
        rows = [{"receiver": str(i), "defect_alarm": 0, "sham_alarm": 0} for i in range(21)]
        result = paired_summary(rows, "alarm", CFG)
        self.assertEqual(result["bootstrap_ci"], [0, 0])
        self.assertGreater(result["ci"][1], CFG["practical_remote_alarm_difference"])

    def test_branch_classification_is_fail_closed(self):
        positive = {"n_receivers": 40, "ci": [.2, .4]}
        local = {"ci": [.1, .5]}
        options = dict(valid=True, data_kind="mvtec", reviewed=True)
        self.assertEqual(classify(positive, local, CFG, **options), "candidate_for_independent_replication")
        self.assertEqual(classify(positive, local, CFG, **dict(options, data_kind="procedural")), "engineering_only")
        self.assertEqual(classify(positive, local, CFG, **dict(options, reviewed=False)), "pending_visual_and_provenance_review")
        self.assertEqual(classify(positive, local, CFG, **dict(options, valid=False)), "invalid_implementation_or_stimuli")
        self.assertEqual(classify({"n_receivers": 40, "ci": [-.1, .3]}, local, CFG, **options), "inconclusive_remote_effect")
        self.assertEqual(classify({"n_receivers": 40, "ci": [-.05, .05]}, local, CFG, **options), "no_large_effect_in_this_setting")
        self.assertEqual(classify(positive, {"ci": [-.1, .5]}, CFG, **options), "inconclusive_local_positive_control")

    def test_prescore_crop_uses_ground_truth(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[20:24, 25:29] = True
        top, left = densest_mask_crop(mask, 8)
        self.assertEqual(mask[top:top + 8, left:left + 8].sum(), 16)

    def test_split_and_content_leakage_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("grid/train/good", "grid/test/good", "grid/test/bent"):
                (root / sub).mkdir(parents=True)
            for i in range(10):
                Image.fromarray(np.full((3, 3, 3), i, dtype=np.uint8)).save(root / "grid/train/good" / f"{i:03}.png")
            for sub in ("good", "bent"):
                for i in range(2):
                    Image.fromarray(np.full((3, 3, 3), 20 + i, dtype=np.uint8)).save(root / f"grid/test/{sub}/{i:03}.png")
            cfg = dict(CFG, reference_images=2, calibration_images=3)
            split = select_split(root, "grid", cfg)
            roles = [set(split[k]) for k in ("bank", "calibration", "clean_donors", "receivers")]
            for i, first in enumerate(roles):
                for second in roles[i + 1:]:
                    self.assertFalse(first & second)
            with self.assertRaises(ValueError):
                check_unique_files({"receivers": split["receivers"], "donors": split["defect_candidates"]})

    def test_block_pages_and_unsafe_archives_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.tar.xz"
            path.write_text("<html>Access Blocked - Internet Security Policy</html>")
            with self.assertRaises(PermissionError):
                validate_archive(path)
            with tarfile.open(path, "w:xz") as archive:
                member = tarfile.TarInfo("../escape")
                member.size = 3
                archive.addfile(member, io.BytesIO(b"bad"))
            with self.assertRaises(ValueError):
                validate_archive(path)

    def test_assessment_binds_review_to_run_and_cannot_promote_procedural_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "status.json", {"state": "blocked", "reason": "data absent"})
            self.assertEqual(assess(root)["decision"], "blocked")
            write_json(root / "status.json", {"state": "complete_engineering_only"})
            write_json(root / "manifest.json", {"protocol": CFG, "kind": "procedural"})
            digest = sha256(root / "manifest.json")
            write_json(root / "summary.json", {
                "manifest_sha256": digest, "integrity_passed": True, "additional_gates": ["real data"],
                "summaries": {"grid/hard": {"raw_remote": {"n_receivers": 40, "ci": [.2, .4]},
                                            "local": {"ci": [.1, .5]}}}
            })
            write_json(root / "review-template.json", {"manifest_sha256": digest, "required_files": ["gallery.png"]})
            review = {"manifest_sha256": digest, "reviewer": "test", "inspected_files": ["gallery.png"],
                      "data_provenance_verified": True, "all_edits_plausible": True,
                      "largest_effects_and_controls_reviewed": True, "no_postscore_exclusions": True}
            result = assess(root, review)
            self.assertEqual(result["decision"], "engineering_only")
            self.assertTrue(result["visual_inspection_recorded"])
            self.assertTrue(result["review_gate_passed"])
            with self.assertRaises(ValueError):
                assess(root, dict(review, manifest_sha256="wrong-run"))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
