"""CPU-only Green Context partition arithmetic and uninitialized configuration."""

import unittest

from ditserve.runtime.green_context import GreenPool, partition_indices


class PartitionIndicesTests(unittest.TestCase):
    def test_complementary_video_and_image_partitions(self):
        self.assertEqual(partition_indices(12, 2, "video", 4, 8), [0, 1])
        self.assertEqual(partition_indices(12, 2, "image", 4, 8), [2, 3, 4, 5])
        self.assertEqual(partition_indices(12, 2, "video", 12, 0), [0, 1, 2, 3, 4, 5])
        self.assertEqual(partition_indices(12, 2, "image", 12, 0), [])

    def test_actual_a100_granularity_arithmetic(self):
        self.assertEqual(partition_indices(108, 4, "video", 16, 92), [0, 1, 2, 3])
        self.assertEqual(partition_indices(108, 4, "image", 16, 92), list(range(4, 27)))

    def test_full_card_overlap_and_zero_quota(self):
        for role in ("video", "image"):
            with self.subTest(role=role):
                self.assertEqual(partition_indices(8, 2, role, 8, 8), [0, 1, 2, 3])
                self.assertEqual(partition_indices(8, 2, role, 0, 8),
                                 [] if role == "video" else [0, 1, 2, 3])

    def test_invalid_role_negative_and_non_positive_sizes(self):
        cases = [
            (8, 2, "audio", 4, 4),
            (8, 2, None, 4, 4),
            (-8, 2, "video", -4, -4),
            (8, -2, "video", 4, 4),
            (8, 2, "video", -4, 12),
            (8, 2, "image", 12, -4),
            (0, 2, "video", 0, 0),
            (8, 0, "video", 8, 0),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                partition_indices(*arguments)

    def test_non_divisible_unbalanced_and_over_total_quotas(self):
        cases = [
            (12, 5, "video", 12, 12),
            (12, 4, "video", 6, 6),
            (12, 2, "image", 5, 7),
            (12, 4, "video", 8, 8),
            (12, 4, "image", 4, 4),
            (12, 4, "video", 16, 16),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                partition_indices(*arguments)

    def test_non_integer_arguments_are_rejected(self):
        cases = [
            (8.0, 2, "video", 4, 4),
            (8, 2.0, "video", 4, 4),
            (8, 2, "video", True, 4),
            (8, 2, "image", 4, "4"),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                partition_indices(*arguments)


class GreenPoolCpuTests(unittest.TestCase):
    def test_uninitialized_report_needs_no_cuda(self):
        pool = GreenPool("video", {"small": (4, 8), "large": (8, 4)}, "small")
        self.addCleanup(pool.close)
        report = pool.report()
        self.assertTrue(report["enabled"])
        self.assertFalse(report["initialized"])
        self.assertEqual(report["role"], "video")
        self.assertEqual(report["default"], "small")
        self.assertIsNone(report["granularity_sm"])
        self.assertIsNone(report["physical_sm"])
        self.assertEqual(report["split_group_sm"], [])
        self.assertEqual(report["split_group_fingerprints"], [])
        self.assertEqual(report["labels"], {
            "small": dict(requested_sm=4, observed_sm=None, group_indices=[]),
            "large": dict(requested_sm=8, observed_sm=None, group_indices=[]),
        })
        self.assertIn("not Nsight mask proof", report["mask_evidence"])
        self.assertTrue(report["same_process_partitions_disjoint_by_construction"])
        self.assertIsNone(report["cross_process_masks_verified"])

    def test_uninitialized_image_report_uses_image_quota(self):
        report = GreenPool("image", {"all-image": (0, 12)}, "all-image").report()
        self.assertEqual(report["labels"]["all-image"]["requested_sm"], 12)
        self.assertFalse(report["initialized"])

    def test_simple_invalid_configurations(self):
        valid = {"default": (4, 8)}
        cases = [
            ("audio", valid, "default"),
            ("video", {}, "default"),
            ("video", None, "default"),
            ("video", valid, "missing"),
            ("video", valid, None),
            ("video", {"": (4, 8)}, ""),
            ("video", {"default": (4,)}, "default"),
            ("video", {"default": (-4, 16)}, "default"),
            ("video", {"default": (True, 8)}, "default"),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                GreenPool(*arguments)


if __name__ == "__main__":
    unittest.main()
