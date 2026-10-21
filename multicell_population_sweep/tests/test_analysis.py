import unittest

import numpy as np

from multicell_shared_pool.analysis import (
    cross_reporter_population_correlation,
    summarize_condition,
    summarize_reporter,
)


class ReporterAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.times = np.array([0, 1, 2])
        self.values = np.array(
            [
                [[1, 2, 3], [3, 4, 5]],
                [[2, 3, 4], [4, 5, 6]],
            ],
            dtype=float,
        )

    def test_reporter_summary_matches_hand_calculated_statistics(self):
        summary = summarize_reporter(
            self.values,
            self.times,
            window=(1, 2),
            valid_replicates=np.array([True, True]),
        )

        self.assertAlmostEqual(summary["single_cell_mean"], 4.0)
        self.assertAlmostEqual(summary["single_cell_variance"], 12 / 7)
        self.assertAlmostEqual(summary["single_cell_sd"], np.sqrt(12 / 7))
        self.assertAlmostEqual(summary["single_cell_cv2"], (12 / 7) / 16)
        self.assertAlmostEqual(summary["single_cell_fano"], (12 / 7) / 4)
        self.assertEqual(summary["single_cell_n"], 8)

        self.assertAlmostEqual(summary["population_mean_mean"], 4.0)
        self.assertAlmostEqual(summary["population_mean_variance"], 2 / 3)
        self.assertAlmostEqual(summary["population_mean_cv2"], (2 / 3) / 16)
        self.assertAlmostEqual(summary["population_mean_fano"], (2 / 3) / 4)
        self.assertAlmostEqual(summary["within_population_cv2_mean"], (2 / 9 + 1 / 8 + 1 / 8 + 2 / 25) / 4)
        self.assertAlmostEqual(summary["mean_cell_correlation"], 1.0)

    def test_invalid_replicates_are_excluded(self):
        summary = summarize_reporter(
            self.values,
            self.times,
            window=(1, 2),
            valid_replicates=np.array([True, False]),
        )

        self.assertEqual(summary["single_cell_n"], 4)
        self.assertAlmostEqual(summary["single_cell_mean"], 3.5)
        self.assertEqual(summary["valid_replicates"], 1)

    def test_zero_mean_returns_nan_relative_metrics(self):
        zeros = np.zeros((2, 2, 3), dtype=float)

        summary = summarize_reporter(
            zeros,
            self.times,
            window=(1, 2),
            valid_replicates=np.array([True, True]),
        )

        self.assertTrue(np.isnan(summary["single_cell_cv2"]))
        self.assertTrue(np.isnan(summary["single_cell_fano"]))
        self.assertTrue(np.isnan(summary["population_mean_cv2"]))
        self.assertTrue(np.isnan(summary["within_population_cv2_mean"]))

    def test_population_correlation_uses_reporters_separately(self):
        correlation = cross_reporter_population_correlation(
            self.values,
            self.values * 2,
            self.times,
            window=(1, 2),
            valid_replicates=np.array([True, True]),
        )

        self.assertAlmostEqual(correlation, 1.0)

    def test_correlations_are_computed_within_replicates_before_averaging(self):
        values = np.array(
            [
                [[1, 2], [2, 1]],
                [[10, 11], [11, 10]],
            ],
            dtype=float,
        )
        summary = summarize_reporter(
            values,
            np.array([0, 1]),
            window=(0, 1),
            valid_replicates=np.array([True, True]),
        )
        first_reporter = np.array(
            [
                [[1, 2], [1, 2]],
                [[10, 11], [10, 11]],
            ],
            dtype=float,
        )
        second_reporter = np.array(
            [
                [[2, 1], [2, 1]],
                [[11, 10], [11, 10]],
            ],
            dtype=float,
        )
        cross = cross_reporter_population_correlation(
            first_reporter,
            second_reporter,
            np.array([0, 1]),
            window=(0, 1),
            valid_replicates=np.array([True, True]),
        )

        self.assertAlmostEqual(summary["mean_cell_correlation"], -1.0)
        self.assertAlmostEqual(cross, -1.0)

    def test_condition_summary_preserves_reporter_labels(self):
        rows = summarize_condition(
            reporter_data={"pLux_eGFP": self.values, "pLas_reporter": self.values * 2},
            reporter_labels={"pLux_eGFP": "pLux-eGFP", "pLas_reporter": "pLas-mScarlet"},
            times=self.times,
            window=(1, 2),
            valid_replicates=np.array([True, True]),
        )

        self.assertEqual([row["reporter_label"] for row in rows], ["pLux-eGFP", "pLas-mScarlet"])
        self.assertTrue(all("population_mean_cv2" in row for row in rows))
        self.assertAlmostEqual(rows[0]["cross_reporter_population_correlation"], 1.0)
        self.assertAlmostEqual(rows[1]["cross_reporter_population_correlation"], 1.0)


if __name__ == "__main__":
    unittest.main()
