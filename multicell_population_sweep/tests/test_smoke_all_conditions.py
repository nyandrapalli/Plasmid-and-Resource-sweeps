import tempfile
import unittest
from pathlib import Path

import pandas as pd

from multicell_shared_pool.config import CONDITIONS
from multicell_shared_pool.runner import main


class AllConditionSmokeTests(unittest.TestCase):
    def test_all_conditions_and_copy_numbers_complete_with_valid_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "smoke_outputs"
            main([
                "--smoke",
                "--out",
                str(output_root),
                "--batch-size",
                "1",
                "--seed",
                "4242",
            ])

            combined = pd.read_csv(output_root / "all_conditions_summary.csv")
            self.assertEqual(len(combined), 12)
            self.assertEqual(set(combined["condition"]), set(CONDITIONS))
            self.assertEqual(set(combined["copies"]), {5, 40})

            for condition in CONDITIONS:
                for copies in (5, 40):
                    directory = output_root / condition / f"copies_{copies}"
                    qc = pd.read_csv(directory / "run_qc.csv")
                    self.assertEqual(qc["replicate_id"].tolist(), [0, 1])
                    self.assertFalse(qc["hit_event_cap"].any())
                    self.assertTrue(qc["valid_replicate"].all())
                    self.assertTrue((directory / "summary.csv").exists())
                    self.assertTrue((directory / "steady_cell_trajectories.npz").exists())

            auto_nc6 = pd.read_csv(output_root / "autocrine_nc6" / "copies_5" / "summary.csv")
            para_c6 = pd.read_csv(output_root / "paracrine_c6" / "copies_5" / "summary.csv")
            self.assertEqual(set(auto_nc6["reporter_label"]), {"pLux-eGFP", "pLas-mScarlet"})
            self.assertEqual(set(para_c6["reporter_label"]), {"pLux-eGFP", "pLas-eGFP"})


if __name__ == "__main__":
    unittest.main()
