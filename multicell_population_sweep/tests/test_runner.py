import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from multicell_shared_pool.config import SimulationConfig, derive_seed
from multicell_shared_pool.runner import (
    aggregate_outputs,
    build_parser,
    build_replicate_tasks,
    run_condition_copy,
)


class PopulationRunnerTests(unittest.TestCase):
    def test_cli_defaults_match_approved_run(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.conditions, [
            "autocrine_nc3",
            "paracrine_c3",
            "autocrine_nc6",
            "paracrine_c6",
        ])
        self.assertEqual(args.copies, [5, 40])
        self.assertEqual(args.N, 500)
        self.assertEqual(args.cells, 10)
        self.assertEqual(args.T, 3500)
        self.assertEqual((args.w0, args.w1), (3000, 3500))
        self.assertEqual(args.tau_ahl, 240.0)
        self.assertEqual(args.tau_s, 240.0)
        self.assertTrue(args.resume)

    def test_replicate_tasks_use_order_independent_seed_identity(self):
        config = SimulationConfig(condition="paracrine_c6", copies=40)
        tasks = build_replicate_tasks(config, base_seed=20261019, replicate_ids=[2, 0, 1])

        self.assertEqual([task.replicate_id for task in tasks], [2, 0, 1])
        self.assertEqual(tasks[1].seed, derive_seed(20261019, "paracrine_c6", 40, 0))
        self.assertEqual(tasks[2].seed, derive_seed(20261019, "paracrine_c6", 40, 1))

    def test_small_run_writes_expected_outputs_and_resumes_without_duplicates(self):
        config = SimulationConfig(
            condition="autocrine_nc3",
            copies=5,
            cells_per_population=2,
            volume_scale=2,
            t_end=10,
            window=(5, 10),
            max_events=100_000,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            outdir = Path(temporary_directory)
            run_condition_copy(
                config=config,
                N=2,
                base_seed=100,
                outdir=outdir,
                batch_size=1,
                workers=1,
                resume=True,
            )
            first_qc = pd.read_csv(outdir / "run_qc.csv")

            run_condition_copy(
                config=config,
                N=2,
                base_seed=100,
                outdir=outdir,
                batch_size=1,
                workers=1,
                resume=True,
            )
            second_qc = pd.read_csv(outdir / "run_qc.csv")

            for filename in (
                "summary.csv",
                "population_mean_timelapse.csv",
                "steady_cell_time_means.csv",
                "within_population_noise.csv",
                "run_qc.csv",
                "metadata.json",
                "steady_cell_trajectories.npz",
            ):
                self.assertTrue((outdir / filename).exists(), filename)
            pd.testing.assert_frame_equal(first_qc, second_qc)
            self.assertEqual(second_qc["replicate_id"].tolist(), [0, 1])
            metadata = json.loads((outdir / "metadata.json").read_text())
            self.assertEqual(metadata["cells_per_population"], 2)
            self.assertEqual(metadata["copies"], 5)

    def test_resume_rejects_incompatible_scientific_metadata(self):
        first = SimulationConfig(
            condition="autocrine_nc3",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
            t_end=5,
            window=(2, 5),
        )
        incompatible = SimulationConfig(
            condition="autocrine_nc3",
            copies=40,
            cells_per_population=1,
            volume_scale=1,
            t_end=5,
            window=(2, 5),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            outdir = Path(temporary_directory)
            run_condition_copy(first, 1, 7, outdir, batch_size=1, workers=1, resume=True)
            with self.assertRaises(ValueError):
                run_condition_copy(incompatible, 1, 7, outdir, batch_size=1, workers=1, resume=True)

    def test_duplicate_replicate_ids_fail_aggregation(self):
        config = SimulationConfig(
            condition="autocrine_nc3",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
            t_end=5,
            window=(2, 5),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            outdir = Path(temporary_directory)
            run_condition_copy(config, 1, 9, outdir, batch_size=1, workers=1, resume=True)
            original = next((outdir / "batches").glob("batch_*.npz"))
            duplicate = original.with_name("batch_duplicate.npz")
            duplicate.write_bytes(original.read_bytes())

            with self.assertRaises(ValueError):
                aggregate_outputs(outdir, config, expected_N=1)

    def test_event_capped_replicates_are_excluded_from_summary(self):
        config = SimulationConfig(
            condition="autocrine_nc3",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
            t_end=10,
            window=(5, 10),
            max_events=1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            outdir = Path(temporary_directory)
            run_condition_copy(config, 2, 12, outdir, batch_size=2, workers=1, resume=True)

            summary = pd.read_csv(outdir / "summary.csv")
            qc = pd.read_csv(outdir / "run_qc.csv")
            self.assertEqual(summary.loc[0, "valid_replicates"], 0)
            self.assertTrue(qc["hit_event_cap"].all())


if __name__ == "__main__":
    unittest.main()
