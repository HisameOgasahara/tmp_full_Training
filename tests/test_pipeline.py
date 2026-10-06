"""Exercise stage handoffs, interactive GRPO, resume, and held-out evaluation."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
import nbformat
from maze_training.runtime import read_config, load_model
from maze_training.training import run_stage, normalize_group_rewards
from maze_training.evaluation import evaluate_world, evaluate_navigation
from maze_training.data import build_maps
from maze_training.encoding import MazeEncoder


class PipelineTests(unittest.TestCase):
    def test_equal_rewards_have_no_advantage_or_learning_signal(self):
        advantages, useful = normalize_group_rewards(torch.tensor([[1., 1., 1., 1.], [0., 1., 0., 1.]]), 1e-6)
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertEqual(useful.tolist(), [False, True])
        self.assertEqual(advantages[0].abs().sum().item(), 0)
        self.assertAlmostEqual(advantages[1].mean().item(), 0)

    def test_map_splits_and_query_lengths(self):
        config = read_config("configs/smoke.json")
        pools = [build_maps(config, split) for split in ("train", "validation", "test")]
        seed_sets = [{maze.seed for maze in maps} for maps in pools]
        self.assertFalse(seed_sets[0] & seed_sets[1] or seed_sets[0] & seed_sets[2] or seed_sets[1] & seed_sets[2])
        encoder = MazeEncoder(config["environment"]["max_size"])
        maze = pools[0][0]
        self.assertEqual(len(encoder.encode_query(maze, maze.start, 0)), encoder.context_length)

    def test_all_stages_resume_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            full, resumed = directory / "full", directory / "resumed"
            run_stage("pretrain", "configs/smoke.json", full)
            run_stage("pretrain", "configs/smoke.json", resumed, steps=2)
            run_stage("pretrain", "configs/smoke.json", resumed, resume=True)
            _, _, full_payload = load_model(full / "pretrain.pt", torch.device("cpu"))
            _, _, resumed_payload = load_model(resumed / "pretrain.pt", torch.device("cpu"))
            for name, value in full_payload["model"].items():
                self.assertTrue(torch.equal(value, resumed_payload["model"][name]), name)
            run_stage("sft", "configs/smoke.json", full)
            run_stage("grpo", "configs/smoke.json", full)
            run_stage("grpo", "configs/smoke.json", resumed, steps=1, initialize_from=full / "sft.pt")
            run_stage("grpo", "configs/smoke.json", resumed, resume=True)
            sft, encoder, payload = load_model(full / "sft.pt", torch.device("cpu"))
            grpo, _, grpo_payload = load_model(full / "grpo.pt", torch.device("cpu"))
            _, _, resumed_grpo_payload = load_model(resumed / "grpo.pt", torch.device("cpu"))
            for name, value in grpo_payload["model"].items():
                self.assertTrue(torch.equal(value, resumed_grpo_payload["model"][name]), name)
            world_model, _, _ = load_model(full / "pretrain.pt", torch.device("cpu"))
            world = evaluate_world(world_model, encoder, payload["config"])
            self.assertTrue(0 <= world["total_variation_distance"] <= 1)
            metrics = evaluate_navigation(payload["config"], {"sft": (sft, encoder), "grpo": (grpo, encoder), "BFS": "bfs", "optimal": "optimal"})
            for result in metrics["policies"].values():
                self.assertAlmostEqual(result["success_rate"] + result["hole_rate"] + result["timeout_rate"], 1)
            dry = evaluate_navigation(payload["config"], {"BFS": "bfs"}, slippery=False)
            self.assertEqual(dry["policies"]["BFS"]["success_rate"], 1)
            json.dumps(metrics, allow_nan=False)

    def test_notebook_is_clean_and_code_cells_compile(self):
        notebook = nbformat.read("notebooks/Maze_Training.ipynb", as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
                compile(cell.source, f"cell_{index}", "exec")


if __name__ == "__main__":
    unittest.main()
