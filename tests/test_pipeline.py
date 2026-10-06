"""Verify stage handoffs, GAE episode boundaries, exact resume and held-out costs."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
import nbformat
from maze_training.runtime import read_config, load_model
from maze_training.training import run_stage, compute_gae
from maze_training.evaluation import evaluate_world, evaluate_navigation
from maze_training.data import build_maps
from maze_training.encoding import MazeEncoder


class PipelineTests(unittest.TestCase):
    def test_gae_matches_complete_episode_returns(self):
        advantages, returns = compute_gae(torch.tensor([-1.,2.]), torch.tensor([0.5,0.25]), 1.0, 1.0)
        torch.testing.assert_close(returns, torch.tensor([1.,2.]))
        torch.testing.assert_close(advantages, torch.tensor([0.5,1.75]))
        one_step, target = compute_gae(torch.tensor([-0.3]), torch.tensor([0.7]), 1.0, 0.95)
        torch.testing.assert_close(one_step, torch.tensor([-1.]))
        torch.testing.assert_close(target, torch.tensor([-0.3]))

    def test_map_splits_and_query_lengths(self):
        config = read_config("configs/smoke.json")
        pools = [build_maps(config, split) for split in ("train","validation","test")]
        seeds = [{maze.seed for maze in maps} for maps in pools]
        self.assertFalse(seeds[0]&seeds[1] or seeds[0]&seeds[2] or seeds[1]&seeds[2])
        encoder = MazeEncoder(config["environment"]["max_size"])
        self.assertEqual(len(encoder.encode_query(pools[0][0], pools[0][0].start, 0)), encoder.context_length)

    def test_all_stages_resume_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            full, resumed = directory/"full", directory/"resumed"
            run_stage("pretrain","configs/smoke.json",full)
            run_stage("pretrain","configs/smoke.json",resumed,steps=2)
            run_stage("pretrain","configs/smoke.json",resumed,resume=True)
            _,_,a = load_model(full/"pretrain.pt",torch.device("cpu"))
            _,_,b = load_model(resumed/"pretrain.pt",torch.device("cpu"))
            for name,value in a["model"].items():
                self.assertTrue(torch.equal(value,b["model"][name]),name)
            run_stage("sft","configs/smoke.json",full)
            run_stage("ppo","configs/smoke.json",full)
            run_stage("ppo","configs/smoke.json",resumed,steps=1,initialize_from=full/"sft.pt")
            run_stage("ppo","configs/smoke.json",resumed,resume=True)
            sft,encoder,payload = load_model(full/"sft.pt",torch.device("cpu"))
            ppo,_,a = load_model(full/"ppo.pt",torch.device("cpu"))
            _,_,b = load_model(resumed/"ppo.pt",torch.device("cpu"))
            for name,value in a["model"].items():
                self.assertTrue(torch.equal(value,b["model"][name]),name)
            self.assertFalse(torch.equal(payload["model"]["predict_value.weight"],a["model"]["predict_value.weight"]))
            world_model,_,_ = load_model(full/"pretrain.pt",torch.device("cpu"))
            world = evaluate_world(world_model,encoder,payload["config"])
            self.assertTrue(0<=world["next_position_accuracy"]<=1)
            self.assertGreaterEqual(world["cost_mae"],0)
            metrics = evaluate_navigation(payload["config"],{"SFT":(sft,encoder),"PPO":(ppo,encoder),"BFS":"bfs","Dijkstra":"dijkstra"})
            self.assertEqual(metrics["policies"]["BFS"]["success_rate"],1)
            self.assertEqual(metrics["policies"]["Dijkstra"]["success_rate"],1)
            self.assertAlmostEqual(metrics["policies"]["Dijkstra"]["mean_excess_cost_success"],0)
            self.assertGreater(metrics["policies"]["BFS"]["mean_cost_success"],metrics["policies"]["Dijkstra"]["mean_cost_success"])
            for result in metrics["policies"].values():
                self.assertAlmostEqual(result["success_rate"]+result["timeout_rate"],1)
            json.dumps(metrics,allow_nan=False)

    def test_notebook_is_clean_and_code_cells_compile(self):
        notebook = nbformat.read("notebooks/Maze_Training.ipynb",as_version=4)
        nbformat.validate(notebook)
        for index,cell in enumerate(notebook.cells):
            if cell.cell_type=="code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs,[])
                compile(cell.source,f"cell_{index}","exec")


if __name__ == "__main__":
    unittest.main()
