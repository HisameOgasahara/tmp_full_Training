"""Verify stage handoffs, GAE episode boundaries, exact resume and held-out costs."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
import numpy as np
from unittest.mock import patch
import nbformat
from maze_training.runtime import read_config, load_model
from maze_training.training import run_stage, compute_gae, collect_rollouts, warm_value_head, shortest_action_loss
from maze_training.evaluation import evaluate_world, evaluate_navigation
from maze_training.data import build_maps, build_expert_examples
from maze_training.policy import build_action_masks, mask_action_logits
from maze_training.environment import Maze, FLOOR, GOAL, compute_goal_distances
from maze_training.runtime import create_model, seed_runtime
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

    def test_masks_and_shortest_path_ties(self):
        config=read_config("configs/smoke.json")
        grid=np.full((3,3),FLOOR);grid[0,2]=GOAL
        maze=Maze(grid,(2,0),(0,2),0)
        encoder=MazeEncoder(5)
        queries,targets=build_expert_examples([maze],encoder)
        index=next(i for i,q in enumerate(queries) if tuple(q)==tuple(encoder.encode_query(maze,maze.start)))
        self.assertEqual(targets[index].tolist(),[True,True,False,False])
        masks=build_action_masks([maze],[maze.start])
        np.testing.assert_array_equal(masks,encoder.decode_action_masks([encoder.encode_query(maze,maze.start)]))
        logits=mask_action_logits(torch.tensor([[0.,0.,100.,100.]]),masks)
        probs=logits.softmax(-1)
        self.assertEqual(probs[0,2:].sum().item(),0)
        self.assertLess(shortest_action_loss(logits,torch.tensor(targets[index:index+1])).item(),1e-6)

    def test_critic_warmup_keeps_actor_and_rollout_masks_unchanged(self):
        config=read_config("configs/smoke.json")
        seed_runtime(config["seed"])
        model,encoder=create_model(config,torch.device("cpu"))
        reference, _=create_model(config,torch.device("cpu"))
        reference.load_state_dict(model.state_dict());reference.eval()
        maps=build_maps(config,"train")
        rollout,lengths,_=collect_rollouts(model,encoder,maps,config,np.random.default_rng(9),reference)
        self.assertTrue(rollout["masks"].gather(1,rollout["actions"][:,None]).all())
        logits=mask_action_logits(model(rollout["queries"])[:,encoder.action_slice],rollout["masks"])
        log_probs=logits.log_softmax(-1).gather(1,rollout["actions"][:,None]).squeeze(1)
        torch.testing.assert_close(log_probs,rollout["log_probs"])
        actor={name:p.detach().clone() for name,p in model.named_parameters() if not name.startswith("predict_value.")}
        optimizer=torch.optim.AdamW(model.parameters(),lr=config["ppo"]["critic_learning_rate"])
        scaler=torch.amp.GradScaler("cuda",enabled=False)
        warm_value_head(model,rollout,lengths,config["ppo"],optimizer,scaler,1.0)
        for name,value in actor.items():
            self.assertTrue(torch.equal(value,dict(model.named_parameters())[name]),name)
        self.assertGreater(model.predict_value.weight.abs().sum().item(),0)

    def test_potential_keeps_goal_and_timeout_objective(self):
        config=read_config("configs/smoke.json")
        from maze_training.environment import make_demo_maze,step_environment
        maze=make_demo_maze(config["environment"])
        distances=compute_goal_distances(maze)
        for actions in ([1]*4,[2,0]*16):
            position=maze.start;raw=0.;shaped=0.
            for i,action in enumerate(actions):
                result=step_environment(maze,position,action,config["environment"])
                before=-config["ppo"]["potential_scale"]*distances[position]
                after=0. if result.done or i+1==len(actions) else -config["ppo"]["potential_scale"]*distances[result.position]
                raw+=result.reward;shaped+=result.reward+after-before;position=result.position
            self.assertAlmostEqual(shaped-raw,config["ppo"]["potential_scale"]*distances[maze.start])

    def test_sft_correction_sampling_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            run_stage("pretrain","configs/smoke.json",root)
            run_stage("sft","configs/smoke.json",root/"full",initialize_from=root/"pretrain.pt")
            run_stage("sft","configs/smoke.json",root/"resumed",steps=2,initialize_from=root/"pretrain.pt")
            run_stage("sft","configs/smoke.json",root/"resumed",resume=True)
            _,_,a=load_model(root/"full/sft.pt",torch.device("cpu"))
            _,_,b=load_model(root/"resumed/sft.pt",torch.device("cpu"))
            for key,value in a["model"].items():self.assertTrue(torch.equal(value,b["model"][key]),key)
            torch.testing.assert_close(a["sampling_weights"],b["sampling_weights"])

    def test_cycle_is_reported_without_escaping_by_heuristic(self):
        config=read_config("configs/smoke.json")
        grid=np.full((3,3),FLOOR);grid[0,2]=GOAL
        maze=Maze(grid,(2,0),(0,2),0);encoder=MazeEncoder(5)
        class CycleModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1))
            def forward(self,tokens):
                logits=torch.zeros((len(tokens),encoder.vocabulary_size))
                column=(tokens[:,2+encoder.position_count]-encoder.position_offset)%encoder.max_size
                actions=torch.where(column>0,3,1)
                logits[torch.arange(len(tokens)),actions+encoder.action_offset]=10.
                return logits
        with patch("maze_training.evaluation.build_maps",return_value=[maze]):
            report=evaluate_navigation(config,{"cycle":(CycleModel(),encoder)})["policies"]["cycle"]
        self.assertEqual(report["loop_rate"],1.)
        self.assertEqual(report["timeout_rate"],1.)
        self.assertEqual(report["mean_collisions"],0.)

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
