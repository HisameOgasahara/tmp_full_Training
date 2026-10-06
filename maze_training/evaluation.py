"""Held-out deterministic dynamics and navigation cost evaluation."""

import numpy as np
import torch
from .data import build_maps, sample_world_batch
from .environment import WALL, GOAL, step_environment, find_bfs_path, find_min_cost_path, get_path_actions, calculate_path_cost
from .runtime import mixed_precision


@torch.no_grad()
def predict_action_probabilities(model, encoder, mazes, positions):
    device = next(model.parameters()).device
    tokens = torch.tensor([encoder.encode_query(maze, state) for maze, state in zip(mazes, positions)], device=device)
    with mixed_precision(device):
        logits = model(tokens)[:, encoder.action_slice].float()
    return logits.softmax(-1)


@torch.no_grad()
def evaluate_world(model, encoder, config, split="validation"):
    maps = build_maps(config, split)
    inputs, targets, costs = sample_world_batch(maps, encoder, config["environment"], config["evaluation"]["world_samples"], np.random.default_rng(config["seed"] + 70))
    device = next(model.parameters()).device
    correct, errors = [], []
    for start in range(0, len(inputs), config["evaluation"]["batch_size"]):
        stop = start + config["evaluation"]["batch_size"]
        with mixed_precision(device):
            logits, predicted_cost, _ = model(torch.as_tensor(inputs[start:stop], device=device), with_auxiliary=True)
        correct.extend((logits[:, encoder.position_slice].argmax(-1).cpu().numpy() == targets[start:stop]).tolist())
        errors.extend((np.abs(predicted_cost.float().cpu().numpy() - costs[start:stop]) * config["environment"]["mud_cost"]).tolist())
    return {"split": split, "samples": len(inputs), "next_position_accuracy": float(np.mean(correct)), "cost_mae": float(np.mean(errors))}


@torch.no_grad()
def evaluate_navigation(config, policies, split="validation"):
    maps, rules = build_maps(config, split), config["environment"]
    caches = {}
    for maze in maps:
        for position in np.argwhere((maze.grid != WALL) & (maze.grid != GOAL)):
            position = tuple(map(int, position))
            for kind, path in (("bfs", find_bfs_path(maze, position)), ("dijkstra", find_min_cost_path(maze, rules, position))):
                caches[maze.seed, position, kind] = get_path_actions(path)[0] if len(path) > 1 else 0
    minimum_costs = [calculate_path_cost(maze, find_min_cost_path(maze, rules), rules) for maze in maps]
    results = {}
    for name, policy in policies.items():
        positions = [maze.start for maze in maps]
        records = [{"success":False, "return":0.0, "cost":0.0, "steps":0, "collisions":0} for _ in maps]
        active = list(range(len(maps)))
        for _ in range(rules["max_steps"]):
            if not active:
                break
            if isinstance(policy, str):
                if policy not in ("bfs", "dijkstra"):
                    raise ValueError("평가 기준은 bfs 또는 dijkstra입니다.")
                actions = [caches[maps[i].seed, positions[i], policy] for i in active]
            else:
                model, encoder = policy
                model.eval()
                actions = []
                for start in range(0, len(active), config["evaluation"]["batch_size"]):
                    indices = active[start:start + config["evaluation"]["batch_size"]]
                    probs = predict_action_probabilities(model, encoder, [maps[i] for i in indices], [positions[i] for i in indices])
                    actions.extend(probs.argmax(-1).cpu().tolist())
            remaining = []
            for index, action in zip(active, actions):
                outcome = step_environment(maps[index], positions[index], action, rules)
                positions[index] = outcome.position
                record = records[index]
                record["return"] += outcome.reward
                record["cost"] += outcome.cost
                record["steps"] += 1
                record["collisions"] += int(outcome.collision)
                record["success"] = outcome.success
                if not outcome.done:
                    remaining.append(index)
            active = remaining
        successful = [r for r in records if r["success"]]
        excess = [r["cost"] - minimum_costs[i] for i, r in enumerate(records) if r["success"]]
        mean_success = lambda key: float(np.mean([r[key] for r in successful])) if successful else None
        results[name] = {
            "episodes":len(maps), "success_rate":len(successful)/len(maps), "timeout_rate":len(active)/len(maps),
            "mean_return":float(np.mean([r["return"] for r in records])),
            "mean_cost_all":float(np.mean([r["cost"] for r in records])),
            "mean_cost_success":mean_success("cost"), "mean_steps_success":mean_success("steps"),
            "mean_excess_cost_success":float(np.mean(excess)) if excess else None,
            "mean_collisions":float(np.mean([r["collisions"] for r in records])),
        }
    return {"split":split, "maps":len(maps), "policies":results}
