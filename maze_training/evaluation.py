"""Evaluate held-out dynamics and paired closed-loop navigation episodes."""

import numpy as np
import torch
from .data import build_maps, sample_world_batch
from .environment import HOLE, GOAL, WALL, step_environment, transition_distribution, find_bfs_path, get_path_actions
from .runtime import mixed_precision


@torch.no_grad()
def predict_action_probabilities(model, encoder, mazes, positions, temperature=1.0):
    device = next(model.parameters()).device
    tokens = torch.tensor([encoder.encode_query(maze, state) for maze, state in zip(mazes, positions)], device=device)
    with mixed_precision(device):
        logits = model(tokens)[:, encoder.action_slice].float()
    return torch.softmax(logits / temperature, dim=-1)


@torch.no_grad()
def evaluate_world(model, encoder, config, split="validation"):
    maps = build_maps(config, split)
    rng = np.random.default_rng(config["seed"] + 70)
    inputs, targets, exact = sample_world_batch(maps, encoder, config["environment"], config["evaluation"]["world_samples"], rng)
    device = next(model.parameters()).device
    predictions = []
    batch_size = config["evaluation"]["batch_size"]
    for start in range(0, len(inputs), batch_size):
        tokens = torch.as_tensor(inputs[start:start + batch_size], device=device)
        with mixed_precision(device):
            logits = model(tokens)[:, encoder.position_slice].float()
        predictions.append(logits.softmax(-1).cpu().numpy())
    predicted = np.concatenate(predictions)
    safe_predicted = np.clip(predicted, 1e-10, 1)
    kl = np.sum(exact * (np.log(np.clip(exact, 1e-10, 1)) - np.log(safe_predicted)), axis=-1)
    return {
        "split": split, "samples": len(inputs),
        "sampled_next_position_accuracy": float(np.mean(predicted.argmax(-1) == targets)),
        "total_variation_distance": float(np.abs(predicted - exact).sum(-1).mean() / 2),
        "kl_exact_to_model": float(kl.mean()),
        "exact_support_probability": float((predicted * (exact > 0)).sum(-1).mean()),
    }


def compute_optimal_policy(maze, rules, slippery=True):
    """Finite-horizon dynamic programming for the very same episode reward."""
    horizon = rules["max_steps"]
    policies = np.zeros((horizon + 1, maze.size, maze.size), dtype=np.int64)
    values = np.zeros((maze.size, maze.size), dtype=np.float64)
    states = [tuple(map(int, p)) for p in np.argwhere((maze.grid != WALL) & (maze.grid != HOLE) & (maze.grid != GOAL))]
    transitions = {}
    for position in states:
        for action in range(4):
            items = []
            for target, probability in transition_distribution(maze, position, action, rules, slippery).items():
                success, hole = target == maze.goal, maze.grid[target] == HOLE
                reward = rules["step_reward"] + (rules["collision_reward"] if target == position else 0)
                reward += rules["goal_reward"] if success else (rules["hole_reward"] if hole else 0)
                items.append((target, probability, reward, success or hole))
            transitions[position, action] = items
    for remaining in range(1, horizon + 1):
        updated = np.zeros_like(values)
        for position in states:
            action_values = [sum(probability * (reward + (0 if done else values[target])) for target, probability, reward, done in transitions[position, action]) for action in range(4)]
            policies[remaining][position] = int(np.argmax(action_values))
            updated[position] = max(action_values)
        values = updated
    return policies, float(values[maze.start])


def summarize_episodes(records):
    successes = [record for record in records if record["success"]]
    return {
        "episodes": len(records),
        "success_rate": float(np.mean([r["success"] for r in records])),
        "hole_rate": float(np.mean([r["hole"] for r in records])),
        "timeout_rate": float(np.mean([r["timeout"] for r in records])),
        "mean_return": float(np.mean([r["return"] for r in records])),
        "mean_steps_all": float(np.mean([r["steps"] for r in records])),
        "mean_steps_success": float(np.mean([r["steps"] for r in successes])) if successes else None,
        "mean_collisions": float(np.mean([r["collisions"] for r in records])),
    }


@torch.no_grad()
def evaluate_navigation(config, policies, split="validation", slippery=True):
    """policies: name -> (model, encoder), or 'bfs' / 'optimal'."""
    maps = build_maps(config, split)
    rules = config["environment"]
    trials = config["evaluation"]["trials_per_map"]
    episodes = [maze for maze in maps for _ in range(trials)]
    # Seed by map and trial, not model name or how many random draws a policy made.
    noise = [np.random.default_rng(maze.seed * 100 + trial + 123).random(rules["max_steps"]) for maze in maps for trial in range(trials)]
    optimal = {maze.seed: compute_optimal_policy(maze, rules, slippery)[0] for maze in maps} if "optimal" in policies.values() else {}
    bfs_cache = {}
    for maze in maps:
        for position in np.argwhere((maze.grid != WALL) & (maze.grid != HOLE) & (maze.grid != GOAL)):
            position = tuple(map(int, position))
            path = find_bfs_path(maze, position)
            bfs_cache[maze.seed, position] = get_path_actions(path)[0] if len(path) > 1 else 0
    results = {}
    for name, policy in policies.items():
        positions = [maze.start for maze in episodes]
        records = [{"success": False, "hole": False, "timeout": False, "return": 0.0, "steps": 0, "collisions": 0} for _ in episodes]
        active = list(range(len(episodes)))
        for step in range(rules["max_steps"]):
            if not active:
                break
            if isinstance(policy, str):
                actions = [int(optimal[episodes[i].seed][rules["max_steps"] - step][positions[i]]) if policy == "optimal" else bfs_cache.get((episodes[i].seed, positions[i]), 0) for i in active]
            else:
                model, encoder = policy
                model.eval()
                actions = []
                for start in range(0, len(active), config["evaluation"]["batch_size"]):
                    indices = active[start:start + config["evaluation"]["batch_size"]]
                    probabilities = predict_action_probabilities(model, encoder, [episodes[i] for i in indices], [positions[i] for i in indices])
                    actions.extend(probabilities.argmax(-1).cpu().tolist())
            remaining = []
            for index, action in zip(active, actions):
                outcome = step_environment(episodes[index], positions[index], action, rules, noise[index][step], slippery)
                positions[index] = outcome.position
                record = records[index]
                record["return"] += outcome.reward
                record["steps"] += 1
                record["collisions"] += int(outcome.collision)
                record["success"] = outcome.success
                record["hole"] = bool(episodes[index].grid[outcome.position] == HOLE)
                if not outcome.done:
                    remaining.append(index)
            active = remaining
        for index in active:
            records[index]["timeout"] = True
        results[name] = summarize_episodes(records)
    return {"split": split, "slippery": slippery, "maps": len(maps), "trials_per_map": trials, "policies": results}
