"""Disjoint seeded maps and exact transition/cost/BFS labels."""

import numpy as np
from .environment import WALL, GOAL, generate_maze, find_bfs_path, get_path_actions, step_environment, compute_goal_distances, move_nominal

SPLIT_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}


def build_maps(config, split):
    count = config["data"][f"{split}_maps"]
    if not 0 < count < 1_000_000:
        raise ValueError("지도 수는 1부터 999999까지입니다.")
    return [generate_maze(SPLIT_OFFSETS[split] + config["seed"] + i, config["environment"]) for i in range(count)]


def sample_world_batch(maps, encoder, rules, count, rng):
    queries, positions, costs = [], [], []
    for _ in range(count):
        maze = maps[int(rng.integers(len(maps)))]
        states = np.argwhere((maze.grid != WALL) & (maze.grid != GOAL))
        position = tuple(map(int, states[int(rng.integers(len(states)))]))
        action = int(rng.integers(4))
        outcome = step_environment(maze, position, action, rules)
        queries.append(encoder.encode_query(maze, position, action))
        positions.append(encoder.position_index(outcome.position))
        costs.append(outcome.cost / rules["mud_cost"])
    return np.asarray(queries), np.asarray(positions), np.asarray(costs, dtype=np.float32)


def build_expert_examples(maps, encoder):
    queries, targets = [], []
    for maze in maps:
        distances = compute_goal_distances(maze)
        for position, distance in distances.items():
            if distance == 0:
                continue
            optimal = []
            for action in range(4):
                neighbor, collision = move_nominal(maze, position, action)
                optimal.append(not collision and distances.get(neighbor) == distance - 1)
            queries.append(encoder.encode_query(maze, position))
            targets.append(optimal)
    return np.asarray(queries), np.asarray(targets, dtype=bool)
