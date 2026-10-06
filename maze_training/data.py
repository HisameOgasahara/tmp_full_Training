"""Disjoint seeded map pools and automatically generated transition/expert labels."""

import numpy as np
from .environment import WALL, HOLE, GOAL, generate_maze, find_bfs_path, get_path_actions, transition_distribution

SPLIT_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}


def build_maps(config, split):
    count = config["data"][f"{split}_maps"]
    if not 0 < count < 1_000_000:
        raise ValueError("지도 수는 1부터 999999까지입니다.")
    offset = SPLIT_OFFSETS[split]
    return [generate_maze(offset + config["seed"] + i, config["environment"]) for i in range(count)]


def sample_world_batch(maps, encoder, rules, count, rng):
    queries, labels, distributions = [], [], []
    for _ in range(count):
        maze = maps[int(rng.integers(len(maps)))]
        positions = np.argwhere((maze.grid != WALL) & (maze.grid != HOLE) & (maze.grid != GOAL))
        position = tuple(int(v) for v in positions[int(rng.integers(len(positions)))])
        action = int(rng.integers(4))
        distribution = transition_distribution(maze, position, action, rules)
        targets = list(distribution)
        probabilities = list(distribution.values())
        target = targets[int(rng.choice(len(targets), p=probabilities))]
        dense = np.zeros(encoder.position_count, dtype=np.float32)
        for state, probability in distribution.items():
            dense[encoder.position_index(state)] = probability
        queries.append(encoder.encode_query(maze, position, action))
        labels.append(encoder.position_index(target))
        distributions.append(dense)
    return np.asarray(queries), np.asarray(labels), np.asarray(distributions)


def build_expert_examples(maps, encoder):
    queries, labels = [], []
    for maze in maps:
        # Include recoverable off-route positions, rather than only the start-to-goal path.
        for row, column in np.argwhere((maze.grid != WALL) & (maze.grid != HOLE) & (maze.grid != GOAL)):
            position = (int(row), int(column))
            path = find_bfs_path(maze, position)
            if len(path) > 1:
                queries.append(encoder.encode_query(maze, position))
                labels.append(get_path_actions(path)[0])
    return np.asarray(queries), np.asarray(labels)
