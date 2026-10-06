"""Deterministic weighted mazes, shortest paths, and exact movement costs."""

from collections import deque
from dataclasses import dataclass
import heapq
import numpy as np

FLOOR, WALL, MUD, GOAL = range(4)
ACTION_NAMES = ("위", "오른쪽", "아래", "왼쪽")
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass
class Maze:
    grid: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    seed: int

    @property
    def size(self):
        return self.grid.shape[0]


@dataclass
class Outcome:
    position: tuple[int, int]
    reward: float
    cost: float
    done: bool
    success: bool
    collision: bool


def move_nominal(maze, position, action):
    dr, dc = ACTION_DELTAS[action]
    target = (position[0] + dr, position[1] + dc)
    r, c = target
    collision = not (0 <= r < maze.size and 0 <= c < maze.size)
    if not collision:
        collision = maze.grid[r, c] == WALL
    return (position if collision else target), bool(collision)


def step_environment(maze, position, action, rules):
    if position == maze.goal:
        return Outcome(position, 0.0, 0.0, True, True, False)
    target, collision = move_nominal(maze, position, action)
    cost = rules["mud_cost"] if maze.grid[target] == MUD else rules["floor_cost"]
    cost += rules["collision_cost"] if collision else 0.0
    success = target == maze.goal
    reward = -rules["cost_scale"] * cost + (rules["goal_reward"] if success else 0.0)
    return Outcome(target, float(reward), float(cost), success, success, collision)


def reconstruct_path(parent, target):
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    return path[::-1]


def compute_goal_distances(maze):
    """Exact unweighted distances for BFS supervision and potential shaping."""
    distances = {maze.goal: 0}
    queue = deque([maze.goal])
    while queue:
        position = queue.popleft()
        for action in range(4):
            target, collision = move_nominal(maze, position, action)
            if not collision and target not in distances:
                distances[target] = distances[position] + 1
                queue.append(target)
    return distances


def find_bfs_path(maze, start=None):
    start = maze.start if start is None else start
    queue, parent = deque([start]), {start: None}
    while queue:
        position = queue.popleft()
        if position == maze.goal:
            return reconstruct_path(parent, position)
        for action in range(4):
            target, collision = move_nominal(maze, position, action)
            if not collision and target not in parent:
                parent[target] = position
                queue.append(target)
    return []


def find_min_cost_path(maze, rules, start=None):
    start = maze.start if start is None else start
    frontier, distances, parent = [(0.0, start)], {start: 0.0}, {start: None}
    while frontier:
        cost, position = heapq.heappop(frontier)
        if cost != distances[position]:
            continue
        if position == maze.goal:
            return reconstruct_path(parent, position)
        for action in range(4):
            outcome = step_environment(maze, position, action, rules)
            candidate = cost + outcome.cost
            if not outcome.collision and candidate < distances.get(outcome.position, float("inf")):
                distances[outcome.position] = candidate
                parent[outcome.position] = position
                heapq.heappush(frontier, (candidate, outcome.position))
    return []


def get_path_actions(path):
    return [ACTION_DELTAS.index((b[0] - a[0], b[1] - a[1])) for a, b in zip(path, path[1:])]


def calculate_path_cost(maze, path, rules):
    return sum(step_environment(maze, position, action, rules).cost for position, action in zip(path, get_path_actions(path)))


def generate_maze(seed, rules):
    rng = np.random.default_rng(seed)
    size = int(rng.choice(rules["sizes"]))
    if size > rules["max_size"] or size < 5 or not 0 <= rules["wall_probability"] < 1:
        raise ValueError("지도 크기와 벽 확률을 확인하세요.")
    if not 0 <= rules["mud_probability"] <= 1 or not 0 < rules["floor_cost"] < rules["mud_cost"]:
        raise ValueError("늪 확률과 이동 비용을 확인하세요.")
    if rules["cost_scale"] <= 0 or rules["collision_cost"] < 0 or rules["minimum_cost_gap"] <= 0:
        raise ValueError("비용 배율, 충돌 비용, 경로 비용 차이를 확인하세요.")
    for _ in range(rules["generation_attempts"]):
        grid = np.full((size, size), FLOOR, dtype=np.int64)
        grid[rng.random((size, size)) < rules["mud_probability"]] = MUD
        grid[rng.random((size, size)) < rules["wall_probability"]] = WALL
        start, goal = (size - 1, 0), (0, size - 1)
        grid[start], grid[goal] = FLOOR, GOAL
        maze = Maze(grid, start, goal, int(seed))
        if np.count_nonzero(grid == WALL) < size:
            continue
        shortest = find_bfs_path(maze)
        if not shortest:
            continue
        cheapest = find_min_cost_path(maze, rules)
        if len(cheapest) <= len(shortest) or len(cheapest) - 1 > rules["max_steps"]:
            continue
        gap = calculate_path_cost(maze, shortest, rules) - calculate_path_cost(maze, cheapest, rules)
        if gap >= rules["minimum_cost_gap"]:
            return maze
    raise RuntimeError("비용이 다른 두 경로를 만들지 못했습니다. 지도 생성 설정을 확인하세요.")


def make_demo_maze(rules):
    size = rules["max_size"]
    grid = np.full((size, size), WALL, dtype=np.int64)
    grid[0, :] = MUD
    grid[2, :] = FLOOR
    grid[:3, 0] = FLOOR
    grid[:3, -1] = FLOOR
    grid[0, -1] = GOAL
    return Maze(grid, (0, 0), (0, size - 1), -1)
