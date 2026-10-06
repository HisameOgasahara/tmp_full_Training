"""Generate maps, compute exact transitions, and execute navigation episodes."""

from collections import deque
from dataclasses import dataclass
import numpy as np

FLOOR, WALL, ICE, HOLE, GOAL = range(5)
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
    done: bool
    success: bool
    collision: bool


def move_nominal(maze, position, action):
    dr, dc = ACTION_DELTAS[action]
    candidate = (position[0] + dr, position[1] + dc)
    r, c = candidate
    collision = not (0 <= r < maze.size and 0 <= c < maze.size)
    if not collision:
        collision = maze.grid[r, c] == WALL
    return (position if collision else candidate), bool(collision)


def transition_distribution(maze, position, action, rules, slippery=True):
    """Aggregate duplicate outcomes (e.g. two directions hit a wall)."""
    if maze.grid[position] in (HOLE, GOAL):
        return {position: 1.0}
    slip = rules["slip_probability"] if slippery and maze.grid[position] == ICE else 0.0
    directions = ((action, 1.0 - slip), ((action - 1) % 4, slip / 2), ((action + 1) % 4, slip / 2))
    distribution = {}
    for direction, probability in directions:
        if probability:
            target, _ = move_nominal(maze, position, direction)
            distribution[target] = distribution.get(target, 0.0) + probability
    return distribution


def step_environment(maze, position, action, rules, random_number, slippery=True):
    """Use an explicit uniform draw so all policies share the same noise stream."""
    if maze.grid[position] in (HOLE, GOAL):
        return Outcome(position, 0.0, True, position == maze.goal, False)
    slip = rules["slip_probability"] if slippery and maze.grid[position] == ICE else 0.0
    if random_number < 1.0 - slip:
        direction = action
    elif random_number < 1.0 - slip / 2:
        direction = (action - 1) % 4
    else:
        direction = (action + 1) % 4
    target, collision = move_nominal(maze, position, direction)
    success = target == maze.goal
    hole = maze.grid[target] == HOLE
    reward = rules["step_reward"]
    if collision:
        reward += rules["collision_reward"]
    if success:
        reward += rules["goal_reward"]
    elif hole:
        reward += rules["hole_reward"]
    return Outcome(target, float(reward), bool(success or hole), success, collision)


def find_bfs_path(maze, start=None):
    """Shortest four-neighbor path ignoring stochastic slipping; holes are blocked."""
    start = maze.start if start is None else start
    queue = deque([start])
    parent = {start: None}
    while queue:
        position = queue.popleft()
        if position == maze.goal:
            path = []
            while position is not None:
                path.append(position)
                position = parent[position]
            return path[::-1]
        for action in range(len(ACTION_DELTAS)):
            target, collision = move_nominal(maze, position, action)
            if not collision and target not in parent and maze.grid[target] != HOLE:
                parent[target] = position
                queue.append(target)
    return []


def get_path_actions(path):
    return [ACTION_DELTAS.index((b[0] - a[0], b[1] - a[1])) for a, b in zip(path, path[1:])]


def generate_maze(seed, rules):
    rng = np.random.default_rng(seed)
    size = int(rng.choice(rules["sizes"]))
    probabilities = [rules["wall_probability"], rules["hole_probability"], rules["ice_probability"]]
    if size > rules["max_size"] or sum(probabilities) >= 1:
        raise ValueError("지도 크기 또는 타일 확률 설정을 확인하세요.")
    for _ in range(1000):
        sample = rng.random((size, size))
        grid = np.full((size, size), FLOOR, dtype=np.int64)
        lower = 0.0
        for tile, probability in zip((WALL, HOLE, ICE), probabilities):
            grid[(sample >= lower) & (sample < lower + probability)] = tile
            lower += probability
        start, goal = (size - 1, 0), (0, size - 1)
        grid[start], grid[goal] = FLOOR, GOAL
        maze = Maze(grid, start, goal, int(seed))
        if find_bfs_path(maze):
            return maze
    raise RuntimeError("도달 가능한 지도를 만들지 못했습니다. 타일 확률을 낮춰주세요.")


def make_demo_maze(rules):
    """A fixed illustration, separate from training and evaluation maps."""
    size = rules["max_size"]
    grid = np.full((size, size), FLOOR, dtype=np.int64)
    grid[1, 1:size - 1] = HOLE
    grid[0, 1:size - 1] = ICE
    grid[0, size - 1] = GOAL
    return Maze(grid, (0, 0), (0, size - 1), -1)
