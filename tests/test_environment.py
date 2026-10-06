"""Check transition truth and the risk/shortcut distinction used in evaluation."""

import unittest
import numpy as np
from maze_training.environment import FLOOR, WALL, ICE, HOLE, GOAL, Maze, transition_distribution, step_environment, generate_maze, find_bfs_path, get_path_actions, make_demo_maze
from maze_training.evaluation import compute_optimal_policy
from maze_training.runtime import read_config


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.rules = read_config("configs/smoke.json")["environment"]

    def test_transition_probabilities_match_environment_sampling(self):
        grid = np.full((3, 3), FLOOR)
        grid[1, 1] = ICE
        grid[0, 1] = WALL
        grid[2, 1] = HOLE
        grid[0, 2] = GOAL
        maze = Maze(grid, (1, 1), (0, 2), 0)
        exact = transition_distribution(maze, maze.start, 1, self.rules)
        self.assertAlmostEqual(sum(exact.values()), 1.0)
        self.assertAlmostEqual(exact[(1, 2)], 0.7)
        self.assertAlmostEqual(exact[(1, 1)], 0.15)
        self.assertAlmostEqual(exact[(2, 1)], 0.15)
        rng = np.random.default_rng(1)
        observations = [step_environment(maze, maze.start, 1, self.rules, value).position for value in rng.random(6000)]
        for target, probability in exact.items():
            frequency = observations.count(target) / len(observations)
            self.assertLess(abs(frequency - probability), 0.025)

    def test_collision_and_terminal_rewards(self):
        grid = np.array([[FLOOR, WALL, GOAL], [FLOOR, FLOOR, FLOOR], [FLOOR, HOLE, FLOOR]])
        maze = Maze(grid, (0, 0), (0, 2), 0)
        collision = step_environment(maze, (0, 0), 1, self.rules, 0.5)
        self.assertEqual(collision.position, (0, 0))
        self.assertTrue(collision.collision)
        self.assertAlmostEqual(collision.reward, -0.03)
        hole = step_environment(maze, (2, 0), 1, self.rules, 0.5)
        self.assertTrue(hole.done)
        self.assertFalse(hole.success)
        self.assertAlmostEqual(hole.reward, -1.01)
        goal = step_environment(maze, (1, 2), 0, self.rules, 0.5)
        self.assertTrue(goal.success)
        self.assertAlmostEqual(goal.reward, 0.99)

    def test_generated_maps_have_valid_nominal_solutions(self):
        for seed in range(30):
            maze = generate_maze(seed, self.rules)
            path = find_bfs_path(maze)
            self.assertTrue(path)
            position = maze.start
            for action in get_path_actions(path):
                outcome = step_environment(maze, position, action, self.rules, 0.5, slippery=False)
                self.assertFalse(outcome.collision)
                self.assertNotEqual(maze.grid[outcome.position], HOLE)
                position = outcome.position
            self.assertEqual(position, maze.goal)

    def test_demo_has_shortcut_and_optimal_safe_detour(self):
        maze = make_demo_maze(self.rules)
        shortest = get_path_actions(find_bfs_path(maze))
        self.assertEqual(shortest[0], 1)
        policy, value = compute_optimal_policy(maze, self.rules)
        self.assertEqual(policy[self.rules["max_steps"]][maze.start], 2)
        self.assertGreater(value, 0.8)


if __name__ == "__main__":
    unittest.main()
