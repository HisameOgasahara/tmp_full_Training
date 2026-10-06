"""Verify deterministic costs and a real shortest-path/cost-path distinction."""

import unittest
import numpy as np
from maze_training.environment import FLOOR, WALL, MUD, GOAL, Maze, step_environment, generate_maze, find_bfs_path, find_min_cost_path, get_path_actions, calculate_path_cost, make_demo_maze
from maze_training.runtime import read_config


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.rules = read_config("configs/smoke.json")["environment"]

    def test_movement_costs_and_terminal_rewards(self):
        grid = np.array([[FLOOR, WALL, GOAL], [FLOOR, MUD, FLOOR], [FLOOR, FLOOR, FLOOR]])
        maze = Maze(grid, (0,0), (0,2), 0)
        collision = step_environment(maze, (0,0), 1, self.rules)
        self.assertEqual(collision.position, (0,0))
        self.assertTrue(collision.collision)
        self.assertEqual(collision.cost, 3)
        self.assertAlmostEqual(collision.reward, -0.3)
        mud = step_environment(maze, (1,0), 1, self.rules)
        self.assertEqual(mud.position, (1,1))
        self.assertEqual(mud.cost, 4)
        self.assertAlmostEqual(mud.reward, -0.4)
        self.assertFalse(mud.done)
        goal = step_environment(maze, (1,2), 0, self.rules)
        self.assertTrue(goal.success)
        self.assertAlmostEqual(goal.reward, 4.9)
        terminal = step_environment(maze, maze.goal, 2, self.rules)
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.cost, 0)

    def test_demo_shortcut_and_detour_have_known_costs(self):
        maze = make_demo_maze(self.rules)
        shortest = find_bfs_path(maze)
        cheapest = find_min_cost_path(maze, self.rules)
        self.assertEqual(get_path_actions(shortest)[0], 1)
        self.assertEqual(get_path_actions(cheapest)[0], 2)
        self.assertEqual(len(shortest)-1, 4)
        self.assertEqual(len(cheapest)-1, 8)
        self.assertEqual(calculate_path_cost(maze, shortest, self.rules), 13)
        self.assertEqual(calculate_path_cost(maze, cheapest, self.rules), 8)

    def test_generated_maps_require_cost_length_tradeoff(self):
        for seed in range(30):
            maze = generate_maze(seed, self.rules)
            shortest = find_bfs_path(maze)
            cheapest = find_min_cost_path(maze, self.rules)
            self.assertGreater(len(cheapest), len(shortest))
            self.assertLessEqual(len(cheapest)-1, self.rules["max_steps"])
            self.assertGreaterEqual(calculate_path_cost(maze, shortest, self.rules)-calculate_path_cost(maze, cheapest, self.rules), self.rules["minimum_cost_gap"])
            for path in (shortest, cheapest):
                position = maze.start
                for action in get_path_actions(path):
                    outcome = step_environment(maze, position, action, self.rules)
                    self.assertFalse(outcome.collision)
                    position = outcome.position
                self.assertEqual(position, maze.goal)
            repeated = generate_maze(seed, self.rules)
            np.testing.assert_array_equal(maze.grid, repeated.grid)


if __name__ == "__main__":
    unittest.main()
