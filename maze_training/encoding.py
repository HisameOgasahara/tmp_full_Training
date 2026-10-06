"""Encode grids and queries with a small explicit vocabulary, without a tokenizer."""

import numpy as np
from .environment import WALL, ACTION_DELTAS


class MazeEncoder:
    def __init__(self, max_size):
        self.max_size = max_size
        self.position_offset = 4
        self.position_count = max_size ** 2
        self.action_offset = self.position_offset + self.position_count
        self.size_offset = self.action_offset + 4
        self.world_token = self.size_offset + max_size + 1
        self.policy_token = self.world_token + 1
        self.query_token = self.world_token + 2
        self.vocabulary_size = self.query_token + 1
        self.context_length = self.position_count + 5

    def position_index(self, position):
        return position[0] * self.max_size + position[1]

    def decode_position(self, index):
        return divmod(int(index), self.max_size)

    def encode_grid(self, maze):
        grid = np.full((self.max_size, self.max_size), WALL, dtype=np.int64)
        grid[:maze.size, :maze.size] = maze.grid
        return grid.flatten().tolist()

    def encode_query(self, maze, position, action=None):
        task = self.policy_token if action is None else self.world_token
        tokens = [task, self.size_offset + maze.size, *self.encode_grid(maze), self.position_offset + self.position_index(position)]
        if action is not None:
            tokens.append(self.action_offset + action)
        return tokens + [self.query_token]

    @property
    def action_slice(self):
        return slice(self.action_offset, self.action_offset + 4)

    @property
    def position_slice(self):
        return slice(self.position_offset, self.position_offset + self.position_count)

    def decode_action_masks(self, queries):
        """Read local neighbors from encoded maps; no route information is used."""
        queries = np.asarray(queries)
        grids = queries[:, 2:2+self.position_count].reshape(-1,self.max_size,self.max_size)
        positions = queries[:, 2+self.position_count] - self.position_offset
        masks = np.zeros((len(queries),4), dtype=bool)
        sizes = queries[:,1]-self.size_offset
        for index, (grid, position, size) in enumerate(zip(grids, positions, sizes)):
            row,column = divmod(int(position),self.max_size)
            for action,(dr,dc) in enumerate(ACTION_DELTAS):
                r,c = row+dr,column+dc
                masks[index,action] = 0<=r<size and 0<=c<size and grid[r,c]!=WALL
        return masks
