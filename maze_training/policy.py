"""Use the same local wall constraints for training, evaluation and execution."""

import numpy as np
import torch
from .environment import move_nominal


def build_action_masks(mazes, positions):
    return np.asarray([[not move_nominal(maze, position, action)[1] for action in range(4)] for maze, position in zip(mazes, positions)], dtype=bool)


def mask_action_logits(logits, masks):
    masks = torch.as_tensor(masks, device=logits.device, dtype=torch.bool)
    if not masks.any(dim=-1).all():
        raise ValueError("이 위치에는 이동 가능한 방향이 없습니다.")
    return logits.float().masked_fill(~masks, -1e9)
