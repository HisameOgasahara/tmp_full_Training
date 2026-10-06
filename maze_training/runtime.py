"""Configuration, reproducibility, mixed precision, and stage checkpoint storage."""

from contextlib import nullcontext
import json
from pathlib import Path
import random
import torch
import numpy as np
from .encoding import MazeEncoder
from .model import MazeTransformer


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mixed_precision(device):
    return torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def create_model(config, device):
    encoder = MazeEncoder(config["environment"]["max_size"])
    return MazeTransformer(encoder.vocabulary_size, encoder.context_length, config["model"]).to(device), encoder


def save_checkpoint(path, model, config, stage, step, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "config": config, "stage": stage, "step": step, **extra}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, device=None):
    device = choose_device() if device is None else device
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model, encoder = create_model(payload["config"], device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, encoder, payload


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
