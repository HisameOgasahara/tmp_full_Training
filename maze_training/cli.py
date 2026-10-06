"""Command-line evaluation for the same checkpoints used in Colab."""

import argparse
import json
from pathlib import Path
from .runtime import read_config, load_model, seed_runtime, write_json
from .evaluation import evaluate_world, evaluate_navigation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("world", "navigation"))
    parser.add_argument("--config", default="configs/t4.json")
    parser.add_argument("--output-dir", default="runs/weighted_maze_v2/t4")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    arguments = parser.parse_args()
    config = read_config(arguments.config)
    seed_runtime(config["seed"])
    directory = Path(arguments.output_dir)
    if arguments.task == "world":
        model, encoder, payload = load_model(directory / "pretrain.pt")
        if payload["config"] != config:
            raise ValueError("설정 파일과 체크포인트가 다릅니다.")
        report = evaluate_world(model, encoder, config, arguments.split)
    else:
        policies = {"BFS": "bfs", "Dijkstra": "dijkstra"}
        for stage in ("sft", "ppo"):
            checkpoint = directory / f"{stage}.pt"
            if checkpoint.exists():
                model, encoder, payload = load_model(checkpoint)
                if payload["config"] != config:
                    raise ValueError("설정 파일과 체크포인트가 다릅니다.")
                policies[stage] = (model, encoder)
        report = evaluate_navigation(config, policies, arguments.split)
    write_json(directory / f"{arguments.task}_{arguments.split}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
