"""Stage reports and interactive deterministic weighted-maze execution."""

import html
import json
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ipywidgets as widgets
from IPython.display import display, clear_output
from .environment import ACTION_NAMES, FLOOR, WALL, MUD, GOAL, generate_maze, make_demo_maze, step_environment
from .evaluation import evaluate_world, evaluate_navigation, predict_action_probabilities
from .runtime import load_model, read_config, write_json, mixed_precision

TILE_COLORS = ("#f1f5f9", "#334155", "#d6a15e", "#86efac")


def draw_maze(axis, maze, position, path=None, title="", probabilities=None, observed_position=None):
    axis.imshow(maze.grid, cmap=ListedColormap(TILE_COLORS), vmin=FLOOR, vmax=GOAL)
    for row, column in np.argwhere(maze.grid == MUD):
        axis.text(column, row - 0.2, "M", ha="center", va="center", fontsize=9)
    if probabilities:
        for (row, column), probability in probabilities.items():
            if row < maze.size and column < maze.size and probability >= 0.01:
                axis.text(column, row + 0.25, f"{probability:.0%}", ha="center", va="center", fontsize=9)
    if path and len(path) > 1:
        axis.plot([p[1] for p in path], [p[0] for p in path], color="#7c3aed", alpha=0.7, linewidth=2)
    axis.scatter([position[1]], [position[0]], s=100, color="#1d4ed8", edgecolors="white", zorder=5)
    if observed_position is not None:
        axis.scatter([observed_position[1]], [observed_position[0]], s=180, facecolors="none", edgecolors="#ea580c", linewidths=2, zorder=6)
    axis.text(maze.goal[1], maze.goal[0] - 0.18, "G", ha="center", va="center", weight="bold")
    axis.set_xticks(np.arange(maze.size))
    axis.set_yticks(np.arange(maze.size))
    axis.set_xticks(np.arange(-0.5, maze.size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, maze.size, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1)
    axis.tick_params(which="minor", length=0)
    axis.set_title(title)


def display_legend(rules):
    display(widgets.HTML(f"<p>파란 점: 현재 위치 · 흰색: 일반 바닥(비용 {rules['floor_cost']:g}) · 진회색: 벽 · 갈색 M: 늪(비용 {rules['mud_cost']:g}) · 초록색 G: 목표<br>도착 칸의 비용을 지불하고, 벽 충돌에는 추가 비용 {rules['collision_cost']:g}이 붙습니다.</p>"))


def report_world(checkpoint, split="validation"):
    model, encoder, payload = load_model(checkpoint)
    metrics = evaluate_world(model, encoder, payload["config"], split)
    write_json(Path(checkpoint).parent / f"{payload['stage']}_world_{split}.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("next_position_accuracy는 높을수록, cost_mae는 낮을수록 좋습니다.")
    return metrics


def report_navigation(checkpoints, config_path="configs/t4.json", split="validation"):
    config = read_config(config_path)
    policies = {"BFS": "bfs", "Dijkstra": "dijkstra"}
    for name, path in checkpoints.items():
        model, encoder, payload = load_model(path)
        if payload["config"] != config:
            raise ValueError("평가 설정과 체크포인트 설정이 다릅니다.")
        policies[name] = (model, encoder)
    metrics = evaluate_navigation(config, policies, split)
    if checkpoints:
        parent = Path(next(iter(checkpoints.values()))).parent
        write_json(parent / f"navigation_{split}.json", metrics)
    headings = (("success_rate", "도착률"), ("timeout_rate", "시간초과율"), ("mean_return", "평균 보상"), ("mean_cost_success", "성공 시 비용"), ("mean_excess_cost_success", "최소 비용 대비 초과"), ("mean_steps_success", "성공 시 이동 수"), ("mean_collisions", "충돌 수"))
    rows = ""
    for name, values in metrics["policies"].items():
        entries = ["—" if values[key] is None else f"{values[key]:.3f}" for key, _ in headings]
        rows += "<tr><th>" + html.escape(name) + "</th>" + "".join(f"<td>{entry}</td>" for entry in entries) + "</tr>"
    display(widgets.HTML("<table><tr><th>정책</th>" + "".join(f"<th>{name}</th>" for _, name in headings) + "</tr>" + rows + "</table>"))
    print(f"{split}: 같은 미로 {metrics['maps']}개, 모델 행동은 argmax입니다.")
    print("도착률·평균 보상을 먼저 비교하고 성공 시 비용과 초과 비용을 확인하세요.")
    return metrics


def show_training_curve(output_dir, stage):
    history = json.loads((Path(output_dir) / f"{stage}_history.json").read_text(encoding="utf-8"))
    keys = [key for key in ("loss", "accuracy", "mean_return", "sampled_success_rate", "mean_cost", "value_loss") if key in history[0]]
    figure, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3), squeeze=False)
    for axis, key in zip(axes[0], keys):
        axis.plot([row["step"] for row in history], [row[key] for row in history])
        axis.set_title(key)
        axis.set_xlabel("update")
    plt.tight_layout()
    plt.show()
    plt.close(figure)


def create_map_controls():
    choice = widgets.Dropdown(options=[("처음 보는 미로", "random"), ("늪 지름길과 일반 우회로", "demo")], description="지도")
    seed = widgets.IntText(value=3_000_042, description="지도 seed")
    return choice, seed


def launch_world(checkpoint):
    model, encoder, payload = load_model(checkpoint)
    rules, output = payload["config"]["environment"], widgets.Output()
    choice, seed = create_map_controls()
    action = widgets.Dropdown(options=[(name, i) for i, name in enumerate(ACTION_NAMES)], value=1, description="행동")
    step_button = widgets.Button(description="실제 한 걸음 실행")
    reset_button = widgets.Button(description="지도 초기화")
    state = {}

    @torch.no_grad()
    def render():
        maze, position = state["maze"], state["prediction_position"]
        chosen = state["prediction_action"]
        tokens = torch.tensor([encoder.encode_query(maze, position, chosen)], device=next(model.parameters()).device)
        with mixed_precision(tokens.device):
            logits, cost, _ = model(tokens, with_auxiliary=True)
        probabilities = logits[:, encoder.position_slice].float().softmax(-1)[0].cpu().numpy()
        predicted = {encoder.decode_position(i):float(p) for i,p in enumerate(probabilities)}
        exact = step_environment(maze, position, chosen, rules)
        with output:
            clear_output(wait=True)
            figure, axes = plt.subplots(1, 2, figsize=(9, 4))
            draw_maze(axes[0], maze, position, state["path"], "Model next position", predicted, state["observed_position"])
            draw_maze(axes[1], maze, position, state["path"], "Exact next position", {exact.position:1.0}, state["observed_position"])
            plt.tight_layout()
            plt.show()
            plt.close(figure)
            print(f"위치 {position} / {ACTION_NAMES[chosen]}: 예측 비용 {float(cost[0])*rules['mud_cost']:.2f}, 실제 비용 {exact.cost:.2f}")
            print(state["message"])

    def reset(_=None):
        maze = make_demo_maze(rules) if choice.value == "demo" else generate_maze(seed.value, rules)
        state.update(maze=maze, position=maze.start, prediction_position=maze.start, prediction_action=action.value, observed_position=None, path=[maze.start], done=False, steps=0, message="행동을 바꾸면 다음 위치와 비용 예측을 갱신합니다.")
        step_button.disabled = False
        render()

    def predict(_=None):
        state.update(prediction_position=state["position"], prediction_action=action.value, observed_position=None)
        render()

    def step(_=None):
        if state["done"]:
            return
        state.update(prediction_position=state["position"], prediction_action=action.value)
        outcome = step_environment(state["maze"], state["position"], action.value, rules)
        state.update(position=outcome.position, observed_position=outcome.position, steps=state["steps"]+1)
        state["path"].append(outcome.position)
        state["done"] = outcome.done or state["steps"] >= rules["max_steps"]
        state["message"] = f"실제 위치 {outcome.position}, 비용 {outcome.cost:g}, 충돌 {outcome.collision}, 종료 {state['done']}"
        step_button.disabled = state["done"]
        render()

    step_button.on_click(step)
    reset_button.on_click(reset)
    action.observe(predict, names="value")
    display_legend(rules)
    display(widgets.HTML("<p>파란 점: 행동 전 위치 · 주황색 테두리: 실제 이동 후 위치</p>"))
    display(widgets.VBox([widgets.HBox([choice, seed, action]), widgets.HBox([step_button, reset_button]), output]))
    reset()
    return state


def launch_navigation(checkpoints, config_path="configs/t4.json"):
    config, output = read_config(config_path), widgets.Output()
    rules = config["environment"]
    models, titles = {}, {}
    for name, path in checkpoints.items():
        model, encoder, payload = load_model(path)
        if payload["config"] != config or payload["stage"] not in ("sft", "ppo"):
            raise ValueError("동일 설정의 SFT 또는 PPO 체크포인트를 사용하세요.")
        models[name] = (model, encoder)
        titles[name] = payload["stage"].upper()
    if not models:
        raise ValueError("실행할 모델이 필요합니다.")
    choice, seed = create_map_controls()
    step_button = widgets.Button(description="한 걸음 실행")
    auto_button = widgets.Button(description="자동 실행")
    reset_button = widgets.Button(description="지도 초기화")
    state = {}

    def render():
        with output:
            clear_output(wait=True)
            figure, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4), squeeze=False)
            for axis, (name, episode) in zip(axes[0], state["episodes"].items()):
                draw_maze(axis, state["maze"], episode["position"], episode["path"], f"{titles[name]}: {episode['steps']} steps / cost {episode['cost']:g}")
            plt.tight_layout()
            plt.show()
            plt.close(figure)
            for name, episode in state["episodes"].items():
                print(f"{name}: {episode['status']} | 총비용 {episode['cost']:g} | 충돌 {episode['collisions']}회")

    def reset(_=None):
        maze = make_demo_maze(rules) if choice.value == "demo" else generate_maze(seed.value, rules)
        state["maze"] = maze
        state["episodes"] = {name:{"position":maze.start,"path":[maze.start],"steps":0,"cost":0.0,"collisions":0,"done":False,"status":"진행 중"} for name in models}
        render()

    def advance(_=None):
        advanced = False
        for name, (model, encoder) in models.items():
            episode = state["episodes"][name]
            if episode["done"]:
                continue
            action = int(predict_action_probabilities(model, encoder, [state["maze"]], [episode["position"]]).argmax(-1)[0])
            outcome = step_environment(state["maze"], episode["position"], action, rules)
            episode["position"] = outcome.position
            episode["path"].append(outcome.position)
            episode["steps"] += 1
            episode["cost"] += outcome.cost
            episode["collisions"] += int(outcome.collision)
            episode["done"] = outcome.done or episode["steps"] >= rules["max_steps"]
            episode["status"] = "도착" if outcome.success else ("시간초과" if episode["done"] else "진행 중")
            advanced = True
        render()
        return advanced

    def autoplay(_):
        auto_button.disabled = step_button.disabled = reset_button.disabled = True
        try:
            while advance():
                time.sleep(config["display"]["animation_delay"])
        finally:
            auto_button.disabled = step_button.disabled = reset_button.disabled = False

    step_button.on_click(advance)
    auto_button.on_click(autoplay)
    reset_button.on_click(reset)
    display_legend(rules)
    display(widgets.VBox([widgets.HBox([choice,seed]),widgets.HBox([step_button,auto_button,reset_button]),output]))
    reset()
    return state
