"""Notebook entry points: per-stage reports and interactive prediction/navigation."""

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
from .environment import ACTION_NAMES, FLOOR, WALL, ICE, HOLE, GOAL, generate_maze, make_demo_maze, step_environment, transition_distribution
from .evaluation import evaluate_world, evaluate_navigation, predict_action_probabilities
from .runtime import load_model, read_config, write_json, mixed_precision

TILE_COLORS = ("#f1f5f9", "#334155", "#7dd3fc", "#fb7185", "#86efac")


def draw_maze(axis, maze, position, path=None, title="", probabilities=None, observed_position=None):
    axis.imshow(maze.grid, cmap=ListedColormap(TILE_COLORS), vmin=FLOOR, vmax=GOAL)
    if probabilities is not None:
        for row in range(maze.size):
            for column in range(maze.size):
                probability = probabilities.get((row, column), 0.0)
                if probability >= 0.01:
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


def display_legend():
    display(widgets.HTML("<p>파란 점: 현재 위치 · 흰색: 바닥 · 진회색: 벽 · 하늘색: 미끄러운 바닥 · 분홍색: 구멍 · 초록색 G: 목표<br>행동은 항상 4방향이며, 미끄러짐은 <b>출발하는 칸</b>이 하늘색일 때 발생합니다.</p>"))


def report_world(checkpoint, split="validation"):
    model, encoder, payload = load_model(checkpoint)
    metrics = evaluate_world(model, encoder, payload["config"], split)
    write_json(Path(checkpoint).parent / f"{payload['stage']}_world_{split}.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("TV distance와 KL은 낮을수록, exact_support_probability는 높을수록 좋습니다.")
    return metrics


def report_navigation(checkpoints, config_path="configs/t4.json", split="validation", slippery=True):
    config = read_config(config_path)
    policies = {"BFS": "bfs", "optimal": "optimal"}
    for name, path in checkpoints.items():
        model, encoder, payload = load_model(path)
        if payload["config"] != config:
            raise ValueError("평가 설정과 체크포인트 설정이 다릅니다.")
        policies[name] = (model, encoder)
    metrics = evaluate_navigation(config, policies, split, slippery)
    if checkpoints:
        parent = Path(next(iter(checkpoints.values()))).parent
        write_json(parent / f"navigation_{split}_{'slippery' if slippery else 'dry'}.json", metrics)
    headings = (("success_rate", "도착률"), ("hole_rate", "추락률"), ("timeout_rate", "시간초과율"), ("mean_return", "평균 보상"), ("mean_steps_success", "성공 시 이동 수"), ("mean_collisions", "충돌 수"))
    rows = ""
    for name, values in metrics["policies"].items():
        entries = ["—" if values[key] is None else f"{values[key]:.3f}" for key, _ in headings]
        rows += "<tr><th>" + html.escape(name) + "</th>" + "".join(f"<td>{entry}</td>" for entry in entries) + "</tr>"
    display(widgets.HTML("<table><tr><th>정책</th>" + "".join(f"<th>{name}</th>" for _, name in headings) + "</tr>" + rows + "</table>"))
    print(f"{split}: {metrics['maps']}개 지도 × {metrics['trials_per_map']}회. 모든 정책에 같은 지도·환경 난수를 적용했습니다.")
    print("모델 행동은 argmax입니다. 도착률·추락률·평균 보상을 함께 비교하세요.")
    return metrics


def show_training_curve(output_dir, stage):
    history = json.loads((Path(output_dir) / f"{stage}_history.json").read_text(encoding="utf-8"))
    keys = [key for key in ("loss", "accuracy", "mean_return", "sampled_success_rate", "useful_group_fraction") if key in history[0]]
    figure, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3), squeeze=False)
    for axis, key in zip(axes[0], keys):
        axis.plot([row["step"] for row in history], [row[key] for row in history])
        axis.set_title(key)
        axis.set_xlabel("update")
    plt.tight_layout()
    plt.show()


def launch_world(checkpoint):
    model, encoder, payload = load_model(checkpoint)
    config, output = payload["config"], widgets.Output()
    rules = config["environment"]
    maze_choice = widgets.Dropdown(options=[("위험한 지름길 예시", "demo"), ("처음 보는 무작위 지도", "random")], description="지도")
    seed = widgets.IntText(value=3_000_042, description="지도 seed")
    action = widgets.Dropdown(options=[(name, i) for i, name in enumerate(ACTION_NAMES)], description="행동")
    predict_button = widgets.Button(description="이동 결과 예측")
    step_button = widgets.Button(description="실제 한 걸음 실행")
    reset_button = widgets.Button(description="지도 초기화")
    state = {}

    def reset(_=None):
        maze = make_demo_maze(rules) if maze_choice.value == "demo" else generate_maze(seed.value, rules)
        state.update(maze=maze, position=maze.start, path=[maze.start], done=False, rng=np.random.default_rng(seed.value + 1), message="행동을 고르고 예측 확률을 확인하세요.")
        state.update(prediction_position=maze.start, prediction_action=action.value, observed_position=None)
        predict_button.disabled = step_button.disabled = False
        render()

    @torch.no_grad()
    def render(_=None):
        maze, position = state["maze"], state["prediction_position"]
        predicted_action = state["prediction_action"]
        device = next(model.parameters()).device
        tokens = torch.tensor([encoder.encode_query(maze, position, predicted_action)], device=device)
        with mixed_precision(device):
            prediction = model(tokens)[:, encoder.position_slice].float().softmax(-1)[0].cpu().numpy()
        predicted = {encoder.decode_position(i): float(p) for i, p in enumerate(prediction)}
        exact = transition_distribution(maze, position, predicted_action, rules)
        with output:
            clear_output(wait=True)
            figure, axes = plt.subplots(1, 2, figsize=(9, 4))
            draw_maze(axes[0], maze, position, state["path"], "Model prediction", predicted, state["observed_position"])
            draw_maze(axes[1], maze, position, state["path"], "Exact environment distribution", exact, state["observed_position"])
            plt.tight_layout()
            plt.show()
            invalid_mass = sum(p for (r, c), p in predicted.items() if r >= maze.size or c >= maze.size or maze.grid[r, c] == WALL)
            print(state["message"])
            print(f"예측 대상: 위치 {position}에서 {ACTION_NAMES[predicted_action]} 행동")
            print(f"예측한 벽·지도 밖 확률: {invalid_mass:.1%}")
            print("모델의 다음 위치 상위 5개:", [(p, round(probability, 3)) for p, probability in sorted(predicted.items(), key=lambda item: item[1], reverse=True)[:5]])

    def step(_):
        if state["done"]:
            return
        state.update(prediction_position=state["position"], prediction_action=action.value)
        outcome = step_environment(state["maze"], state["position"], action.value, rules, state["rng"].random())
        state["position"], state["done"] = outcome.position, outcome.done
        state["observed_position"] = outcome.position
        state["path"].append(outcome.position)
        state["message"] = f"실제 결과: 위치 {outcome.position}, 보상 {outcome.reward:.2f}, 종료 {outcome.done}"
        predict_button.disabled = step_button.disabled = outcome.done
        render()

    def predict(_):
        if not state["done"]:
            state.update(prediction_position=state["position"], prediction_action=action.value, observed_position=None, message="현재 위치에서 지정 행동의 결과 예측입니다.")
            render()

    predict_button.on_click(predict)
    step_button.on_click(step)
    reset_button.on_click(reset)
    display_legend()
    display(widgets.HTML("<p>이 예측 화면의 파란 점은 <b>행동 전 위치</b>, 주황색 테두리는 <b>실제 이동 결과</b>입니다. 한 걸음 실행 후에도 그 행동의 예측 확률을 유지합니다.</p>"))
    display(widgets.VBox([widgets.HBox([maze_choice, seed, action]), widgets.HBox([predict_button, step_button, reset_button]), output]))
    reset()
    return state


def launch_navigation(checkpoints, config_path="configs/t4.json"):
    config, output = read_config(config_path), widgets.Output()
    rules = config["environment"]
    models = {}
    for name, path in checkpoints.items():
        model, encoder, payload = load_model(path)
        if payload["config"] != config:
            raise ValueError("실행 설정과 체크포인트 설정이 다릅니다.")
        if payload["stage"] == "pretrain":
            raise ValueError("사전학습 모델은 launch_world로 다음 상태를 확인하세요.")
        models[name] = (model, encoder)
    if not models:
        raise ValueError("실행할 SFT 또는 GRPO 체크포인트가 필요합니다.")
    maze_choice = widgets.Dropdown(options=[("위험한 지름길 예시", "demo"), ("처음 보는 무작위 지도", "random")], description="지도")
    seed = widgets.IntText(value=3_000_042, description="지도 seed")
    slippery = widgets.Checkbox(value=True, description="미끄러짐 켜기")
    step_button = widgets.Button(description="한 걸음 실행")
    auto_button = widgets.Button(description="자동 실행")
    reset_button = widgets.Button(description="지도 초기화")
    state = {}

    def render():
        with output:
            clear_output(wait=True)
            figure, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
            for axis, (name, episode) in zip(axes[0], state["episodes"].items()):
                draw_maze(axis, state["maze"], episode["position"], episode["path"], f"{name}: {episode['steps']} steps, return {episode['return']:.2f}")
            plt.tight_layout()
            plt.show()
            for name, episode in state["episodes"].items():
                print(f"{name}: {episode['status']} | 충돌 {episode['collisions']}회")

    def reset(_=None):
        maze = make_demo_maze(rules) if maze_choice.value == "demo" else generate_maze(seed.value, rules)
        state.update(maze=maze, noise=np.random.default_rng(seed.value + 1).random(rules["max_steps"]), slippery=slippery.value)
        state["episodes"] = {name: {"position": maze.start, "path": [maze.start], "return": 0.0, "steps": 0, "collisions": 0, "done": False, "status": "진행 중"} for name in models}
        render()

    def advance():
        advanced = False
        for name, (model, encoder) in models.items():
            episode = state["episodes"][name]
            if episode["done"]:
                continue
            action = int(predict_action_probabilities(model, encoder, [state["maze"]], [episode["position"]]).argmax(-1)[0])
            outcome = step_environment(state["maze"], episode["position"], action, rules, state["noise"][episode["steps"]], state["slippery"])
            episode["position"] = outcome.position
            episode["path"].append(outcome.position)
            episode["return"] += outcome.reward
            episode["steps"] += 1
            episode["collisions"] += int(outcome.collision)
            episode["done"] = outcome.done or episode["steps"] >= rules["max_steps"]
            episode["status"] = "도착" if outcome.success else ("구멍 추락" if outcome.done else ("시간초과" if episode["done"] else "진행 중"))
            advanced = True
        render()
        return advanced

    def autoplay(_):
        auto_button.disabled = step_button.disabled = reset_button.disabled = True
        try:
            for _ in range(rules["max_steps"]):
                if not advance():
                    break
                time.sleep(config["display"]["animation_delay"])
        finally:
            auto_button.disabled = step_button.disabled = reset_button.disabled = False

    step_button.on_click(lambda _: advance())
    auto_button.on_click(autoplay)
    reset_button.on_click(reset)
    display_legend()
    display(widgets.VBox([widgets.HBox([maze_choice, seed, slippery]), widgets.HBox([step_button, auto_button, reset_button]), output]))
    print("지도·seed·미끄러짐 설정을 바꾼 뒤 ‘지도 초기화’를 누르세요. 정책 행동은 argmax입니다.")
    reset()
    return state
