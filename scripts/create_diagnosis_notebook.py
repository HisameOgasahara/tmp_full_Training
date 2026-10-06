"""Generate the checkpoint-only loop diagnosis notebook."""

import json
from pathlib import Path
import textwrap


cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": textwrap.dedent(source).strip().splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": textwrap.dedent(source).strip().splitlines(keepends=True)})


markdown("""
    # 미로 반복 실패 진단: SFT와 PPO 비교

    **런타임 → 모두 실행**으로 학습·검증 지도, 저장된 PPO 구간, 반복 실패와 SFT 보정 상태를 한 번에 비교합니다.
    마지막에 권장 모델, 반복 원인, 다음 변경 우선순위를 담은 `report.md`와 결과 ZIP을 제공합니다.

    체크포인트는 읽기만 하며 학습을 실행하지 않습니다. 결과는 매 실행 새 폴더에 저장합니다.
    샘플링으로 탈출해도 argmax 길찾기가 해결된 것은 아닙니다.
""")

markdown("""
    ## 1. 실행 환경

    Colab에서는 T4를 선택하세요. 준비 셀은 `experiment/loop-diagnosis` 브랜치를 내려받습니다.
    로컬에서는 프로젝트의 uv 환경을 노트북 커널로 선택하고 프로젝트 폴더에서 실행하세요.
""")

code("""
    import hashlib
    import importlib
    import io
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import tarfile
    import tempfile
    import urllib.request
    from datetime import datetime, timezone

    REPO_URL = "https://github.com/HisameOgasahara/tmp_full_Training.git"
    REPO_REF = "experiment/loop-diagnosis"
    IN_COLAB = importlib.util.find_spec("google.colab") is not None if importlib.util.find_spec("google") else False
    if IN_COLAB:
        REPO_ROOT = Path(tempfile.mkdtemp(prefix="maze_diagnosis_")) / "repository"
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF,
                        REPO_URL, str(REPO_ROOT)], check=True)
        uv_executable = shutil.which("uv")
        if uv_executable is None:
            archive_url = "https://github.com/astral-sh/uv/releases/download/0.12.23/uv-x86_64-unknown-linux-gnu.tar.gz"
            data = urllib.request.urlopen(archive_url, timeout=120).read()
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                member = next(item for item in archive.getmembers() if item.name.endswith("/uv") and item.isfile())
                uv_path = REPO_ROOT.parent / "uv"
                uv_path.write_bytes(archive.extractfile(member).read())
            uv_path.chmod(0o755)
            uv_executable = str(uv_path)
        import torch
        subprocess.run([uv_executable, "pip", "install", "--system", "--python", sys.executable,
                        "numpy>=1.26,<3", "matplotlib>=3.8,<4", "ipywidgets>=8.1,<9"], check=True)
        subprocess.run([uv_executable, "pip", "install", "--system", "--python", sys.executable,
                        "--no-deps", "--editable", str(REPO_ROOT)], check=True)
    else:
        REPO_ROOT = next((path for path in (Path.cwd(), *Path.cwd().parents)
                          if (path / "maze_training").is_dir()), None)
        if REPO_ROOT is None:
            raise RuntimeError("프로젝트 폴더에서 노트북을 실행하세요.")
    for name in list(sys.modules):
        if name == "maze_training" or name.startswith("maze_training."):
            del sys.modules[name]
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    importlib.invalidate_caches()
    REVISION = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    import numpy as np
    import torch
    from IPython.display import display, HTML
    from html import escape
    from maze_training.runtime import load_model, choose_device, seed_runtime
    from maze_training.data import build_maps, build_expert_examples
    from scripts.summarize_diagnosis import summarize_results
    from maze_training.environment import (
        ACTION_NAMES, WALL, GOAL, generate_maze, compute_goal_distances,
        move_nominal, step_environment, find_min_cost_path, calculate_path_cost,
    )
    from maze_training.evaluation import predict_action_probabilities
    print("실행 커밋:", REVISION)
    print("장치:", choose_device())
""")

markdown("""
    ## 2. 체크포인트와 실험 조건

    Colab에서는 기본으로 Drive를 연결합니다. 기존 학습 저장 폴더를 그대로 사용합니다.
    시작 SFT와 마지막 PPO를 불러오고, 같은 폴더의 PPO 구간 파일도 자동 비교합니다.
    선택 모델이 SFT여도 마지막 PPO와 비교를 진행합니다.

    실패 seed는 문서의 세 사례입니다. 샘플링은 지도·모델별 20회, 제한은 기존 설정의 64회입니다.
    검증 비교는 기존 검증 지도 전체를 사용합니다. 테스트 지도는 모델 선택에 사용하지 마세요.
""")

code("""
    USE_DRIVE = IN_COLAB
    PROFILE = "t4"
    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
    if USE_DRIVE:
        CHECKPOINT_ROOT = Path("/content/drive/MyDrive/weighted_maze_v2_training") / PROFILE
    elif IN_COLAB:
        CHECKPOINT_ROOT = Path("/content/weighted_maze_v2_runs") / PROFILE
    else:
        CHECKPOINT_ROOT = REPO_ROOT / "runs" / "weighted_maze_v2" / PROFILE
    CHECKPOINTS = {
        "SFT": CHECKPOINT_ROOT / "ppo_refinement" / "initial_sft.pt",
        "PPO": CHECKPOINT_ROOT / "ppo_refinement" / "ppo.pt",
    }
    FAILURE_SEEDS = [3000042, 765563455, 657777688777]
    SAMPLING_REPEATS = 20
    SAMPLING_SEED = 42
    SPLIT = "validation"
    RESULT_ROOT = Path("/content/maze_diagnosis_results") if IN_COLAB else REPO_ROOT / "runs" / "diagnosis"

    def show_table(rows):
        if not rows:
            print("해당 기록이 없습니다.")
            return
        columns = list(rows[0])
        header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
        body = "".join("<tr>" + "".join(f"<td>{escape(str(row[column]))}</td>" for column in columns) + "</tr>" for row in rows)
        display(HTML(f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"))

    def hash_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    if not CHECKPOINTS["SFT"].exists() and (CHECKPOINT_ROOT / "sft.pt").exists():
        CHECKPOINTS["SFT"] = CHECKPOINT_ROOT / "sft.pt"
    for path in CHECKPOINTS.values():
        if not path.exists():
            raise FileNotFoundError(f"필수 학습 파일이 없습니다: {path}. CHECKPOINT_ROOT를 실제 저장 폴더로 지정하세요.")
    for path in sorted(CHECKPOINTS["PPO"].parent.glob("ppo_[0-9][0-9][0-9][0-9].pt")):
        CHECKPOINTS[f"PPO@{int(path.stem.split('_')[1])}"] = path
    models, checkpoint_metadata = {}, {}
    sft_sampling_weights = None
    checkpoint_hashes = set()
    for name, path in CHECKPOINTS.items():
        digest = hash_file(path)
        if digest in checkpoint_hashes:
            continue
        model, encoder, payload = load_model(path)
        if payload["stage"] != name.split("@")[0].lower():
            raise ValueError(f"{name} 경로의 단계는 {payload['stage']}입니다. 비교할 체크포인트를 확인하세요.")
        if name == "SFT" and "sampling_weights" in payload:
            sft_sampling_weights = payload["sampling_weights"].cpu().numpy()
        checkpoint_hashes.add(digest)
        models[name] = (model, encoder)
        checkpoint_metadata[name] = {
            "path": str(path.resolve()), "sha256": digest,
            "stage": payload["stage"], "step": payload["step"], "config": payload["config"],
        }
    config = checkpoint_metadata["SFT"]["config"]
    if any(config != item["config"] for item in checkpoint_metadata.values()):
        raise ValueError("두 체크포인트의 설정이 다릅니다. 같은 실험의 파일을 선택하세요.")
    rules = config["environment"]
    selection_path = CHECKPOINTS["PPO"].parent / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None
    if selection:
        print("같은 폴더의 선택 기록 (지정 파일의 단계·업데이트 수도 확인):", selection)
    show_table([{"모델": name, "단계": item["stage"], "업데이트": item["step"], "파일": item["path"]}
                for name, item in checkpoint_metadata.items()])
    seed_runtime(SAMPLING_SEED)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_DIR = Path(tempfile.mkdtemp(prefix="loop_", dir=RESULT_ROOT))
    print("진단 결과 위치:", RESULT_DIR)
""")

markdown("""
    ## 3. 위치별 확률과 이동 기록

    벽 방향을 제외한 실제 행동 확률을 기록합니다. BFS 정답은 목표까지 거리를 한 칸 줄이는 모든 방향입니다.
    늪을 우회하는 PPO 행동은 BFS 정답과 다를 수 있으므로, BFS 불일치만으로 오류를 판정하지 마세요.
    반복 여부는 같은 위치를 다시 방문했는지이며, 반복이 있어도 샘플링 실행은 성공할 수 있습니다.
""")

code("""
    probability_cache = {}

    def cache_map_probabilities(mazes):
        batch_size = config["evaluation"]["batch_size"]
        for name, (model, encoder) in models.items():
            pending = [(maze, position) for maze in mazes for position in compute_goal_distances(maze)
                       if position != maze.goal and (name, maze.seed, position) not in probability_cache]
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                probabilities = predict_action_probabilities(model, encoder, [item[0] for item in batch],
                                                              [item[1] for item in batch]).cpu().numpy()
                for (maze, position), probabilities_at_state in zip(batch, probabilities):
                    probability_cache[name, maze.seed, position] = probabilities_at_state.astype(float)
            print("행동 확률 준비 완료:", name, "지도", len(mazes), "개", flush=True)

    def get_probabilities(name, maze, position):
        key = (name, maze.seed, position)
        if key not in probability_cache:
            model, encoder = models[name]
            probability_cache[key] = predict_action_probabilities(model, encoder, [maze], [position])[0].cpu().numpy().astype(float)
        return probability_cache[key]

    def run_episode(name, maze, mode="argmax", random_seed=None):
        rng = np.random.default_rng(random_seed)
        distances = compute_goal_distances(maze)
        position, seen = maze.start, {maze.start}
        trace = []
        total_cost, total_return, revisits, collisions = 0.0, 0.0, 0, 0
        success = False
        for step in range(rules["max_steps"]):
            probs = get_probabilities(name, maze, position)
            bfs_actions = [action for action in range(len(ACTION_NAMES))
                           if not move_nominal(maze, position, action)[1]
                           and distances.get(move_nominal(maze, position, action)[0]) == distances[position] - 1]
            if mode == "argmax":
                action = int(probs.argmax())
            elif mode == "sampling":
                action = int(rng.choice(len(probs), p=probs / probs.sum()))
            else:
                raise ValueError("실행 방식은 argmax 또는 sampling입니다.")
            outcome = step_environment(maze, position, action, rules)
            repeated = outcome.position in seen and not outcome.done
            revisits += int(repeated)
            collisions += int(outcome.collision)
            total_cost += outcome.cost
            total_return += outcome.reward
            trace.append({
                "step": step + 1, "position": list(position), "probabilities": probs.tolist(),
                "bfs_actions": bfs_actions, "bfs_probability": float(probs[bfs_actions].sum()),
                "action": action, "next_position": list(outcome.position), "revisit": repeated,
                "cost": outcome.cost, "return": outcome.reward,
            })
            position = outcome.position
            seen.add(position)
            success = outcome.success
            if outcome.done:
                break
        return {"model": name, "maze_seed": maze.seed, "mode": mode, "random_seed": random_seed,
                "success": success, "timeout": not success, "loop": revisits > 0,
                "revisits": revisits, "collisions": collisions, "steps": len(trace),
                "cost": total_cost, "return": total_return, "trace": trace}

    failure_maps = [generate_maze(seed, rules) for seed in FAILURE_SEEDS]
    cache_map_probabilities(failure_maps)
    failure_records = []
    for maze in failure_maps:
        print("지도 seed:", maze.seed, "시작:", maze.start, "목표:", maze.goal)
        print(np.array([".", "#", "~", "G"])[maze.grid])
        for name in models:
            record = run_episode(name, maze)
            failure_records.append(record)
            print(name, "도착:", record["success"], "이동:", record["steps"],
                  "비용:", record["cost"], "재방문:", record["revisits"])
            show_table([{
                "이동": row["step"], "현재": row["position"],
                **{direction: round(probability, 4) for direction, probability in zip(ACTION_NAMES, row["probabilities"])},
                "BFS 정답": [ACTION_NAMES[action] for action in row["bfs_actions"]],
                "정답 확률 합": round(row["bfs_probability"], 4),
                "선택": ACTION_NAMES[row["action"]], "다음": row["next_position"], "재방문": row["revisit"],
            } for row in record["trace"]])
""")

markdown("""
    ## 4. argmax와 반복 샘플링 비교

    두 모델에 같은 지도별 난수 seed 목록을 사용합니다. 각 실행의 seed와 전체 이동 기록을 저장합니다.
    성공률과 반복률을 먼저 보고, 비용은 성공한 실행들의 평균으로 해석하세요.
""")

code("""
    sampling_records, mode_rows = [], []
    for maze in failure_maps:
        for name in models:
            argmax = next(row for row in failure_records if row["model"] == name and row["maze_seed"] == maze.seed)
            sampled = [run_episode(name, maze, "sampling", SAMPLING_SEED + repeat)
                       for repeat in range(SAMPLING_REPEATS)]
            sampling_records.extend(sampled)
            for mode, records in (("argmax", [argmax]), ("sampling", sampled)):
                successful = [row for row in records if row["success"]]
                mode_rows.append({"seed": maze.seed, "모델": name, "방식": mode, "실행 수": len(records),
                                  "도착률": float(np.mean([row["success"] for row in records])),
                                  "반복률": float(np.mean([row["loop"] for row in records])),
                                  "평균 보상": float(np.mean([row["return"] for row in records])),
                                  "성공 시 비용": float(np.mean([row["cost"] for row in successful])) if successful else None})
    show_table(mode_rows)
""")

markdown("""
    ## 5. 검증 지도별 성공 유지·개선·퇴행

    기존 검증 지도 전체를 두 모델에서 argmax로 실행합니다.
    성공 유지: 둘 다 성공, 개선: SFT 실패→PPO 성공, 퇴행: SFT 성공→PPO 실패, 실패 유지: 둘 다 실패입니다.
    두 모델이 모두 성공한 지도에서 비용 차이 `PPO−SFT`가 음수이면 PPO 비용이 작습니다.
    최소 비용 대비 초과 비용도 같은 지도들에서 비교합니다.
""")

code("""
    validation_maps = build_maps(config, SPLIT)
    cache_map_probabilities(validation_maps)
    validation_records, paired_rows = [], []
    labels = {(True, True): "성공 유지", (False, True): "개선", (True, False): "퇴행", (False, False): "실패 유지"}
    for index, maze in enumerate(validation_maps):
        sft, ppo = (run_episode(name, maze) for name in ("SFT", "PPO"))
        validation_records.extend((sft, ppo))
        minimum_cost = calculate_path_cost(maze, find_min_cost_path(maze, rules), rules)
        paired_rows.append({"seed": maze.seed, "변화": labels[sft["success"], ppo["success"]],
                            "SFT 도착": sft["success"], "PPO 도착": ppo["success"],
                            "SFT 반복": sft["loop"], "PPO 반복": ppo["loop"],
                            "SFT 비용": sft["cost"], "PPO 비용": ppo["cost"],
                            "최소 비용": minimum_cost,
                            "비용 차이 PPO−SFT": ppo["cost"] - sft["cost"] if sft["success"] and ppo["success"] else None})
        if (index + 1) % 8 == 0 or index + 1 == len(validation_maps):
            print("검증 진행:", index + 1, "/", len(validation_maps))
    show_table(paired_rows)
    transition_counts = {label: sum(row["변화"] == label for row in paired_rows) for label in labels.values()}
    show_table([{"변화": label, "지도 수": count} for label, count in transition_counts.items()])
    summary_rows = []
    for name in ("SFT", "PPO"):
        records = [row for row in validation_records if row["model"] == name]
        summary_rows.append({"모델": name, "지도 수": len(records),
                             "도착률": float(np.mean([row["success"] for row in records])),
                             "반복률": float(np.mean([row["loop"] for row in records])),
                             "평균 보상": float(np.mean([row["return"] for row in records]))})
    show_table(summary_rows)
    common = [row for row in paired_rows if row["변화"] == "성공 유지"]
    common_costs = {"지도 수": len(common)}
    for name in ("SFT", "PPO"):
        common_costs[f"{name} 비용"] = float(np.mean([row[f"{name} 비용"] for row in common])) if common else None
        common_costs[f"{name} 초과 비용"] = float(np.mean([row[f"{name} 비용"] - row["최소 비용"] for row in common])) if common else None
    common_costs["비용 차이 PPO−SFT"] = float(np.mean([row["비용 차이 PPO−SFT"] for row in common])) if common else None
    show_table([common_costs])
""")

markdown("""
    ## 6. 학습·검증 전체 비교와 보정 상태 자동 진단

    학습 지도 전체와 검증 지도 전체를 같은 argmax 방식으로 평가합니다.
    저장된 PPO 구간마다 SFT 대비 개선·퇴행·공통 성공 비용을 비교합니다.
    검증에서 실패한 지도도 자동으로 20회 샘플링해 실행 방식 차이를 확인합니다.

    SFT 학습 지도에서 방문한 오답 위치를 모아 체크포인트에 저장된 보정 가중치와 대조합니다.
    저장 가중치는 마지막 갱신 시점의 상태이며 과거에 학습된 횟수를 뜻하지 않습니다.
    전체 지도 확률을 배치로 계산하며 진행 상황을 출력합니다.
""")

code("""
    print("학습 지도 전체 생성 중:", config["data"]["train_maps"], "개", flush=True)
    train_maps = build_maps(config, "train")
    cache_map_probabilities(train_maps)
    train_records = []
    for name in models:
        train_records.extend(run_episode(name, maze) for maze in train_maps)
        print("학습 지도 평가 완료:", name, len(train_maps), "개", flush=True)
    all_validation_records = list(validation_records)
    for name in models:
        if name not in ("SFT", "PPO"):
            all_validation_records.extend(run_episode(name, maze) for maze in validation_maps)

    def summarize_episodes(name, records):
        subset = [row for row in records if row["model"] == name]
        return {"model": name, "maps": len(subset),
                "success_rate": float(np.mean([row["success"] for row in subset])),
                "loop_rate": float(np.mean([row["loop"] for row in subset])),
                "mean_return": float(np.mean([row["return"] for row in subset]))}

    train_validation = [{"model": name, "train": summarize_episodes(name, train_records),
                         "validation": summarize_episodes(name, all_validation_records)} for name in models]
    baseline = {row["maze_seed"]: row for row in all_validation_records if row["model"] == "SFT"}
    candidate_comparisons = []
    for name in models:
        candidate = {row["maze_seed"]: row for row in all_validation_records if row["model"] == name}
        pairs = [(baseline[seed], candidate[seed]) for seed in baseline]
        common_pairs = [(a, b) for a, b in pairs if a["success"] and b["success"]]
        candidate_comparisons.append({
            **summarize_episodes(name, all_validation_records),
            "improved_seeds": [a["maze_seed"] for a, b in pairs if not a["success"] and b["success"]],
            "regressed_seeds": [a["maze_seed"] for a, b in pairs if a["success"] and not b["success"]],
            "common_success_maps": len(common_pairs),
            "common_cost_delta": float(np.mean([b["cost"] - a["cost"] for a, b in common_pairs])) if common_pairs else None,
        })
    validation_sampling = []
    for name in models:
        failed_seeds = {row["maze_seed"] for row in all_validation_records if row["model"] == name and not row["success"]}
        for maze in validation_maps:
            if maze.seed in failed_seeds:
                validation_sampling.extend(run_episode(name, maze, "sampling", SAMPLING_SEED + repeat)
                                           for repeat in range(SAMPLING_REPEATS))
        print("검증 실패 지도 샘플링 완료:", name, flush=True)

    encoder = models["SFT"][1]
    expert_queries, _ = build_expert_examples(train_maps, encoder)
    weight_lookup = None
    if sft_sampling_weights is not None:
        if len(sft_sampling_weights) != len(expert_queries):
            raise ValueError("저장된 SFT 보정 가중치 수와 현재 학습 데이터 수가 다릅니다.")
        weight_lookup = {tuple(query): float(weight) for query, weight in zip(expert_queries, sft_sampling_weights)}
    train_map_lookup = {maze.seed: maze for maze in train_maps}
    wrong_states = {}
    for record in train_records:
        if record["model"] != "SFT":
            continue
        maze = train_map_lookup[record["maze_seed"]]
        for row in record["trace"]:
            if row["action"] not in row["bfs_actions"]:
                key = (maze.seed, tuple(row["position"]))
                query = tuple(encoder.encode_query(maze, tuple(row["position"])))
                wrong_states[key] = {"maze_seed": maze.seed, "position": row["position"],
                                     "bfs_probability": row["bfs_probability"],
                                     "saved_weight": weight_lookup[query] if weight_lookup is not None else None}
    correction = {
        "weights_available": weight_lookup is not None,
        "wrong_visited_states": len(wrong_states),
        "weighted_wrong_states": sum(row["saved_weight"] > 1 for row in wrong_states.values()) if weight_lookup is not None else None,
        "saved_weighted_states": int(np.count_nonzero(sft_sampling_weights > 1)) if weight_lookup is not None else None,
        "wrong_states": list(wrong_states.values()),
        "refresh_maps": config["sft"]["correction_maps"], "train_maps": len(train_maps),
        "snapshot_only": True,
    }
    show_table([{"모델": row["model"], "학습 도착률": row["train"]["success_rate"],
                 "검증 도착률": row["validation"]["success_rate"],
                 "학습 반복률": row["train"]["loop_rate"], "검증 반복률": row["validation"]["loop_rate"]}
                for row in train_validation])
    print("보정 가중치 집계:", {key: value for key, value in correction.items() if key != "wrong_states"})
""")

markdown("""
    ## 7. 최종 보고서와 결과 저장

    `manifest.json`에는 실행 커밋, 체크포인트 SHA-256·단계·업데이트 수·설정, 난수 조건을 저장합니다.
    `trajectories.json`에는 모든 실행의 위치별 확률과 이동 기록을,
    `summary.json`에는 실행 방식 비교·지도별 변화·공통 성공 지도의 비용을 저장합니다.

    `report.md`에는 비교한 모델 중 권장 모델과 그 근거, 반복 지점의 확률,
    SFT 보정 상태, 다음 변경 우선순위를 자동 작성합니다. 검증 결과에 따른 권장이며 최종 테스트 성능은 별개입니다.
    Colab에서는 결과 ZIP을 자동으로 다운로드합니다. Drive에도 결과 폴더를 저장합니다.
""")

code("""
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "revision": REVISION,
        "checkpoints": checkpoint_metadata, "selection_file": str(selection_path), "selection": selection,
        "failure_seeds": FAILURE_SEEDS, "sampling_seed": SAMPLING_SEED,
        "sampling_repeats": SAMPLING_REPEATS, "split": SPLIT,
        "validation_seeds": [maze.seed for maze in validation_maps],
        "train_seeds": [maze.seed for maze in train_maps],
        "device": str(choose_device()), "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "max_steps": rules["max_steps"],
    }
    artifacts = {
        "manifest.json": manifest,
        "trajectories.json": {"failure_argmax": failure_records, "failure_sampling": sampling_records,
                              "validation_argmax": all_validation_records, "train_argmax": train_records,
                              "validation_failure_sampling": validation_sampling},
        "summary.json": {"mode_comparison": mode_rows, "validation": summary_rows,
                         "paired_maps": paired_rows, "transitions": transition_counts,
                         "common_success_costs": common_costs, "train_validation": train_validation,
                         "candidate_comparisons": candidate_comparisons, "correction": correction},
    }
    for filename, value in artifacts.items():
        destination = RESULT_DIR / filename
        destination.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        print("저장:", destination)
    report = summarize_results(manifest, artifacts["summary.json"], artifacts["trajectories.json"])
    (RESULT_DIR / "report.md").write_text(report, encoding="utf-8")
    from IPython.display import Markdown
    display(Markdown(report))
    archive_path = shutil.make_archive(str(RESULT_DIR), "zip", RESULT_DIR)
    if USE_DRIVE:
        saved_dir = CHECKPOINT_ROOT / "diagnosis_results" / RESULT_DIR.name
        shutil.copytree(RESULT_DIR, saved_dir)
        shutil.copy2(archive_path, saved_dir.parent)
        print("Drive 결과 저장:", saved_dir)
    print("전체 결과 ZIP:", archive_path)
    if IN_COLAB:
        from google.colab import files
        files.download(archive_path)
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "name": "Maze_Diagnosis.ipynb"}, "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 4,
}
destination = Path(__file__).resolve().parents[1] / "notebooks" / "Maze_Diagnosis.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(destination)
