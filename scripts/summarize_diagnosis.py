"""Produce a Korean decision report from saved diagnostic artifacts."""

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from maze_training.environment import ACTION_NAMES


def build_report_relative_path(manifest):
    timestamp = datetime.fromisoformat(manifest["created_at_utc"].replace("Z", "+00:00"))
    korean_time = timestamp.astimezone(timezone(timedelta(hours=9)))
    return Path("docs") / "진단보고서" / korean_time.strftime("%Y-%m-%d") / korean_time.strftime("%H%M%S_%f_진단보고서.md")


def find_cycle(record):
    visited = {}
    for index, step in enumerate(record["trace"]):
        position = tuple(step["position"])
        if position in visited:
            return record["trace"][visited[position]:index]
        visited[position] = index
    return []


def summarize_results(manifest, summary, trajectories):
    candidates = summary.get("candidate_comparisons")
    if candidates is None:
        candidates = []
        for row in summary["validation"]:
            name = row["모델"]
            candidates.append({
                "model": name, "maps": row["지도 수"], "success_rate": row["도착률"],
                "loop_rate": row["반복률"], "mean_return": row["평균 보상"],
                "improved_seeds": [item["seed"] for item in summary["paired_maps"] if name != "SFT" and item["변화"] == "개선"],
                "regressed_seeds": [item["seed"] for item in summary["paired_maps"] if name != "SFT" and item["변화"] == "퇴행"],
                "common_success_maps": summary["common_success_costs"]["지도 수"],
                "common_cost_delta": summary["common_success_costs"]["비용 차이 PPO−SFT"] if name != "SFT" else 0.0,
            })
    best = max(candidates, key=lambda row: (row["success_rate"], row["mean_return"]))
    sft = next(row for row in candidates if row["model"] == "SFT")
    timestamp = datetime.fromisoformat(manifest["created_at_utc"].replace("Z", "+00:00"))
    korean_time = timestamp.astimezone(timezone(timedelta(hours=9)))
    lines = ["# 미로 학습 진단 결과", "", f"진단 일시: {korean_time:%Y-%m-%d %H:%M:%S} (한국 시간)", "",
             f"**권장 모델: {best['model']}**. 같은 검증 지도 {best['maps']}개에서 도착률을 우선하고 평균 보상으로 동점을 비교했습니다.", "",
             "## 모델 비교", "",
             "| 모델 | 업데이트 | 도착률 | 반복률 | 평균 보상 | 개선 / 퇴행 | 공통 성공 비용 차이 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in candidates:
        metadata = manifest["checkpoints"][row["model"]]
        delta = row["common_cost_delta"]
        delta_text = f"{delta:+.3f}" if delta is not None else "공통 성공 없음"
        lines.append(f"| {row['model']} | {metadata['step']} | {row['success_rate']:.1%} | {row['loop_rate']:.1%} | "
                     f"{row['mean_return']:.3f} | {len(row['improved_seeds'])} / {len(row['regressed_seeds'])} | {delta_text} |")
    lines += ["", "비용 차이는 두 모델이 모두 성공한 동일 지도에서 후보 비용−SFT 비용입니다. 음수이면 비용이 줄었습니다.", ""]
    selection = manifest.get("selection")
    if selection:
        lines += [f"학습 중 저장된 선택 기록은 {selection['source']} {selection['step']}회입니다.", ""]

    validation = trajectories["validation_argmax"]
    baseline_records = {record["maze_seed"]: record for record in validation if record["model"] == "SFT"}
    lines += ["## PPO에서 생긴 퇴행", ""]
    regressions_found = False
    for candidate in candidates:
        name = candidate["model"]
        for seed in candidate["regressed_seeds"]:
            regressions_found = True
            record = next(item for item in validation if item["model"] == name and item["maze_seed"] == seed)
            sft_states = {tuple(step["position"]): step for step in baseline_records[seed]["trace"]}
            divergence = next((step for step in record["trace"] if tuple(step["position"]) in sft_states
                               and step["action"] != sft_states[tuple(step["position"])]["action"]), None)
            if divergence:
                before = sft_states[tuple(divergence["position"])]
                action = before["action"]
                lines += [f"- {name}, seed `{seed}`: `{tuple(divergence['position'])}`에서 "
                          f"SFT의 {ACTION_NAMES[action]} 선택이 {ACTION_NAMES[divergence['action']]} 선택으로 바뀌었습니다. "
                          f"{ACTION_NAMES[action]} 확률은 {before['probabilities'][action]:.1%}→"
                          f"{divergence['probabilities'][action]:.1%}이며 이후 반복으로 시간초과했습니다."]
            else:
                lines += [f"- {name}, seed `{seed}`: SFT 성공에서 반복 시간초과로 바뀌었습니다."]
    if not regressions_found:
        lines += ["이번 검증 지도에서는 SFT 성공→PPO 실패가 없습니다."]

    lines += ["", "## 반복 원인과 샘플링 효과", ""]
    failed_records = [record for record in trajectories["failure_argmax"] if not record["success"]]
    if not failed_records:
        lines += ["지정한 실패 사례 지도에서는 이번 모델의 argmax 실패가 없습니다."]
    for record in failed_records:
        cycle = find_cycle(record)
        wrong = next((step for step in cycle if step["action"] not in step["bfs_actions"]), None)
        if wrong is None:
            continue
        directions = "·".join(ACTION_NAMES[action] for action in wrong["bfs_actions"])
        cycle_positions = " → ".join(str(tuple(step["position"])) for step in cycle)
        lines += [f"- {record['model']}, seed `{record['maze_seed']}`: `{cycle_positions}` 순환. "
                  f"`{tuple(wrong['position'])}`에서 {ACTION_NAMES[wrong['action']]}가 "
                  f"{wrong['probabilities'][wrong['action']]:.1%}, BFS 정답 {directions}의 확률 합은 {wrong['bfs_probability']:.1%}입니다."]
    sampled = trajectories.get("failure_sampling", []) + trajectories.get("validation_failure_sampling", [])
    sampling_pairs = []
    for record in failed_records + [item for item in validation if not item["success"]]:
        key = (record["model"], record["maze_seed"])
        if key not in sampling_pairs:
            sampling_pairs.append(key)
    for name, seed in sampling_pairs:
        runs = [record for record in sampled if record["model"] == name and record["maze_seed"] == seed]
        if runs:
            successes = sum(record["success"] for record in runs)
            lines += [f"- {name}, seed `{seed}`: argmax 실패, 샘플링 {len(runs)}회 중 {successes}회 성공."]
    lines += ["", "샘플링 성공은 탈출 행동에 확률이 남아 있다는 증거입니다. argmax의 잘못된 최댓값 선택은 그대로 남으며, 샘플링 비용도 별도로 봐야 합니다."]

    lines += ["", "## SFT 학습과 보정 범위", ""]
    comparisons = summary.get("train_validation", [])
    if comparisons:
        for row in comparisons:
            train, val = row["train"], row["validation"]
            lines += [f"- {row['model']}: 학습 {train['maps']}개 도착률 {train['success_rate']:.1%}, "
                      f"검증 {val['maps']}개 도착률 {val['success_rate']:.1%}; "
                      f"학습 반복률 {train['loop_rate']:.1%}, 검증 반복률 {val['loop_rate']:.1%}."]
    else:
        lines += ["이 실행은 학습 지도 이동 결과를 포함하지 않으므로 과적합 여부는 판정하지 않습니다."]
    correction = summary.get("correction")
    if correction and correction["weights_available"]:
        lines += [f"학습 실행에서 방문한 SFT 오답 위치 {correction['wrong_visited_states']}개 중 "
                  f"마지막 저장 가중치가 강화된 위치는 {correction['weighted_wrong_states']}개입니다. "
                  "이 값은 마지막 보정 상태이며 과거 학습 횟수는 아닙니다."]
    else:
        lines += ["저장된 보정 가중치 집계가 없어 실제 보정 상태의 수치는 판정하지 않습니다."]
    settings = manifest["checkpoints"]["SFT"]["config"]
    lines += [f"현재 보정 구현은 갱신마다 가중치를 1로 초기화하고 학습 지도 "
              f"{settings['data']['train_maps']}개 중 {settings['sft']['correction_maps']}개를 복원추출합니다. "
              "이전 오답 위치의 강화가 다음 갱신에도 누적되지 않습니다. 이는 구현의 한계이며 실패 전체의 단일 원인으로 단정하지 않습니다."]

    lines += ["", "## 다음 변경 우선순위", ""]
    if best["model"] == "SFT":
        lines += ["1. 현재 사용할 모델은 SFT로 유지합니다. 비교한 PPO가 검증 선택 기준을 개선하지 못했습니다."]
    else:
        lines += [f"1. 현재 사용할 모델은 {best['model']}로 선택합니다. 비교한 체크포인트 중 검증 선택 기준이 가장 높습니다."]
    if sft["success_rate"] < 1 or any(record["model"] == "SFT" for record in failed_records):
        lines += ["2. 첫 학습 변경은 SFT 보정입니다. 학습 지도에서 발견한 오답 위치를 누적 보관하고 "
                  "후속 갱신에서도 학습에 계속 반영하도록 바꿉니다. 검증·테스트 실패 위치는 학습 정답에 넣지 않습니다."]
    if regressions_found:
        lines += ["3. PPO 변경은 도착 능력 유지에 우선순위를 둡니다. SFT의 학습 지도 BFS 예제를 보조 손실로 함께 사용해 "
                  "잘되던 방향의 확률이 뒤집히는 퇴행을 줄이는 실험을 우선합니다."]
    lines += ["", "방문 이력 추가와 모델 확대는 이번 결과만으로 우선할 근거가 없습니다. "
              "위 변경은 권장 실험이며 아직 개선 효과를 측정한 결과는 아닙니다.", "",
              "권장 모델은 검증 기준의 선택입니다. 새 학습 변경의 효과와 최종 테스트 성능은 이 보고서의 범위를 벗어납니다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    report = summarize_results(read(args.manifest), read(args.summary), read(args.trajectories))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
