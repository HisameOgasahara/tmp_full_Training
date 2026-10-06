# 비용이 다른 미로의 3단계 Transformer 학습

작은 Transformer 하나로 **이동 위치·비용 사전학습 → BFS SFT → PPO**를 실습합니다. 벽을 피해 목표에 도착하고, 짧은 늪길과 긴 일반 길 중 총비용이 작은 경로를 학습합니다.

<!-- COLAB_BADGE_START -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_full_Training/blob/6bd7e47582f36d7261d7962f025e8b663053f9c3/notebooks/Maze_Training.ipynb)
<!-- COLAB_BADGE_END -->

Colab에서 **T4 GPU**를 선택해 실행하세요.

## 단계별 목표

| 단계 | 학습 목표 | 확인할 능력 |
|---|---|---|
| 사전학습 | 무작위 전이의 다음 위치와 비용 예측 | 벽 충돌, 일반 바닥·늪의 비용 이해 |
| SFT | 최단 경로의 행동 집합과 실패 위치 보정 | 벽을 피해 목표로 가는 최단 경로 |
| PPO | 분리된 가치 학습·GAE·SFT 기준 KL | 목표 도착을 유지하며 총비용 절감 |

지도는 BFS 경로보다 길지만 비용이 작은 우회로를 갖도록 생성합니다. 이동은 결정적이며 전체 지도와 현재 위치를 매번 입력합니다. 비교 기준은 BFS와 최소 비용 경로를 구하는 Dijkstra입니다.

행동 선택은 바로 옆 벽과 지도 밖 방향을 제외합니다. SFT는 실제 막히는 위치를 더 자주 학습하며, PPO의 가치 손실은 행동 Transformer를 바꾸지 않습니다. 학습·검증 행동 정확도와 반복률로 길찾기를 확인합니다.

모델 선택은 검증 도착률을 우선하고 동점이면 평균 보상을 비교합니다. 시작 SFT도 후보에 포함합니다. 환경 규칙과 구현 구조는 [학습설계](docs/학습설계.md)에 있습니다.

## 설정과 구조

- 모델: 4층, 차원 128, attention head 4개, 파라미터 809,286개.
- 출력: 다음 위치·행동 토큰, 비용 출력, 가치 출력. 입력 최대 54토큰, 어휘 68개.
- 설정: `configs/t4.json`. 짧은 실행은 `configs/smoke.json`.
- 저장: `/content/weighted_maze_v2_runs/t4`, Drive는 `weighted_maze_v2_training/t4`.
- 이 환경은 새 체크포인트로 학습합니다. 이번 보정 학습은 새 저장 폴더에서 시작합니다. 이전 체크포인트는 원래 폴더에 보관됩니다.

| 경로 | 역할 |
|---|---|
| `notebooks/Maze_Training.ipynb` | 단계별 학습·평가·실행 화면 |
| `maze_training/environment.py` | 미로 생성, 이동 비용, BFS·Dijkstra |
| `maze_training/model.py` · `encoding.py` | Transformer와 입력·출력 표현 |
| `maze_training/data.py` | 지도 분리와 전이·BFS 데이터 |
| `maze_training/policy.py` | 학습·평가·실행에 공통으로 적용하는 벽 방향 제외 |
| `maze_training/training.py` | 사전학습·SFT·PPO/GAE, 저장·재개 |
| `maze_training/evaluation.py` | 도착률·비용·최소 비용 대비 초과 평가 |
| `maze_training/notebook.py` | 보고서와 실행 화면 |
| `scripts/create_notebook.py` | 노트북 셀 원본 |

노트북은 `scripts/create_notebook.py`를 수정하고 `python scripts/create_notebook.py`로 생성합니다.

## 로컬 실행

Python 3.10 이상과 [uv](https://docs.astral.sh/uv/)를 사용합니다.

```cmd
git clone https://github.com/HisameOgasahara/tmp_full_Training.git
cd tmp_full_Training
uv sync
uv run python -m maze_training.training pretrain --config configs/t4.json --output-dir runs/weighted_maze_v2/t4
uv run python -m maze_training.training sft --config configs/t4.json --output-dir runs/weighted_maze_v2/t4
uv run python -m maze_training.training ppo --config configs/t4.json --output-dir runs/weighted_maze_v2/t4
uv run python -m maze_training.cli navigation --config configs/t4.json --output-dir runs/weighted_maze_v2/t4 --split test
```

이어 학습은 `--resume`, 총 업데이트 수 지정은 `--steps`를 사용합니다. 같은 설정과 출력 폴더로 재개하세요. 검사: `uv run --extra dev python -m unittest discover -s tests -v`.
