# 미끄러운 미로의 3단계 Transformer 학습

작은 Transformer 하나로 **환경 규칙 사전학습 → BFS 행동 지도학습(SFT) → 환경 상호작용 GRPO**를 실습합니다. 단계마다 모델을 직접 실행해 이동 규칙 예측, 기본 길찾기, 위험과 이동 비용을 고려한 행동을 확인합니다.

<!-- COLAB_BADGE_START -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_full_Training/blob/081500fd7af43494c1edbf832b18eb33e8408e93/notebooks/Maze_Training.ipynb)
<!-- COLAB_BADGE_END -->

Colab에서 **T4 GPU**를 선택해 실행하세요.

## 단계별 목표

| 단계 | 학습 데이터와 목표 | 확인할 능력 |
|---|---|---|
| 사전학습 | 무작위 이동 기록으로 다음 위치 분포 예측 | 벽 충돌과 미끄러짐의 규칙 이해 |
| SFT | BFS의 다음 행동 모방 | 벽·구멍을 피해 목표로 이동 |
| GRPO | 후보 에피소드의 실제 보상 비교 | 추락 위험과 이동 비용을 반영한 행동 |

전체 지도와 현재 위치를 매 이동마다 입력하고 다음 행동 하나를 선택합니다. GRPO의 검증 선택에는 시작 SFT도 포함하며, 평균 보상이 높은 모델을 보관하고 동점이면 도착률로 선택합니다.

환경 규칙, 학습 방식, 평가 지표는 [학습설계](docs/학습설계.md)에 정리했습니다.

## 설정과 구조

- 모델: 4층, 차원 128, attention head 4개, 파라미터 809,157개.
- 입력: 타일·위치·행동 69개 토큰, 최대 입력 길이 54토큰.
- 학습 설정: `configs/t4.json`. 짧은 실행은 `configs/smoke.json`.
- 체크포인트를 이어 쓰려면 설정과 출력 폴더를 유지합니다. 설정을 바꾸는 실험은 새 출력 폴더를 사용합니다.

| 경로 | 역할 |
|---|---|
| `notebooks/Maze_Training.ipynb` | 단계별 학습·평가·실행 화면 |
| `maze_training/environment.py` | 지도 생성, 이동 규칙, 전이확률, BFS |
| `maze_training/model.py` · `encoding.py` | Transformer와 토큰 표현 |
| `maze_training/data.py` | 지도 분리와 학습 데이터 생성 |
| `maze_training/training.py` | 사전학습·SFT·GRPO, 저장·재개 |
| `maze_training/evaluation.py` | 평가 지표와 최적 정책 |
| `maze_training/notebook.py` | 보고서와 실행 화면 |
| `maze_training/cli.py` | 터미널 실행 |
| `scripts/create_notebook.py` | 노트북 셀 원본 |

노트북을 수정할 때는 `scripts/create_notebook.py`를 바꾸고 `python scripts/create_notebook.py`로 생성합니다.

## 로컬 실행

Python 3.10 이상과 [uv](https://docs.astral.sh/uv/)를 사용합니다.

```cmd
git clone https://github.com/HisameOgasahara/tmp_full_Training.git
cd tmp_full_Training
uv sync
uv run python -m maze_training.training pretrain --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.training sft --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.training grpo --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.cli navigation --config configs/t4.json --output-dir runs/t4 --split test
```

이어 학습은 `--resume`, 총 업데이트 수 지정은 `--steps`를 사용합니다. 코드 검사: `uv run --extra dev python -m unittest discover -s tests -v`.
