# 미끄러운 미로의 3단계 Transformer 학습

작은 Transformer 하나를 **환경 규칙 사전학습 → BFS 행동 지도학습(SFT) → 환경 상호작용 GRPO** 순서로 학습합니다. 환경의 이동 규칙, 목표로 가는 행동, 위험과 이동 비용을 고려한 행동을 단계별로 익히는 것이 목적입니다. 각 단계가 끝나면 저장된 모델을 직접 실행하고 다음 단계로 넘어갑니다.

<!-- COLAB_BADGE_START -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_full_Training/blob/e508bb3149fa5ad7ab24fdc42bc28725f2981a6d/notebooks/Maze_Training.ipynb)
<!-- COLAB_BADGE_END -->

## Colab 실행 순서

1. 배지를 눌러 노트북을 열고 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다.
2. **0. 최신 코드 준비** 셀을 실행합니다. 최신 `main`을 복제하고 실행 커밋과 장치를 표시합니다.
3. 설정 셀에서 `PROFILE = "t4"`, `RESUME = False`를 설정합니다.
4. Drive에 저장하려면 학습 전에 선택적 저장 셀에서 `USE_DRIVE = True`를 설정하고 실행합니다.
5. **1. 사전학습** 셀을 실행합니다. 다음 평가 셀에서 행동을 골라 다음 위치의 예측 확률과 실제 이동을 비교합니다.
6. **2. SFT** 셀을 실행합니다. 다음 평가 셀에서 모델을 이동시키고, 미끄러짐을 끈 경우와 켠 경우를 비교합니다.
7. **3. GRPO** 셀을 실행합니다. 다음 평가 셀에서 SFT와 GRPO 모델을 같은 지도에서 나란히 실행합니다.
8. **4. 최종 테스트** 셀을 실행해 도착률·추락률·평균 보상을 비교합니다.

학습 셀과 바로 아래 평가 셀을 한 단계씩 실행하세요. 실행 화면을 확인한 뒤 다음 단계의 학습 셀을 실행합니다.

## 단계별 목표와 실행 화면

| 단계 | 학습 목표 | 완료 후 실행 | 저장 파일 |
|---|---|---|---|
| 사전학습 | 지도·현재 위치·행동에서 다음 위치의 분포 예측 | 행동을 지정해 예측 분포와 실제 이동 확인 | `pretrain.pt` |
| SFT | 목표로 향하는 BFS의 다음 행동 모방 | 미끄러짐을 끄고 켜서 모델의 이동 확인 | `sft.pt` |
| GRPO | 실제 성공·추락·이동·충돌 보상으로 행동 학습 | SFT와 같은 지도에서 행동과 보상 비교 | `grpo.pt` |

1단계는 `launch_world`, 2·3단계는 `launch_navigation` 화면을 사용합니다. 지도·seed·미끄러짐 설정을 바꾼 뒤 **지도 초기화**를 누르세요. 전체 지도와 현재 위치가 매 이동마다 모델에 주어집니다.

## 설정과 저장

| 항목 | 설정 |
|---|---|
| 전체 학습 | `configs/t4.json`: 사전학습 1,500회, SFT 2,000회, GRPO 250회 |
| 짧은 실행 | `configs/smoke.json`: 사전학습 4회, SFT 4회, GRPO 2회 |
| 모델 | 4층, 차원 128, attention head 4개, 파라미터 809,157개 |
| 입력 | 타일·위치·행동을 69개 토큰으로 표현, 최대 54토큰 |
| GPU 계산 | CUDA FP16 autocast와 GradScaler, Colab의 기존 PyTorch 사용 |
| 기본 저장 위치 | `/content/maze_runs/t4` |
| Drive 저장 위치 | `/content/drive/MyDrive/maze_training/t4` |

같은 단계의 새 학습은 해당 체크포인트와 로그를 덮어씁니다. 단계별 체크포인트는 각각 보관합니다. 중단한 단계를 이어가려면 같은 설정과 출력 폴더에서 `RESUME = True`로 실행하세요. 새 단계로 넘어갈 때는 `False`로 설정합니다. 설정을 바꾼 실험은 새 출력 폴더를 사용하세요.

세션 종료 후 모델을 다시 사용하려면 Drive 저장을 선택하거나 마지막 셀에서 `DOWNLOAD_RESULTS = True`로 ZIP을 내려받으세요. ZIP을 같은 출력 폴더에 복원하면 평가와 실행 화면을 다시 사용할 수 있습니다.

## 최신 코드 사용

푸시 후 GitHub Actions가 Colab 배지를 새 커밋 주소로 갱신합니다. 노트북의 준비 셀도 실행 시점의 최신 `main`을 새 폴더에 복제합니다. 새 세션은 현재 README 배지로 열고 준비 셀부터 실행하세요. 한 실험에서는 처음 가져온 코드를 계속 사용하고, 체크포인트와 실행 커밋을 함께 기록하세요.

## 로컬 실행

Python 3.10 이상과 [uv](https://docs.astral.sh/uv/)를 사용합니다. GPU 실행에는 장치에 맞는 PyTorch를 설치하세요.

```cmd
git clone https://github.com/HisameOgasahara/tmp_full_Training.git
cd tmp_full_Training
uv sync
uv run python -m maze_training.training pretrain --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.cli world --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.training sft --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.cli navigation --config configs/t4.json --output-dir runs/t4 --dry
uv run python -m maze_training.cli navigation --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.training grpo --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.cli navigation --config configs/t4.json --output-dir runs/t4 --split test
```

짧게 실행하려면 같은 명령에서 `configs/smoke.json`, `runs/smoke`를 사용합니다. 이어 학습에는 `--resume`을 추가합니다. `--steps`에는 도달할 총 업데이트 수를 지정합니다.

## 파일 구조와 수정 순서

| 경로 | 역할 |
|---|---|
| `notebooks/Maze_Training.ipynb` | Colab 준비, 단계별 학습·평가·실행 화면 |
| `configs/` | 환경, 모델, 단계별 학습 예산과 평가 설정 |
| `maze_training/environment.py` | 지도 생성, 실제 이동, 전이확률, BFS |
| `maze_training/model.py` · `encoding.py` | 공유 Transformer와 토큰 표현 |
| `maze_training/data.py` | 지도 분리, 전이 기록, BFS 학습 예제 |
| `maze_training/training.py` | 사전학습·SFT·GRPO, 체크포인트 저장과 재개 |
| `maze_training/evaluation.py` | 환경 예측 지표, 정책 비교, 유한 시간 최적 정책 |
| `maze_training/notebook.py` | 보고서, 학습 곡선, 실행 화면 |
| `maze_training/cli.py` | 터미널에서 환경 예측과 정책 실행 |
| `scripts/create_notebook.py` | 노트북 셀의 원본과 생성 코드 |
| `scripts/update_colab_badge.py` | 현재 커밋 주소로 Colab 배지 갱신 |
| `docs/학습설계.md` | 환경 규칙, 단계별 목표와 구현 구조 |

노트북 셀을 바꿀 때는 `scripts/create_notebook.py`를 수정한 뒤 `python scripts/create_notebook.py`로 노트북을 생성하세요. 학습과 환경 로직은 `maze_training/`에서 수정합니다. 코드 검사 명령은 `uv run --extra dev python -m unittest discover -s tests -v`입니다.
