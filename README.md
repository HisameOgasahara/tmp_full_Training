# 미끄러운 미로에서 배우는 3단계 Transformer 학습

작은 트랜스포머 하나로 **환경 규칙 사전학습 → BFS 행동 지도학습(SFT) → 실제 이동 결과로 GRPO**를 실행합니다. 각 단계는 독립된 셀이며, 다음 단계 전에 저장된 모델을 직접 실행할 수 있습니다.

<!-- COLAB_BADGE_START -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_full_Training/blob/main/notebooks/Maze_Training.ipynb)
<!-- COLAB_BADGE_END -->

## 시작하기

1. 배지를 눌러 Colab을 엽니다. 런타임에서 GPU를 선택합니다.
2. **준비 셀**을 실행합니다. 최신 `main`을 새 폴더로 복제하고 실제 실행 커밋을 출력합니다.
3. 설정 셀에서 `PROFILE = "t4"`를 유지합니다. `smoke`는 연결 확인용이며 성능 학습용이 아닙니다.
4. **1단계 학습 → 평가 → 예측 화면**을 실행합니다. 행동을 골라 모델 예측과 실제 다음 위치를 비교합니다.
5. **2단계 학습 → 평가 → 이동 화면**을 실행합니다. 미끄러짐을 끄고 켜서 경로를 확인합니다.
6. **3단계 학습 → 평가 → 나란히 이동 화면**을 실행합니다. SFT와 GRPO 모델을 같은 지도에서 비교합니다.

각 학습 셀은 다음 단계를 자동 실행하지 않습니다. `모두 실행` 대신 확인하려는 단계까지 셀을 순서대로 실행하세요.

## 단계별 결과

| 단계 | 배울 것 | 완료 후 직접 확인 | 파일 |
|---|---|---|---|
| 사전학습 | 지도·위치·행동에서 다음 위치의 분포 예측 | 벽 충돌·미끄러짐의 예측 확률과 환경의 정확한 확률 | `pretrain.pt` |
| SFT | 구멍·벽을 피하는 BFS 행동 모방 | 목표까지 이동; 확률적 미끄러짐에서의 실패 | `sft.pt` |
| GRPO | 실제 성공·실패·이동 비용을 고려한 행동 | SFT와 도착률·추락률·평균 보상 비교 | `grpo.pt` |

사전학습 모델의 행동 출력은 아직 학습되지 않았습니다. 1단계에서는 **이동 결과 예측 화면**을 사용합니다. 전체 지도와 현재 위치가 매번 주어지는 완전 관측 문제이며, 모델은 한 번에 다음 행동 하나를 선택합니다.

## 최신 노트북과 코드

GitHub Actions가 푸시 후 Colab 배지 링크를 새 커밋 SHA로 갱신합니다. 커밋마다 노트북 URL이 달라져 이전 주소의 캐시를 재사용하지 않습니다. 자동 갱신 커밋은 README 배지만 바꾸므로 링크가 가리키는 버전과 학습 코드가 같습니다. 워크플로 완료까지 잠깐 지연될 수 있습니다.

노트북의 첫 셀은 링크의 커밋과 별도로 **실행 시점의 최신 `main`을 새로 복제**합니다. 오래전에 저장한 노트북에서도 준비 셀을 다시 실행하면 최신 Python 코드와 설정을 가져옵니다. 같은 실험 중에는 이 셀을 다시 실행하지 마세요. 코드를 바꾸고 이전 체크포인트를 섞으면 재현 조건이 달라질 수 있습니다. 노트북의 셀 구성 자체가 변경됐을 때는 저장한 사본 대신 현재 README 배지로 다시 여세요.

## 실행 규모와 저장

`configs/t4.json`은 4층, 차원 128, 4개 attention head의 작은 causal decoder 설정입니다. 일반 언어 토크나이저 없이 타일·위치·행동을 약 70개 토큰으로 표현하며 입력은 54토큰 이하입니다. CUDA에서는 FP16 autocast와 GradScaler를 사용합니다. Colab의 기존 PyTorch를 유지합니다.

T4 16GB용으로 설계했지만 T4에서 실행 시간·수렴·성능 개선을 확인한 결과는 아닙니다. 실제 사용 장치와 파라미터 수는 준비 셀과 학습 로그에 표시됩니다. 무료 Colab의 GPU 배정과 이용 시간은 달라질 수 있습니다.

CPU에서 줄인 예산으로 단계 연결과 실제 학습·평가를 확인했습니다. 해당 실행의 테스트 도착률은 SFT 62.5%, GRPO 후 37.5%로 **RL 개선을 확인하지 못했습니다**. 상세 조건은 [검증 기록](docs/검증결과.md)에 있습니다. 기본 예산의 결과나 여러 seed의 효과를 미리 주장하지 않습니다.

기본 저장 위치는 Colab의 `/content/maze_runs/t4`입니다. 런타임이 삭제되면 이 파일도 사라집니다. 노트북의 선택적 Drive 셀을 실행하면 `/content/drive/MyDrive/maze_training/t4`에 저장합니다. 같은 단계 재실행은 그 단계 체크포인트와 로그를 덮어씁니다. 단계별 파일은 서로 덮어쓰지 않습니다. 주기적 저장 후 중단됐다면 `RESUME = True`로 같은 설정의 학습을 이어갑니다.

## 로컬 실행

Python 3.10 이상과 [uv](https://docs.astral.sh/uv/)를 사용합니다. 아래 명령은 Windows cmd에서도 실행할 수 있습니다. 로컬 GPU 환경은 장치에 맞는 PyTorch가 필요합니다.

```cmd
git clone https://github.com/HisameOgasahara/tmp_full_Training.git
cd tmp_full_Training
uv sync
uv run python -m maze_training.training pretrain --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.cli world --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.training sft --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.cli navigation --config configs/t4.json --output-dir runs/t4 --dry
uv run python -m maze_training.training grpo --config configs/t4.json --output-dir runs/t4
uv run python -m maze_training.cli navigation --config configs/t4.json --output-dir runs/t4 --split test
```

연결 확인은 같은 명령에서 `configs/smoke.json`, `runs/smoke`를 사용합니다. 이어 학습에는 `--resume`을 추가합니다. `--steps`는 추가 횟수가 아닌 도달할 총 업데이트 수입니다.

코드 검증은 `uv run --extra dev python -m unittest discover -s tests -v`로 실행합니다.

## 구성

| 경로 | 역할 |
|---|---|
| `notebooks/Maze_Training.ipynb` | Colab 준비, 단계별 학습·평가·직접 실행 |
| `maze_training/environment.py` | 지도 생성, 실제 이동, 정확한 전이확률, BFS |
| `maze_training/model.py` · `encoding.py` | 공유 트랜스포머와 토큰 표현 |
| `maze_training/data.py` | 분리된 지도와 전이·BFS 학습 예제 |
| `maze_training/training.py` | 사전학습, SFT, 환경 상호작용 GRPO, 저장·재개 |
| `maze_training/evaluation.py` | 확률 예측 평가, 정책 비교, 유한 시간 최적 정책 |
| `maze_training/notebook.py` | 단계별 보고서·학습 곡선·실행 화면 |
| `docs/학습설계.md` | 규칙, 목표, 얻고자 하는 능력, 단계 효과 확인 방법 |
| `docs/검증결과.md` | 실제 수행한 검증과 아직 확인하지 못한 사항 |

## 참고

- [FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/): 확률적 미끄러짐이 있는 작은 격자 환경.
- [Maze Transformer](https://github.com/understanding-search/maze-transformer): 미로를 이용한 자기회귀 Transformer 학습·분석.
- [TRL GRPO 설명](https://huggingface.co/docs/trl/grpo_trainer): 그룹별 보상 비교와 clipped policy optimization.

이 저장소는 자체 구현한 교육용 설계입니다. 위 프로젝트의 학습 결과나 성능을 그대로 재현한 실험은 아닙니다.
