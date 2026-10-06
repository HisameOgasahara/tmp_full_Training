"""Generate a clean, output-free Colab notebook from readable cell sources."""

import json
from pathlib import Path
import textwrap

cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": textwrap.dedent(source).strip().splitlines(keepends=True)})


markdown("""
    # 미끄러운 미로: 사전학습 → SFT → GRPO

    환경의 이동 규칙, 목표로 가는 행동, 위험과 이동 비용을 고려한 행동을 순서대로 학습합니다.
    Colab에서 T4 GPU를 선택하고 준비 셀부터 실행하세요.
    각 단계는 학습 셀과 바로 아래 평가 셀을 실행해 모델을 직접 확인한 뒤 다음 단계로 넘어갑니다.

    - 사전학습: 행동을 지정하고 다음 위치의 예측 확률과 실제 이동을 비교합니다.
    - SFT: 미끄러짐을 끄고 켜서 목표로 이동하는 행동을 확인합니다.
    - GRPO: 같은 지도에서 SFT와 GRPO 모델을 나란히 실행합니다.

    [환경 규칙과 단계별 목표](https://github.com/HisameOgasahara/tmp_full_Training/blob/main/docs/학습설계.md)
""")
markdown("""
    ## 0. 최신 코드 준비

    Colab 런타임을 T4 GPU로 설정한 뒤 실행하세요. 이 셀은 최신 `main`을 새 폴더에 복제합니다.
    준비 완료 후 표시되는 커밋을 실험 기록에 저장하세요. 한 실험에서는 처음 가져온 코드를 계속 사용합니다.
    Colab에 설치된 PyTorch를 사용합니다.
""")
code('''
    import importlib
    import io
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import tarfile
    import tempfile
    import urllib.request

    REPO_URL = "https://github.com/HisameOgasahara/tmp_full_Training.git"
    REPO_ROOT = Path(tempfile.mkdtemp(prefix="slippery_maze_")) / "repository"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "main", REPO_URL, str(REPO_ROOT)], check=True)
    REVISION = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()

    uv_executable = shutil.which("uv")
    if uv_executable is None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("로컬에서는 uv를 설치한 뒤 이 셀을 실행하세요. Colab에서는 자동 준비됩니다.")
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
    for name in list(sys.modules):
        if name == "maze_training" or name.startswith("maze_training."):
            del sys.modules[name]
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    importlib.invalidate_caches()
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except ImportError:
        pass
    print("실제 실행 커밋:", REVISION)
    print("코드 위치:", REPO_ROOT)
    print("PyTorch:", torch.__version__)
    print("장치:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (t4 프로필은 GPU 권장)")
''')
markdown("""
    ### 설정

    전체 학습은 `PROFILE = "t4"`, 짧은 셀 연결 확인은 `PROFILE = "smoke"`로 실행하세요.
    중단한 단계를 이어가려면 같은 설정과 출력 폴더에서 `RESUME = True`를 사용합니다.
    새 단계로 넘어갈 때는 `False`로 설정하세요. 같은 단계의 새 학습은 해당 체크포인트와 로그를 덮어씁니다.
""")
code('''
    PROFILE = "t4"  # "smoke"로 바꾸면 연결 확인용 짧은 실행
    RESUME = False
    CONFIG_PATH = str(REPO_ROOT / "configs" / f"{PROFILE}.json")
    OUTPUT_ROOT = Path("/content/maze_runs") if Path("/content").exists() else REPO_ROOT / "runs"
    OUTPUT_DIR = OUTPUT_ROOT / PROFILE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from maze_training.training import run_stage
    from maze_training.notebook import report_world, report_navigation, show_training_curve, launch_world, launch_navigation
    from maze_training.runtime import read_config, seed_runtime, create_model, choose_device, save_checkpoint

    config = read_config(CONFIG_PATH)
    model_for_size, _ = create_model(config, choose_device())
    print("파라미터:", f"{sum(p.numel() for p in model_for_size.parameters()):,}")
    print("설정:", CONFIG_PATH)
    print("체크포인트:", OUTPUT_DIR)
    del model_for_size
''')
markdown("""
    ### 선택: Google Drive에 저장

    세션 종료 후 모델을 다시 사용하려면 학습 전에 `USE_DRIVE = True`로 설정하고 실행하세요.
    Drive를 연결하면 `/content/drive/MyDrive/maze_training/프로필명`에 저장합니다.
""")
code('''
    USE_DRIVE = False
    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        OUTPUT_DIR = Path("/content/drive/MyDrive/maze_training") / PROFILE
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("저장 위치:", OUTPUT_DIR)
''')
markdown("""
    ## 1. 사전학습 — 행동의 결과를 예측

    지도·현재 위치·행동으로 다음 위치의 확률 분포를 예측합니다.
    무작위 이동의 전이 기록으로 벽 충돌과 미끄러짐의 규칙을 학습합니다.
""")
code('''
    PRETRAIN_PATH = run_stage("pretrain", CONFIG_PATH, OUTPUT_DIR, resume=RESUME)
    show_training_curve(OUTPUT_DIR, "pretrain")
''')
markdown("""
    ### 1단계 평가와 직접 실행

    평가 셀을 실행한 뒤 화면에서 행동을 선택하고 한 걸음씩 이동하세요.
    모델의 예측 분포와 환경의 정확한 전이확률, 실제 다음 위치를 비교합니다.
    TV distance와 KL은 낮을수록, 실제 가능한 위치에 부여한 확률은 높을수록 좋습니다.
    확인을 마치면 2단계 학습 셀을 실행하세요.
""")
code('''
    PRETRAIN_PATH = str(OUTPUT_DIR / "pretrain.pt")
    world_report = report_world(PRETRAIN_PATH)
    world_demo = launch_world(PRETRAIN_PATH)
''')
markdown("""
    ## 2. SFT — 목표로 이동하는 행동을 모방

    사전학습 체크포인트를 불러와 BFS가 선택한 다음 행동을 학습합니다.
    BFS는 벽·구멍을 피하는 최단 경로를 구하고 미끄러짐은 끈 조건으로 계획합니다.
    목표에 도달할 수 있는 여러 현재 위치에서 행동을 학습합니다.
""")
code('''
    SFT_PATH = run_stage("sft", CONFIG_PATH, OUTPUT_DIR, resume=RESUME)
    show_training_curve(OUTPUT_DIR, "sft")
''')
markdown("""
    ### 2단계 평가와 직접 실행

    평가 셀을 실행해 미끄러짐을 끈 조건과 켠 조건의 도착률·추락률·평균 보상을 비교하세요.
    실행 화면에서 한 걸음씩 이동하거나 자동 실행을 사용합니다.
    ‘미끄러짐 켜기’를 바꾼 뒤 ‘지도 초기화’를 눌러 같은 지도에서 다시 실행하세요.
""")
code('''
    SFT_PATH = str(OUTPUT_DIR / "sft.pt")
    sft_dry_report = report_navigation({"SFT": SFT_PATH}, CONFIG_PATH, slippery=False)
    sft_report = report_navigation({"SFT": SFT_PATH}, CONFIG_PATH, slippery=True)
    sft_demo = launch_navigation({"SFT": SFT_PATH}, CONFIG_PATH)
''')
markdown("""
    ## 3. GRPO — 실제 성공과 위험을 고려

    SFT 모델에서 시작해 지도별 후보 에피소드를 실행합니다.
    도착·추락·이동·충돌 보상의 합을 그룹 안에서 비교해 행동을 학습합니다.
    `useful_group_fraction`은 보상 차이가 있어 학습에 사용한 그룹의 비율입니다.
""")
code('''
    GRPO_PATH = run_stage("grpo", CONFIG_PATH, OUTPUT_DIR, resume=RESUME)
    show_training_curve(OUTPUT_DIR, "grpo")
''')
markdown("""
    ### 3단계 평가와 나란히 실행

    평가 셀을 실행해 SFT와 GRPO의 도착률·추락률·평균 보상을 비교하세요.
    화면에서는 두 모델이 같은 지도와 환경 난수에서 argmax 행동으로 이동합니다.
    지도와 seed를 선택하고 ‘지도 초기화’를 누른 뒤 한 걸음씩 또는 자동으로 실행하세요.
""")
code('''
    SFT_PATH, GRPO_PATH = str(OUTPUT_DIR / "sft.pt"), str(OUTPUT_DIR / "grpo.pt")
    comparison = report_navigation({"SFT": SFT_PATH, "GRPO": GRPO_PATH}, CONFIG_PATH)
    comparison_demo = launch_navigation({"SFT": SFT_PATH, "GRPO": GRPO_PATH}, CONFIG_PATH)
''')
markdown("""
    ## 4. 최종 테스트

    세 단계의 학습을 마친 뒤 실행하세요. 검증과 다른 지도 seed를 사용합니다.
    BFS는 매 이동 후 실제 위치에서 다시 계획합니다.
    `optimal`은 실제 미끄러짐·보상·남은 행동 수를 고려한 유한 시간 최적 정책입니다.
    SFT·GRPO·BFS·optimal의 도착률·추락률·평균 보상을 비교하세요.
""")
code('''
    final_test = report_navigation({"SFT": str(OUTPUT_DIR / "sft.pt"), "GRPO": str(OUTPUT_DIR / "grpo.pt")}, CONFIG_PATH, split="test")
''')
markdown("""
    ## 5. 선택적 단계 생략 비교

    비교할 실험의 스위치를 `True`로 바꾸고 해당 셀을 실행하세요.
    사전학습 생략은 무작위 초기화 → SFT → GRPO,
    SFT 생략은 사전학습 → GRPO 순서입니다.
    생략 실험은 별도 하위 폴더에 저장합니다. 같은 후속 단계 예산에서 검증 지표와 학습 시간을 비교하세요.
""")
code('''
    RUN_WITHOUT_PRETRAIN = False
    if RUN_WITHOUT_PRETRAIN:
        ablation_dir = OUTPUT_DIR / "without_pretrain"
        seed_runtime(config["seed"])
        random_model, _ = create_model(config, choose_device())
        random_path = ablation_dir / "random_initialization.pt"
        save_checkpoint(random_path, random_model, config, "random_initialization", 0)
        del random_model
        run_stage("sft", CONFIG_PATH, ablation_dir, initialize_from=random_path)
        run_stage("grpo", CONFIG_PATH, ablation_dir)
        report_navigation({"전체 흐름": str(OUTPUT_DIR / "grpo.pt"), "사전학습 생략": str(ablation_dir / "grpo.pt")}, CONFIG_PATH)
''')
code('''
    RUN_WITHOUT_SFT = False
    if RUN_WITHOUT_SFT:
        ablation_dir = OUTPUT_DIR / "without_sft"
        run_stage("grpo", CONFIG_PATH, ablation_dir, initialize_from=OUTPUT_DIR / "pretrain.pt")
        report_navigation({"전체 흐름": str(OUTPUT_DIR / "grpo.pt"), "SFT 생략": str(ablation_dir / "grpo.pt")}, CONFIG_PATH)
''')
markdown("""
    ## 6. 체크포인트 다운로드

    `DOWNLOAD_RESULTS = True`로 바꾸고 실행해 모델·설정·학습 로그·평가 결과를 ZIP으로 내려받으세요.
    다음 런타임에서 ZIP을 출력 폴더에 복원한 뒤 평가 셀을 실행하면 저장된 모델을 다시 사용할 수 있습니다.
""")
code('''
    DOWNLOAD_RESULTS = False
    if DOWNLOAD_RESULTS:
        from google.colab import files
        archive_path = shutil.make_archive(str(OUTPUT_DIR.parent / f"maze_results_{PROFILE}"), "zip", OUTPUT_DIR)
        files.download(archive_path)
''')

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "name": "Maze_Training.ipynb"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
destination = Path(__file__).resolve().parents[1] / "notebooks" / "Maze_Training.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(destination)
