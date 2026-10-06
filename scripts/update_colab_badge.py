"""Point the Colab badge at an immutable notebook revision after each push."""

import argparse
from pathlib import Path
import re

REPOSITORY = "HisameOgasahara/tmp_full_Training"
NOTEBOOK = "notebooks/Maze_Training.ipynb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("revision")
    arguments = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", arguments.revision):
        raise ValueError("전체 Git 커밋 SHA가 필요합니다.")
    readme = Path(__file__).resolve().parents[1] / "README.md"
    source = readme.read_text(encoding="utf-8")
    url = f"https://colab.research.google.com/github/{REPOSITORY}/blob/{arguments.revision}/{NOTEBOOK}"
    badge = f"<!-- COLAB_BADGE_START -->\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})\n<!-- COLAB_BADGE_END -->"
    updated, count = re.subn(r"<!-- COLAB_BADGE_START -->.*?<!-- COLAB_BADGE_END -->", badge, source, flags=re.DOTALL)
    if count != 1:
        raise ValueError("README 배지 위치를 찾지 못했습니다.")
    readme.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
