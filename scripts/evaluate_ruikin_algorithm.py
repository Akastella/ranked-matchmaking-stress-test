from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train_ruikin_algorithm.py")], cwd=ROOT, check=True)
