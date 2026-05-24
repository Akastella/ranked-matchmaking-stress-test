from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matchmaking_sim.experiments import run_sensitivity


if __name__ == "__main__":
    df = run_sensitivity()
    print(f"sensitivity rows={len(df)}")
