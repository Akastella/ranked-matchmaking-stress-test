from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matchmaking_sim.experiments import run_diagnostic_timeseries, run_profile


if __name__ == "__main__":
    df = run_profile("small")
    run_diagnostic_timeseries()
    print(f"small profile rows={len(df)}")
