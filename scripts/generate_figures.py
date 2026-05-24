from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matchmaking_sim.experiments import run_diagnostic_timeseries
from matchmaking_sim.plotting import generate_all_figures


if __name__ == "__main__":
    run_diagnostic_timeseries()
    names = generate_all_figures()
    print("\n".join(names))
