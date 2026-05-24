from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matchmaking_sim.experiments import run_medium_profile, run_medium_sensitivity, run_rank_dynamics_trace


if __name__ == "__main__":
    main = run_medium_profile()
    sens = run_medium_sensitivity()
    trace = run_rank_dynamics_trace()
    print(f"medium main rows={len(main)}")
    print(f"medium sensitivity rows={len(sens)}")
    print(f"rank trace rows={len(trace)}")
