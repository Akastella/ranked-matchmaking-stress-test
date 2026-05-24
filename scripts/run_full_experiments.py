from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matchmaking_sim.experiments import run_full_core_profile, run_full_proxy_weight_sensitivity


if __name__ == "__main__":
    df = run_full_core_profile()
    proxy = run_full_proxy_weight_sensitivity()
    print(f"full core profile rows={len(df)}")
    print(f"full proxy sensitivity rows={len(proxy)}")
