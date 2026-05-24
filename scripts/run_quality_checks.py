from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


if __name__ == "__main__":
    required = [
        "config/experiment_config.yaml",
        "config/simulation_config.yaml",
        "config/performance_config.yaml",
        "config/large_robustness_config.yaml",
        "config/stress_grid_config.yaml",
        "results/tables/main_results_full_profile_reframed.csv",
        "results/tables/risk_transfer_experiment.csv",
        "results/tables/large_robustness_results.csv",
        "results/tables/stress_grid_results.csv",
        "results/tables/ranking_stability.csv",
        "results/figures/final/figure_01_stress_test_framework.pdf",
        "results/figures/final/figure_02_rank_mmr_hysteresis_trace.pdf",
        "results/figures/final/fig_risk_transfer_by_smurf_ratio.pdf",
        "results/figures/final/fig_parameter_sensitivity_heatmap.pdf",
        "validation/annotation_protocol.md",
        "validation/annotation_form.csv",
        "validation/sample_synthetic_match_histories.csv",
        "manuscript/main.tex",
        "manuscript/references_final.bib",
    ]
    missing = [p for p in required if not exists(p)]
    test = subprocess.run([sys.executable, "-m", "pytest", "src/tests"], cwd=ROOT)
    status = "PASS" if not missing and test.returncode == 0 else "FAIL"
    report = ROOT / "results/diagnostics/quality_check.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Quality Check\n\n"
        f"Status: {status}\n\n"
        f"Missing: {missing}\n\n"
        f"Pytest return code: {test.returncode}\n",
        encoding="utf-8",
    )
    print(report.read_text(encoding="utf-8"))
    raise SystemExit(0 if status == "PASS" else 1)
