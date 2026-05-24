from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import psutil

from parallel_utils import default_n_jobs, run_parallel_tasks
from matchmaking_sim.experiments import simulate_condition


def _task(task: dict) -> dict:
    start = time.perf_counter()
    summary, _ = simulate_condition(
        int(task["seed"]),
        "benchmark",
        "dynamic_k",
        task["policy"],
        "benchmark",
        n_players=360,
        days=8,
        matches_per_day=80,
    )
    summary.update(task)
    summary["task_seconds"] = time.perf_counter() - start
    return summary


def _tasks(n: int) -> list[dict]:
    policies = ["nearest", "role_aware", "multi_objective", "multi_objective_with_smurf_exposure_penalty"]
    return [{"task_id": f"bench_{i}", "seed": 900 + i, "policy": policies[i % len(policies)]} for i in range(n)]


if __name__ == "__main__":
    out_log = ROOT / "results/logs/performance_benchmark_report.md"
    resource = ROOT / "results/logs/system_resource_report.txt"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text(
        "\n".join(
            [
                f"platform={platform.platform()}",
                f"python={platform.python_version()}",
                f"logical_cores={os.cpu_count()}",
                f"physical_cores={psutil.cpu_count(logical=False)}",
                f"available_ram_gb={psutil.virtual_memory().available / (1024**3):.2f}",
                f"total_ram_gb={psutil.virtual_memory().total / (1024**3):.2f}",
                "parallel_backend=concurrent.futures.ProcessPoolExecutor",
                "blas_threads=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    serial_start = time.perf_counter()
    serial_rows = [_task(t) for t in _tasks(10)]
    serial_elapsed = time.perf_counter() - serial_start
    cpu_before = psutil.cpu_percent(interval=0.2)
    parallel_df, stats = run_parallel_tasks(
        _tasks(20),
        _task,
        ROOT / "results/experiment_runs/performance_benchmark_parallel",
        ROOT / "results/logs/performance_benchmark_parallel.log",
        min(default_n_jobs(), 8),
    )
    cpu_after = psutil.cpu_percent(interval=0.2)
    speedup = (serial_elapsed / 10.0) / (stats["wall_clock_seconds"] / max(1, stats["tasks_completed"]))
    out_log.write_text(
        "\n".join(
            [
                "# Performance Benchmark Report",
                "",
                f"Serial sample tasks: 10",
                f"Serial elapsed seconds: {serial_elapsed:.3f}",
                f"Parallel sample tasks: 20",
                f"Parallel n_jobs: {stats['n_jobs']}",
                f"Parallel elapsed seconds: {stats['wall_clock_seconds']:.3f}",
                f"Estimated per-task speedup vs serial sample: {speedup:.2f}x",
                f"Tasks completed: {stats['tasks_completed']}",
                f"Tasks failed: {stats['tasks_failed']}",
                f"CPU utilization snapshot before/after: {cpu_before:.1f}% / {cpu_after:.1f}%",
                f"Mean serial task seconds: {serial_elapsed/10.0:.3f}",
                f"Mean parallel completed task seconds: {stats['mean_completed_task_seconds']:.3f}",
                "",
                "Bottleneck notes: simulation tasks are Python-heavy and independent; process-level parallelism is used across seed/scenario/policy cells while BLAS threads are capped to avoid oversubscription.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_log)
