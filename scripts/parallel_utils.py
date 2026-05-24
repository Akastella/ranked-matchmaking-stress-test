from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def configure_threads() -> None:
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(key, "1")


def default_n_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def write_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv.gz"), index=False, compression="gzip")


def read_result(path: Path) -> dict | None:
    parquet = path.with_suffix(".parquet")
    csv = path.with_suffix(".csv.gz")
    if parquet.exists():
        return pd.read_parquet(parquet).iloc[0].to_dict()
    if csv.exists():
        return pd.read_csv(csv).iloc[0].to_dict()
    return None


def result_files(folder: Path) -> list[Path]:
    return sorted(list(folder.glob("*.parquet")) + list(folder.glob("*.csv.gz")))


def read_many(folder: Path) -> pd.DataFrame:
    frames = []
    for file in result_files(folder):
        if file.suffix == ".parquet":
            frames.append(pd.read_parquet(file))
        else:
            frames.append(pd.read_csv(file))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_parallel_tasks(
    tasks: list[dict],
    worker: Callable[[dict], dict],
    task_dir: Path,
    log_path: Path,
    n_jobs: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    configure_threads()
    n_jobs = n_jobs or default_n_jobs()
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    completed = 0
    failed = 0
    skipped = 0
    rows = []
    messages = [f"tasks={len(tasks)}", f"n_jobs={n_jobs}"]

    runnable = []
    for task in tasks:
        base = task_dir / str(task["task_id"])
        existing = read_result(base)
        if existing is not None:
            rows.append(existing)
            skipped += 1
        else:
            runnable.append(task)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(worker, task): task for task in runnable}
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
                write_result(task_dir / str(task["task_id"]), row)
                rows.append(row)
                completed += 1
                messages.append(f"ok {completed + failed}/{len(runnable)} {task['task_id']}")
            except Exception as exc:
                failed += 1
                messages.append(f"failed {completed + failed}/{len(runnable)} {task['task_id']} reason={exc!r}")
            if (completed + failed) % 5 == 0 or completed + failed == len(runnable):
                log_path.write_text("\n".join(messages) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - start
    df = pd.DataFrame(rows)
    stats = {
        "tasks_total": len(tasks),
        "tasks_completed": completed,
        "tasks_skipped": skipped,
        "tasks_failed": failed,
        "n_jobs": n_jobs,
        "wall_clock_seconds": elapsed,
        "mean_completed_task_seconds": elapsed / max(1, completed),
    }
    messages.extend([f"{k}={v}" for k, v in stats.items()])
    log_path.write_text("\n".join(messages) + "\n", encoding="utf-8")
    return df, stats
