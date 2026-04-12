"""
Experiment Tracker — PathOptLearn Evaluation Pipeline
======================================================
Lightweight SQLite-based experiment tracker.  Records runs, metrics, and
arbitrary JSON artefacts, then supports comparison and export.

No external dependencies beyond stdlib + pandas (already in requirements.txt).

Schema
------
  runs       : run_id, name, config_json, status, started_at, completed_at
  metrics    : id, run_id, metric_name, value, step, logged_at
  artifacts  : id, run_id, artifact_name, artifact_json, logged_at

Usage
-----
    from pipeline.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker("eval_output/results.db")
    run_id  = tracker.create_run("sim_batch_1", {"topic": "ML"})

    tracker.log_metrics(run_id, {"pass_rate": 0.82, "avg_score": 78.5})
    tracker.log_artifact(run_id, "trajectories", {"beginner_0": [50, 60, 72]})
    tracker.complete_run(run_id)

    df = tracker.compare_runs()
    print(df.to_string())

    tracker.export_csv("eval_output/comparison.csv")
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class ExperimentTracker:
    """
    SQLite-backed experiment tracker for the PathOptLearn evaluation pipeline.

    Each *run* represents one evaluation experiment (e.g., a batch of
    student simulations, a knowledge-tracing benchmark, or an A/B test).
    Runs store numeric metrics and JSON artefacts and can be compared
    side-by-side in a DataFrame.
    """

    def __init__(self, db_path: str = "eval_results.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Connection ─────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id       TEXT PRIMARY KEY,
                    name         TEXT NOT NULL,
                    config_json  TEXT,
                    status       TEXT DEFAULT 'running',
                    started_at   TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
                    metric_name  TEXT NOT NULL,
                    value        REAL,
                    step         INTEGER,
                    logged_at    TEXT
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id         TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
                    artifact_name  TEXT NOT NULL,
                    artifact_json  TEXT,
                    logged_at      TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
            """)

    # ── CRUD — runs ────────────────────────────────────────────────────────────

    def create_run(self, name: str, config: dict | None = None) -> str:
        """
        Create and persist a new experiment run.

        Returns
        -------
        run_id : str  (8-character hex prefix of a UUID4)
        """
        run_id = str(uuid.uuid4())[:8]
        now    = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, name, config_json, status, started_at) "
                "VALUES (?, ?, ?, 'running', ?)",
                (run_id, name, json.dumps(config or {}), now),
            )
        print(f"[Tracker] Run '{name}' created — id={run_id}")
        return run_id

    def complete_run(self, run_id: str, status: str = "completed"):
        """Mark a run as completed, failed, or cancelled."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status=?, completed_at=? WHERE run_id=?",
                (status, now, run_id),
            )

    def delete_run(self, run_id: str):
        """Delete a run and all its metrics and artefacts."""
        with self._conn() as conn:
            conn.execute("DELETE FROM metrics   WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM artifacts WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM runs      WHERE run_id=?", (run_id,))

    def list_runs(self) -> list[dict]:
        """Return a summary list of all runs (no metric detail)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT run_id, name, status, started_at, completed_at "
                "FROM runs ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        """
        Return the full record for a run, including all metrics and artefacts.
        Returns None if the run_id is not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if not row:
                return None

            run = dict(row)
            run["config"] = json.loads(run.pop("config_json", "{}") or "{}")

            metrics = conn.execute(
                "SELECT metric_name, value, step, logged_at "
                "FROM metrics WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
            run["metrics"] = [dict(m) for m in metrics]

            artifacts = conn.execute(
                "SELECT artifact_name, artifact_json FROM artifacts WHERE run_id=?",
                (run_id,),
            ).fetchall()
            run["artifacts"] = {
                a["artifact_name"]: json.loads(a["artifact_json"])
                for a in artifacts
            }

        return run

    # ── Logging metrics ────────────────────────────────────────────────────────

    def log_metric(self, run_id: str, metric_name: str,
                   value: float, step: int | None = None):
        """Log a single scalar metric value."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO metrics (run_id, metric_name, value, step, logged_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, metric_name, float(value), step, now),
            )

    def log_metrics(self, run_id: str, metrics: dict[str, Any],
                    step: int | None = None, prefix: str = ""):
        """
        Log multiple scalar metrics at once.

        Non-numeric values are silently skipped.  Use ``prefix`` to namespace
        metrics (e.g. ``prefix="sim."`` → "sim.pass_rate").
        """
        now  = datetime.now(timezone.utc).isoformat()
        rows = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                rows.append((run_id, f"{prefix}{k}", float(v), step, now))
        if rows:
            with self._conn() as conn:
                conn.executemany(
                    "INSERT INTO metrics "
                    "(run_id, metric_name, value, step, logged_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )

    def log_nested_metrics(self, run_id: str, nested: dict,
                           step: int | None = None, prefix: str = ""):
        """
        Recursively flatten a nested metrics dict and log all scalars.

        Example: {"cohort_stats": {"mean_pass_rate": 0.8}} is logged as
                 "cohort_stats.mean_pass_rate = 0.8".
        """
        flat: dict[str, float] = {}
        self._flatten(nested, prefix, flat)
        self.log_metrics(run_id, flat, step=step)

    @staticmethod
    def _flatten(d: Any, prefix: str, out: dict):
        """Recursively flatten a nested dict into dot-separated scalar keys."""
        if isinstance(d, dict):
            for k, v in d.items():
                ExperimentTracker._flatten(v, f"{prefix}{k}.", out)
        elif isinstance(d, (int, float)) and not isinstance(d, bool):
            out[prefix.rstrip(".")] = float(d)

    # ── Logging artefacts ──────────────────────────────────────────────────────

    def log_artifact(self, run_id: str, name: str, data: Any):
        """
        Store an arbitrary JSON-serialisable artefact.

        Useful for saving full simulation summaries, trajectory arrays,
        benchmark reports, etc.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO artifacts "
                "(run_id, artifact_name, artifact_json, logged_at) "
                "VALUES (?, ?, ?, ?)",
                (run_id, name, json.dumps(data, default=str), now),
            )

    # ── Comparison ─────────────────────────────────────────────────────────────

    def compare_runs(self, run_ids: list[str] | None = None) -> pd.DataFrame:
        """
        Return a DataFrame where each row is a run and each column is a metric
        (the *last* logged value per metric name per run).

        ``run_ids=None`` includes every run in the database.
        """
        with self._conn() as conn:
            if run_ids:
                ph   = ",".join("?" * len(run_ids))
                runs = conn.execute(
                    f"SELECT run_id, name, status, started_at "
                    f"FROM runs WHERE run_id IN ({ph}) ORDER BY started_at",
                    run_ids,
                ).fetchall()
            else:
                runs = conn.execute(
                    "SELECT run_id, name, status, started_at "
                    "FROM runs ORDER BY started_at"
                ).fetchall()

            records = []
            for run in runs:
                record = dict(run)
                # Last-value pivot per metric
                metrics = conn.execute(
                    "SELECT metric_name, value FROM metrics "
                    "WHERE run_id=? ORDER BY id",
                    (run["run_id"],),
                ).fetchall()
                for m in metrics:
                    record[m["metric_name"]] = m["value"]
                records.append(record)

        return pd.DataFrame(records)

    # ── Export ─────────────────────────────────────────────────────────────────

    def export_csv(self, path: str, run_ids: list[str] | None = None):
        """Export the comparison table to a CSV file."""
        df = self.compare_runs(run_ids)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[Tracker] Exported {len(df)} runs → {path}")

    def export_json(self, path: str, run_ids: list[str] | None = None):
        """Export full run details (with metrics and artefacts) to JSON."""
        ids  = run_ids or [r["run_id"] for r in self.list_runs()]
        data = [self.get_run(rid) for rid in ids]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[Tracker] Exported {len(data)} runs → {path}")

    def print_summary(self, run_ids: list[str] | None = None):
        """Print a concise comparison table to stdout."""
        df = self.compare_runs(run_ids)
        if df.empty:
            print("[Tracker] No runs found.")
            return
        # Keep only run_id, name, status and any numeric columns
        num_cols  = df.select_dtypes(include="number").columns.tolist()
        show_cols = ["run_id", "name", "status"] + num_cols[:10]
        show_cols = [c for c in show_cols if c in df.columns]
        print("\n" + "═" * 80)
        print("  PathOptLearn — Experiment Comparison")
        print("═" * 80)
        print(df[show_cols].to_string(index=False))
        print("═" * 80 + "\n")
