"""Trial ledger: every experiment is logged, so the number of trials cannot hide.

The ledger is a SQLite file. Each row records what was tried (signal and
parameters), on what data (window), how (seed, git hash), and the summary
moments needed for deflation. The deflated Sharpe and the tearsheet read the
trial count from here rather than from the user.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .stats import moments

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    study TEXT NOT NULL,
    signal TEXT NOT NULL,
    params TEXT NOT NULL,
    data_window TEXT,
    seed INTEGER,
    git_hash TEXT,
    n_obs INTEGER,
    sr REAL,
    skew REAL,
    kurt REAL,
    gross_sr REAL,
    extra TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_trials_study ON trials(study);
"""


def git_hash(cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, check=True, timeout=5)
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "nogit"


@dataclass(frozen=True)
class StudySummary:
    study: str
    n_trials: int
    sr_mean: float
    sr_variance: float


class Ledger:
    def __init__(self, path: str | Path = "ledger.sqlite") -> None:
        self.path = str(path)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def log(self, study: str, signal: str, params: dict[str, Any], returns: ArrayLike, *,
            gross_returns: ArrayLike | None = None, data_window: str = "", seed: int | None = None,
            extra: dict[str, Any] | None = None) -> int:
        """Log one trial from its (net) per-period returns. Returns the trial id."""
        m = moments(returns)
        gsr = moments(gross_returns).sr if gross_returns is not None else None
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO trials (study, signal, params, data_window, seed, git_hash, n_obs,"
                " sr, skew, kurt, gross_sr, extra, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (study, signal, json.dumps(params, sort_keys=True, default=str), data_window,
                 seed, git_hash(), m.n, m.sr, m.skew, m.kurt, gsr,
                 json.dumps(extra or {}, default=str), datetime.now(UTC).isoformat()))
            return int(cur.lastrowid or 0)

    def trials(self, study: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM trials WHERE study = ? ORDER BY id", (study,)).fetchall()
        return [dict(r) for r in rows]

    def summary(self, study: str) -> StudySummary:
        rows = self.trials(study)
        if not rows:
            raise LookupError(f"no trials logged for study {study!r}")
        srs = np.array([r["sr"] for r in rows], dtype=float)
        var = float(srs.var(ddof=1)) if srs.size > 1 else 0.0
        return StudySummary(study, len(rows), float(srs.mean()), var)

    def n_trials(self, study: str) -> int:
        with self._conn() as c:
            (n,) = c.execute("SELECT COUNT(*) FROM trials WHERE study = ?", (study,)).fetchone()
        return int(n)

    def clear(self, study: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM trials WHERE study = ?", (study,))


def optuna_callback(ledger: Ledger, study_name: str, signal: str) -> Any:
    """Optuna callback that logs every finished trial.

    The objective must store its per-period net returns with
    ``trial.set_user_attr("returns", list_of_floats)``.
    """

    def cb(study: Any, trial: Any) -> None:
        r = trial.user_attrs.get("returns")
        if r is None:
            raise ValueError("objective must set_user_attr('returns', ...) for the ledger")
        ledger.log(study_name, signal, dict(trial.params), r, seed=trial.number,
                   extra={"optuna_value": trial.value, "state": str(trial.state)})

    return cb
