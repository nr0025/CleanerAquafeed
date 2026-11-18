"""Exact ingredient-level attribution of the screening results (Eq. 26).

Cost and all six impacts are linear in the ingredient inclusions, so the change of each ratio relative to the
benchmark splits exactly into one additive contribution per ingredient:

    R_k,f - 1 = sum_i  e_i,k (x_i,f - x_i,b) / I_k,b        C_r,f - 1 = sum_i  p_i (x_i,f - x_i,b) / C_b

No model is fitted, so no validation is needed and collinearity between ingredients does not affect the result.
The eligibility rules are thresholds on these ratios; a failed rule is attributed to the indicator that violates it
and to the ingredient with the largest positive contribution to that indicator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import IMPACTS, IngredientDB
from .evaluation import Rules, evaluate

TARGETS = ["cost", *IMPACTS]          # order of the last axis of contributions()


def contributions(db: IngredientDB, X: np.ndarray, x_bench: np.ndarray, price: np.ndarray | None = None,
                  impacts: np.ndarray | None = None) -> np.ndarray:
    """Array (formulations x ingredients x [cost, six impacts]) of contributions to C_r - 1 and R_k - 1 (Eq. 26).
    Summed over ingredients they reproduce the ratios of evaluate() exactly."""
    X = np.atleast_2d(np.asarray(X, float))
    p = db.price if price is None else price
    E = db.impacts if impacts is None else impacts
    D = X - x_bench
    c_cost = D * p / float(x_bench @ p)
    c_imp = D[:, :, None] * E[None, :, :] / (x_bench @ E)[None, None, :]
    return np.concatenate([c_cost[:, :, None], c_imp], axis=2)


def check_exact(db: IngredientDB, X: np.ndarray, x_bench: np.ndarray, rules: Rules, tol: float = 1e-9) -> float:
    """Largest absolute difference between the summed contributions and the ratios of evaluate()."""
    ev = evaluate(db, X, x_bench, rules)
    c = contributions(db, X, x_bench).sum(axis=1)
    ref = np.column_stack([ev["cost_ratio"]] + [ev[f"{k}_ratio"] for k in IMPACTS]) - 1.0
    err = float(np.abs(c - ref).max())
    err = max(err, float(np.abs(c[:, 1:].mean(axis=1) - (ev["mean_env_ratio"].to_numpy() - 1)).max()))
    assert err < tol, err
    return err


def diet_table(db: IngredientDB, X: np.ndarray, ids: list[str], x_bench: np.ndarray, ranks: list | None = None,
               mean_label: str | None = None, mean_rows: slice | None = None) -> pd.DataFrame:
    """Contributions (percentage points) of every ingredient whose inclusion differs from the benchmark, one block
    per diet, each closed by a 'Total' row equal to the change relative to the benchmark (%)."""
    C = 100 * contributions(db, X, x_bench)
    D = np.atleast_2d(X) - x_bench
    blocks = [(ids[j], None if ranks is None else ranks[j], D[j], C[j]) for j in range(len(ids))]
    if mean_label is not None:
        blocks.append((mean_label, None, D[mean_rows].mean(axis=0), C[mean_rows].mean(axis=0)))
    rows = []
    for diet, rank, d, c in blocks:
        keep = np.abs(d) > 1e-6
        for i in np.where(keep)[0]:
            rows.append(_row(diet, rank, db.names[i], d[i], c[i]))
        rows.append(_row(diet, rank, "Total", d.sum(), c.sum(axis=0)))
    return pd.DataFrame(rows)


def _row(diet, rank, ingredient, d, c) -> dict:
    r = {"diet": diet, "rank": rank, "ingredient": ingredient, "delta_kg_t": d}
    r.update({f"{t}_pp": c[j] for j, t in enumerate(TARGETS)})
    r["mean_env_pp"] = c[1:].mean()
    return r


def rule_failures(db: IngredientDB, X: np.ndarray, x_bench: np.ndarray, rules: Rules) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Why candidates fail R1-R4. Returns (counts per rule, rule x violating indicator x largest contributor)."""
    ev = evaluate(db, X, x_bench, rules)
    C = contributions(db, X, x_bench)
    R = np.column_stack([ev[f"{k}_ratio"] for k in IMPACTS])
    fail = ~ev["robust"].to_numpy()
    desc = {"R1": f"cost ratio > {1 + rules.cost_tol:g}", "R2": "GWP not below the benchmark",
            "R3": f"fewer than {rules.min_reduced} of 6 indicators reduced",
            "R4": f"at least one indicator > {1 + rules.burden_tol:g} x benchmark"}
    counts = []
    for r in ("R1", "R2", "R3", "R4"):
        f = ~ev[r].to_numpy()
        counts.append({"rule": r, "condition_failed": desc[r], "n_failing": int(f.sum()),
                       "n_failing_only_this_rule": int((f & (ev[["R1", "R2", "R3", "R4"]].to_numpy().sum(axis=1) == 3)).sum())})
    counts.append({"rule": "any", "condition_failed": "not a robust cleaner candidate", "n_failing": int(fail.sum()),
                   "n_failing_only_this_rule": np.nan})
    rows = []

    def top(f, j):
        return db.names[int(np.argmax(C[f, :, j]))]
    for f in np.where(~ev["R1"].to_numpy())[0]:
        rows.append(("R1", "cost", top(f, 0)))
    for f in np.where(~ev["R2"].to_numpy())[0]:
        rows.append(("R2", "gwp", top(f, 1)))
    for f in np.where(~ev["R3"].to_numpy())[0]:
        for k in np.where(R[f] >= 1.0)[0]:
            rows.append(("R3", IMPACTS[k], top(f, k + 1)))
    for f in np.where(~ev["R4"].to_numpy())[0]:
        for k in np.where(R[f] > 1.0 + rules.burden_tol + 1e-12)[0]:
            rows.append(("R4", IMPACTS[k], top(f, k + 1)))
    t = pd.DataFrame(rows, columns=["rule", "indicator", "largest_positive_contributor"])
    t = t.value_counts().rename("n_cases").reset_index().sort_values(["rule", "indicator", "n_cases"], ascending=[True, True, False])
    return pd.DataFrame(counts), t


def candidate_table(db: IngredientDB, X: np.ndarray, ids: list[str], x_bench: np.ndarray, rules: Rules) -> pd.DataFrame:
    """Per candidate: failed rules, indicators above the burden threshold and, for cost, each impact and the mean
    impact, the ingredients with the largest positive and negative contributions (percentage points)."""
    ev = evaluate(db, X, x_bench, rules)
    C = 100 * contributions(db, X, x_bench)
    C = np.concatenate([C, C[:, :, 1:].mean(axis=2, keepdims=True)], axis=2)
    R = np.column_stack([ev[f"{k}_ratio"] for k in IMPACTS])
    names = np.array(db.names)
    out = pd.DataFrame({"formula_id": ids, "robust": ev["robust"].to_numpy(), "cleaner_score": ev["cleaner_score"].to_numpy()})
    out["rules_failed"] = [" ".join(r for r in ("R1", "R2", "R3", "R4") if not ev[r].iloc[j]) for j in range(len(ids))]
    out["indicators_above_burden_threshold"] = [" ".join(IMPACTS[k] for k in np.where(R[j] > 1 + rules.burden_tol + 1e-12)[0])
                                                for j in range(len(ids))]
    for j, t in enumerate([*TARGETS, "mean_env"]):
        out[f"{t}_change_pct"] = C[:, :, j].sum(axis=1)
        imax, imin = C[:, :, j].argmax(axis=1), C[:, :, j].argmin(axis=1)
        out[f"{t}_largest_increase"] = names[imax]
        out[f"{t}_largest_increase_pp"] = C[np.arange(len(ids)), imax, j]
        out[f"{t}_largest_decrease"] = names[imin]
        out[f"{t}_largest_decrease_pp"] = C[np.arange(len(ids)), imin, j]
    return out


def composition_by_class(db: IngredientDB, X: np.ndarray, x_bench: np.ndarray, robust: np.ndarray) -> pd.DataFrame:
    """Mean inclusion (kg/t) of qualifying and non-qualifying candidates and of the benchmark."""
    t = pd.DataFrame({"ingredient": db.names, "benchmark_kg_t": x_bench, "robust_mean_kg_t": X[robust].mean(axis=0),
                      "non_robust_mean_kg_t": X[~robust].mean(axis=0)})
    return t[(t.iloc[:, 1:] > 1e-6).any(axis=1)].reset_index(drop=True)
