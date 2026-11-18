"""End-to-end run of the screening framework; every table and figure of the paper is written to the output folder."""
from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from . import benchmarks as bm
from .data import FISHMEAL, IMPACTS, load_ingredients, plausibility_report
from .evaluation import Rules, composition, evaluate, evaluate_full, rank_robust
from .formulation import (TOTAL_KG, MixSettings, NutrientSpec, VertexSettings, candidate_model, generate_candidates,
                          reference_diets)
from .growth import GrowthSettings, cap_binding_weight, simulate_diets
from .uncertainty import lca_robust_probability, price_risk, threshold_sensitivity

RENDERED = ["Meat and bone", "Poultry proteins"]   # rendered animal proteins (blood meal not included)

STAGES = ["formulation", "uncertainty", "benchmarks", "growth", "attribution", "figures", "tables"]


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


class Run:
    def __init__(self, config: str | Path, out: str | Path, root: str | Path | None = None):
        self.cfg_path = Path(config)
        self.cfg = load_config(config)
        self.root = Path(root) if root else self.cfg_path.resolve().parent.parent
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        c = self.cfg
        self.db = load_ingredients(self.root / c["data"]["ingredients"], c["data"].get("overrides"))
        self.spec = NutrientSpec.from_dict(c["nutrients"])
        self.rules = Rules.from_dict(c["rules"])
        self.summary: dict = {}
        sp = self.out / "summary.json"
        if sp.exists():
            self.summary = json.loads(sp.read_text())

    # ---------------------------------------------------------------- helpers
    def save(self, df: pd.DataFrame, name: str, index: bool = False) -> None:
        df.to_csv(self.out / name, index=index)

    def load(self, name: str, **kw) -> pd.DataFrame:
        return pd.read_csv(self.out / name, **kw)

    def note(self, key: str, value) -> None:
        self.summary[key] = value
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=1, default=float))

    @property
    def x_bench(self) -> np.ndarray:
        ref = self.load("reference_diets_composition.csv", index_col=0)
        return ref.loc[self.cfg["reference"]["benchmark"], self.db.names].to_numpy(float)

    def model(self):
        c = self.cfg["candidates"]
        return candidate_model(self.db, self.spec, self.x_bench, c["fishmeal_min"], c["fishmeal_max"])

    # ---------------------------------------------------------------- stages
    def formulation(self) -> None:
        db, c = self.db, self.cfg
        self.save(plausibility_report(db), "ingredient_plausibility_check.csv")
        ref = reference_diets(db, self.spec, c["reference"]["fishmeal_levels"], c["reference"]["labels"])
        self.save(ref, "reference_diets_composition.csv", index=True)
        xb = ref.loc[c["reference"]["benchmark"]].to_numpy(float)
        ev = evaluate(db, ref.to_numpy(float), xb, self.rules)
        ev.insert(0, "diet", ref.index)
        comp = composition(db, ref.to_numpy(float), self.spec.premix_kg, self.spec.premix_dm)
        self.save(pd.concat([ev, comp], axis=1), "reference_diets_evaluation.csv")

        cc = c["candidates"]
        vs = VertexSettings(cc["n_random_objectives"], cc["eps_rel"], cc["dedup_decimals"], cc["vertex_seed"])
        ms = MixSettings(cc["n_combinations"], cc["k_min"], cc["k_max"], cc["mix_seed"])
        m = candidate_model(db, self.spec, xb, cc["fishmeal_min"], cc["fishmeal_max"])
        t0 = time.perf_counter()
        cand = generate_candidates(m, xb, vs, ms)
        t_gen = time.perf_counter() - t0
        self.save(cand, "candidates_composition.csv")
        t0 = time.perf_counter()
        evf = evaluate_full(db, cand, xb, self.rules, self.spec.premix_kg, self.spec.premix_dm)
        t_eval = time.perf_counter() - t0
        self.save(evf, "candidates_evaluation.csv")
        n_vert = int((cand["source"] == "lp_vertex").sum())
        self.note("n_objectives", 9 + cc["n_random_objectives"])
        self.note("n_vertices", n_vert)
        self.note("n_candidates", len(cand))
        self.note("candidate_generation_s", t_gen)
        self.note("evaluation_all_candidates_ms", 1e3 * t_eval)

    def uncertainty(self) -> None:
        db, c = self.db, self.cfg["uncertainty"]
        xb = self.x_bench
        cand = self.load("candidates_composition.csv")
        evf = self.load("candidates_evaluation.csv")
        X = cand[db.names].to_numpy(float)
        pr = price_risk(db, X, xb, c["price_samples"], 0.0, c["price_seed"], self.rules.cost_tol)
        for col in pr.columns:
            evf[col] = pr[col].to_numpy()
        rows = [{"rho": 0.0, "samples": c["price_samples"], **self._risk_summary(evf, pr)}]
        for rho in c["price_correlations"]:
            prc = price_risk(db, X, xb, c["price_samples_correlated"], rho, c["price_seed"] + int(100 * rho), self.rules.cost_tol)
            evf[f"p_risk_rho{rho:g}"] = prc["p_risk"].to_numpy()
            rows.append({"rho": rho, "samples": c["price_samples_correlated"], **self._risk_summary(evf, prc)})
        self.save(pd.DataFrame(rows), "price_uncertainty_summary.csv")
        lrows = []
        for spread in c["lca_spreads"]:
            for shared in (False, True):
                p = lca_robust_probability(db, X, xb, self.rules, c["lca_samples"], spread, shared, c["lca_seed"] + int(100 * spread) + shared)
                key = f"p_robust_{int(100 * spread)}_{'shared' if shared else 'indep'}"
                evf[key] = p
                rob = evf["robust"].to_numpy()
                top = self._top_index(evf)
                lrows.append({"spread": spread, "factor": "shared per ingredient" if shared else "independent",
                              "n_P_ge_095": int((p >= 0.95).sum()), "median_P_robust_set": float(np.median(p[rob])),
                              "n_robust_below_05": int((p[rob] < 0.5).sum()), "n_nonrobust_ge_05": int((p[~rob] >= 0.5).sum()),
                              "max_P_nonrobust": float(p[~rob].max()) if (~rob).any() else np.nan,
                              "top20_min_P": float(p[top].min())})
        self.save(pd.DataFrame(lrows), "lca_uncertainty_summary.csv")
        self.save(evf, "candidates_evaluation.csv")
        ranked = rank_robust(evf)
        one, grid = threshold_sensitivity(db, X, list(cand["formula_id"]), xb, self.rules, list(ranked["formula_id"].head(20)))
        self.save(one, "threshold_one_at_a_time.csv")
        self.save(grid, "threshold_grid.csv")

    def _risk_summary(self, evf: pd.DataFrame, pr: pd.DataFrame) -> dict:
        rob = evf["robust"].to_numpy()
        top = self._top_index(evf)
        return {"mean_p_risk_robust": float(pr["p_risk"][rob].mean()), "max_p_risk_robust": float(pr["p_risk"][rob].max()),
                "n_robust_p_risk_gt0": int((pr["p_risk"][rob] > 0).sum()), "top20_max_p_risk": float(pr["p_risk"].to_numpy()[top].max())}

    @staticmethod
    def _top_index(evf: pd.DataFrame, n: int = 20) -> np.ndarray:
        r = rank_robust(evf).head(n)
        return r.index.to_numpy()

    def benchmarks(self) -> None:
        db = self.db
        xb = self.x_bench
        m = self.model()
        evf = self.load("candidates_evaluation.csv")
        ranked = rank_robust(evf)
        rows = []
        for label, energy in [("direct LP", None), ("direct LP, E_A = benchmark", "equal"), ("direct LP, E_A >= benchmark", "at_least")]:
            x = bm.direct_lp(m, xb, self.rules, energy=energy)
            if x is None:
                rows.append({"case": label, "feasible": False}); continue
            e = evaluate(db, x[None, :], xb, self.rules).iloc[0]
            rows.append({"case": label, "feasible": True, "cleaner_score": e.cleaner_score, "cost_change_pct": e.cost_change_pct,
                         "mean_env_change_pct": e.mean_env_change_pct, "n_reduced": int(e.n_reduced), "R3": bool(e.R3),
                         **{f"{k}_change_pct": e[f"{k}_change_pct"] for k in IMPACTS},
                         "ea_mj_kg": x @ db.ea_coef / 1000, "ge_mj_kg": x @ db.ge_coef / 1000, "lipid_pct": x @ db.frac("lipid") / 10,
                         **{n: float(v) for n, v in zip(db.names, x)}})
        direct = pd.DataFrame(rows)
        self.save(direct, "benchmark_direct_lp.csv")
        s_star = float(direct.loc[0, "cleaner_score"])
        best = ranked.iloc[0]
        x_star = direct.loc[0, db.names].to_numpy(float)
        cand = self.load("candidates_composition.csv")
        x_best = cand.loc[cand.formula_id == best.formula_id, db.names].to_numpy(float)[0]
        self.note("direct_lp_score", s_star)
        self.note("best_enumerated", {"id": best.formula_id, "score": float(best.cleaner_score)})
        self.note("best_enumerated_equals_lp", bool(np.allclose(x_best, x_star, atol=1e-3)))
        self.note("enumeration_gap", float(best.cleaner_score - s_star))
        self.note("near_optimal_counts", {str(t): int((ranked.cleaner_score <= s_star + t).sum()) for t in (0.005, 0.01, 0.02)})

        step = self.cfg["benchmarks"]["gl_alpha_step"]
        alphas = np.round(np.arange(0, 1 + 1e-9, step), 4)
        fronts = {}
        for tag, cp_cap in (("cp_cap", True), ("no_cp_cap", False)):
            f = bm.garcia_launay_front(m, xb, alphas, self.cfg["benchmarks"]["gl_impact_cap"], cp_cap)
            ev = evaluate(db, f[db.names].to_numpy(float), xb, self.rules)
            f = pd.concat([f[["alpha", "C_r", "E_m"]], ev[["cleaner_score", "robust", "n_reduced", "cost_change_pct", "mean_env_change_pct"]], f[db.names]], axis=1)
            self.save(f, f"benchmark_garcia_launay_{tag}.csv")
            fronts[tag] = f
            distinct = np.unique(np.round(f[db.names].to_numpy(), 3), axis=0)
            a_lim = bm.alpha_lim(f)
            row = f.loc[np.isclose(f.alpha, a_lim)].iloc[0]
            x_lim = row[db.names].to_numpy(float)
            match = [fid for fid, xx in zip(cand.formula_id, cand[db.names].to_numpy(float)) if np.allclose(xx, x_lim, atol=1e-3)]
            dist = bm.distance_to_front(f, ranked.cost_ratio.to_numpy()[:20], ranked.mean_env_ratio.to_numpy()[:20])
            same_as_lp = f.index[np.all(np.isclose(f[db.names].to_numpy(float), x_star, atol=1e-3), axis=1)]
            self.note(f"gl_{tag}", {"n_alpha": len(f), "n_distinct": len(distinct), "alpha_lim": a_lim,
                                     "alpha_lim_score": float(row.cleaner_score), "alpha_lim_robust": bool(row.robust),
                                     "alpha_lim_candidate": match, "alpha_lim_cost_change_pct": float(row.cost_change_pct),
                                     "alpha_lim_env_change_pct": float(row.mean_env_change_pct),
                                     "alphas_equal_direct_lp": [float(a) for a in f.alpha[same_as_lp]],
                                     "top20_median_gap": float(np.nanmedian(dist)), "top20_max_gap": float(np.nanmax(dist))})

        cases = [(f"poultry proteins <= {v}", {"Poultry proteins": (None, v)}) for v in (100, 150, 200, 250)]
        cases += [(f"meat and bone <= {v}", {"Meat and bone": (None, v)}) for v in (80, 120, 160, 200)]
        cases += [(f"fishmeal >= {v}", {FISHMEAL: (v, None)}) for v in (300, 350, 400)]
        cases += [("no poultry proteins or meat and bone", {"Poultry proteins": (0, 0), "Meat and bone": (0, 0)})]
        lim = bm.inclusion_limit_sensitivity(m, xb, self.rules, cases)
        r = self.cfg["reference"]
        i_low = int(np.argmin(r["fishmeal_levels"]))
        plant = bm.plant_limit_reference(db, self.spec, xb, self.rules, r["fishmeal_levels"][i_low], r["labels"][i_low])
        self.save(pd.concat([lim, plant], ignore_index=True), "benchmark_inclusion_limits.csv")
        self.note("lp_solve_ms", bm.lp_timing(m, xb))
        X = cand[db.names].to_numpy(float)
        t0 = time.perf_counter()
        for _ in range(50):
            evaluate(db, X, xb, self.rules)
        self.note("evaluation_all_candidates_ms", 1e3 * (time.perf_counter() - t0) / 50)

    def growth(self) -> None:
        db = self.db
        gs = GrowthSettings.from_dict(self.cfg["growth"])
        ref = self.load("reference_diets_composition.csv", index_col=0)
        evf = self.load("candidates_evaluation.csv")
        cand = self.load("candidates_composition.csv")
        direct = self.load("benchmark_direct_lp.csv")
        ranked = rank_robust(evf)
        ids = list(ref.index) + list(ranked.formula_id)
        Xr = ref[db.names].to_numpy(float)
        Xc = cand.set_index("formula_id").loc[ranked.formula_id, db.names].to_numpy(float)
        extra_ids, extra_X = [], []
        for _, r in direct[direct.feasible].iterrows():
            extra_ids.append({"direct LP": "LP_opt", "direct LP, E_A = benchmark": "LP_iso", "direct LP, E_A >= benchmark": "LP_ea_ge"}[r.case])
            extra_X.append(r[db.names].to_numpy(float))
        X = np.vstack([Xr, Xc, *([np.array(extra_X)] if extra_X else [])])
        ids = ids + extra_ids
        ea = X @ db.ea_coef / 1000
        unit = X @ db.price / 1000
        summ, ts = simulate_diets(ids, ea, unit, gs)
        comp = composition(db, X, self.spec.premix_kg, self.spec.premix_dm)
        summ = pd.concat([summ, comp.drop(columns="ea_mj_kg")], axis=1)
        summ = pd.concat([summ, self._per_gain_and_protein(X)], axis=1)
        b = summ.set_index("formula_id").loc[self.cfg["reference"]["benchmark"]]
        for col in ["days", "FCR", "ea_mj_kg", "lipid_pct", "feed_cost_population_cny", "feed_cost_per_kg_gain_cny"]:
            summ[f"{col}_pct_vs_benchmark"] = 100 * (summ[col] / b[col] - 1)
        # impacts per kg of predicted weight gain = impact per kg feed x FCR
        ratio = summ["FCR"].to_numpy() / float(b["FCR"])
        for k in IMPACTS:
            summ[f"{k}_per_gain_change_pct"] = 100 * (summ[f"{k}_ratio_feed"] * ratio - 1)
        summ["mean_env_per_gain_change_pct"] = 100 * (summ["mean_env_ratio_feed"] * ratio - 1)
        summ["gwp_kg_co2e_per_kg_gain"] = summ["gwp_kg_co2e_per_kg_feed"] * summ["FCR"]
        summ["delta_feed_cost_per_kg_gain_cny"] = summ["feed_cost_per_kg_gain_cny"] - b["feed_cost_per_kg_gain_cny"]
        summ["delta_gwp_per_kg_gain_kg_co2e"] = summ["gwp_kg_co2e_per_kg_gain"] - b["gwp_kg_co2e_per_kg_feed"] * b["FCR"]
        self.save(summ, "growth_summary.csv")
        self.save(ts, "growth_timeseries.csv")
        top = summ[summ.formula_id.isin(ranked.formula_id.head(20))]
        self.note("growth_cap_binding_below_g", cap_binding_weight(gs))
        self.note("growth_top20_r_ea_days", float(np.corrcoef(top.ea_mj_kg, top.days)[0, 1]))
        self.note("growth_top20_r_lipid_days", float(np.corrcoef(top.lipid_pct, top.days)[0, 1]))
        lit_path = self.cfg["data"].get("adc_literature")
        if lit_path:
            self.adc_check(X[: len(ref) + 20], ids[: len(ref) + 20], gs)

    def _per_gain_and_protein(self, X: np.ndarray) -> pd.DataFrame:
        """Impact ratios and GWP per kg feed, digestible crude protein and the share of crude protein from
        rendered animal proteins, for the rows of X (kg/t)."""
        db = self.db
        ev = evaluate(db, X, self.x_bench, self.rules)
        out = pd.DataFrame({f"{k}_ratio_feed": ev[f"{k}_ratio"].to_numpy() for k in IMPACTS})
        out["mean_env_ratio_feed"] = ev["mean_env_ratio"].to_numpy()
        out["gwp_kg_co2e_per_kg_feed"] = ev["gwp"].to_numpy()
        cp = db.frac("cp")
        out["dcp_pct_database_adc"] = X @ (cp * db.col("adc_protein")) / TOTAL_KG * 100
        lit_path = self.cfg["data"].get("adc_literature")
        if lit_path:
            from .adc_check import compare, load_literature
            cmp = compare(db, load_literature(self.root / lit_path))
            lmb = dict(zip(cmp.ingredient, cmp.lmb_N_mean / 100))
            adc = np.array([lmb.get(n, a) for n, a in zip(db.names, db.col("adc_protein"))])
            out["dcp_pct_lmb_adc"] = X @ (cp * adc) / TOTAL_KG * 100
        ren = [db.idx(n) for n in RENDERED if n in db.names]
        out["cp_share_rendered_pct"] = 100 * (X[:, ren] @ cp[ren]) / (X @ cp)
        return out

    def adc_check(self, X: np.ndarray, ids: list[str], gs: GrowthSettings) -> None:
        """Database ADCs against largemouth bass measurements, and growth with the latter substituted."""
        from .adc_check import compare, growth_scenarios, load_literature
        lit = load_literature(self.root / self.cfg["data"]["adc_literature"])
        cmp = compare(self.db, lit)
        sc = growth_scenarios(self.db, cmp, X, ids, self.cfg["reference"]["benchmark"], gs)
        self.save(lit, "adc_literature_largemouth_bass.csv")
        self.save(cmp, "adc_comparison.csv")
        self.save(sc, "adc_growth_scenarios.csv")
        top = sc.iloc[len(self.cfg["reference"]["labels"]):]
        self.note("adc_n_within_lmb_range", int((cmp.db_N_vs_lmb_range == "within").sum()))
        self.note("adc_n_ingredients_compared", int(len(cmp)))
        for key in ("database", "lmb_N", "lmb_L", "lmb_N_and_L"):
            self.note(f"adc_top20_days_pct_vs_benchmark_{key}_min", float(top[f"days_pct_vs_benchmark_{key}"].min()))
            self.note(f"adc_top20_days_pct_vs_benchmark_{key}_max", float(top[f"days_pct_vs_benchmark_{key}"].max()))

    def attribution(self) -> None:
        """Exact ingredient contributions (Eq. 26) for the reference diets, the top 20 and every candidate."""
        from . import attribution as at
        db, xb, rules = self.db, self.x_bench, self.rules
        cand = self.load("candidates_composition.csv")
        evf = self.load("candidates_evaluation.csv")
        X = cand[db.names].to_numpy(float)
        ref = self.load("reference_diets_composition.csv", index_col=0)
        self.note("attribution_max_abs_error", max(at.check_exact(db, X, xb, rules), at.check_exact(db, ref[db.names].to_numpy(float), xb, rules)))
        b = self.cfg["reference"]["benchmark"]
        others = [d for d in ref.index if d != b]
        top = rank_robust(evf).head(20)
        ids = others + list(top.formula_id)
        Xd = np.vstack([ref.loc[others, db.names].to_numpy(float), cand.set_index("formula_id").loc[top.formula_id, db.names].to_numpy(float)])
        ranks = [None] * len(others) + list(range(1, len(top) + 1))
        tab = at.diet_table(db, Xd, ids, xb, ranks, mean_label="Top-20 mean", mean_rows=slice(len(others), None))
        self.save(tab, "attribution_diets.csv")
        counts, fails = at.rule_failures(db, X, xb, rules)
        self.save(counts, "attribution_rule_counts.csv")
        self.save(fails, "attribution_rule_failures.csv")
        self.save(at.composition_by_class(db, X, xb, evf["robust"].to_numpy()), "attribution_composition_by_class.csv")
        self.save(at.candidate_table(db, X, list(cand.formula_id), xb, rules), "attribution_candidates.csv")

    def figures(self) -> None:
        from . import figures
        figures.make_all(self)

    def tables(self) -> None:
        from . import reporting
        reporting.workbook(self)

    def run(self, stages=None, log=print) -> None:
        for st in stages or STAGES:
            t0 = time.perf_counter()
            log(f"[{st}]")
            getattr(self, st)()
            log(f"[{st}] {time.perf_counter() - t0:.1f} s")
