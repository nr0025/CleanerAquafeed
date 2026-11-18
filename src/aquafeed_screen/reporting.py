"""Supplementary workbook assembled from the result tables."""
from __future__ import annotations

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .data import IMPACT_LABELS, IMPACT_UNITS, IMPACTS


def workbook(run, name: str = "supplementary_tables.xlsx") -> None:
    out = run.out
    exists = lambda f: (out / f).exists()  # noqa: E731
    sheets: list[tuple[str, pd.DataFrame, str]] = []
    t = run.db.table
    s1 = pd.DataFrame({
        "Ingredient": t["name"], "Group": t["origin"] if "origin" in t else "",
        "Price, base (CNY/t)": t["price"], "Price, low (CNY/t)": t["price_low"], "Price, high (CNY/t)": t["price_high"],
        "Inclusion, min (kg/t)": t["lb"], "Inclusion, max (kg/t)": t["ub"],
        "Dry matter (%)": t["dm"], "Crude protein (%)": t["cp"], "Crude fibre (%)": t["fibre"], "Crude lipid (%)": t["lipid"],
        "Ash (%)": t["ash"], "Starch (%)": t["starch"], "Lysine (%)": t["lys"], "Total P (%)": t["p"]})
    if "avp" in t:
        s1["Available P (%)"] = t["avp"]
    s1["ADC-N (%)"] = 100 * t["adc_protein"]
    s1["ADC-L (%)"] = 100 * t["adc_lipid"]
    if "de" in t:
        s1["Digestible energy, reported (MJ/kg)"] = t["de"]
    b = run.cfg["reference"]["benchmark"]
    sheets.append(("S1a_ingredient_database", s1,
                   "Ingredient database (25 ingredients; composition as-fed). Price: base price used in the screening and the market range used in the "
                   "price uncertainty analysis. Inclusion limits: practical limits LB_i and UB_i of Equation (5); for candidate formulations "
                   f"fishmeal was additionally limited to at most its level in {b} (Equation 7), and the ingredients of the group 'plant' were limited "
                   "together to 450 kg/t (Table 1). ADC-N: salmonid nitrogen digestibility and ADC-L: salmonid energy digestibility (used because the "
                   "tables give no lipid digestibility for fish), INRAE-CIRAD-AFZ feed tables (Feedipedia); compared with largemouth bass values in "
                   "Section S3 and Table S11. Digestible energy is the value reported in the source and is not used in any calculation."))
    src = run.root / run.cfg["data"].get("price_sources", "data/price_sources.csv")
    if src.exists():
        ps = pd.read_csv(src).rename(columns={"source": "Source", "website": "Website", "access_date": "Access date", "note": "Data"})
        sheets.append(("S1b_price_sources", ps, "Sources of the ingredient prices and price ranges in Table S1a."))
    s2 = pd.DataFrame({"Ingredient": t["name"]})
    for k in IMPACTS:
        s2[f"{IMPACT_LABELS[k]} ({IMPACT_UNITS[k]}/kg)"] = t[k]
    sheets.append(("S2_LCA_coefficients", s2,
                   "Ingredient level: life-cycle impact coefficients per kg of as-fed ingredient at the production gate (cradle-to-gate), calculated in SimaPro 9.0.0 with "
                   "Agri-footprint and ecoinvent 3 inventory datasets, economic allocation and ReCiPe 2016 Midpoint (H). GWP, global warming potential; "
                   "FEU, freshwater eutrophication; LU, land use; WU, water consumption; MEU, marine eutrophication; TA, terrestrial acidification."))
    sheets.append(("S3a_thresholds", run.load("threshold_one_at_a_time.csv"),
                   "Robust-set size when one eligibility threshold is changed at a time (base: cost <= +10 %, GWP < benchmark, >= 4 of 6 indicators reduced, no indicator > +10 %)."))
    sheets.append(("S3b_threshold_grid", run.load("threshold_grid.csv"), "All 80 combinations of cost tolerance, minimum number of reduced indicators and burden-shifting tolerance."))
    sheets.append(("S5a_LCA_uncertainty", run.load("lca_uncertainty_summary.csv"),
                   "Probability of satisfying R1-R4 when every impact factor is multiplied by U(1 - a, 1 + a); 2,000 samples."))
    sheets.append(("S5b_price_uncertainty", run.load("price_uncertainty_summary.csv"),
                   "Cost-risk probability with independent (rho = 0) and correlated ingredient prices (Gaussian copula, one common factor)."))
    ev = run.load("candidates_evaluation.csv")
    pcols = ["formula_id", "robust", "cleaner_score", "p_risk"] + [c for c in ev.columns if c.startswith("p_risk_rho") or c.startswith("p_robust_")]
    sheets.append(("S5c_per_formulation", ev[pcols], "Cost-risk probability and P_robust of every candidate."))
    if exists("attribution_diets.csv"):
        sheets.append(("S6_contrib_diets", run.load("attribution_diets.csv"),
                       "Formulation level: exact contributions (percentage points, Equation 26) of each ingredient to the change in feed cost and in the six impact indicators relative to FM45 "
                       "for the other reference diets, the 20 top-ranked robust cleaner candidates (rank 1-20) and their mean; delta_kg_t = change in inclusion. "
                       "The 'Total' row of each diet equals its change relative to FM45 (%); mean_env_pp is the contribution to the mean of the six impact ratios."))
        sheets.append(("S7a_rule_counts", run.load("attribution_rule_counts.csv"),
                       f"Number of the {len(ev):,} candidates failing each eligibility rule (a candidate can fail several)."))
        sheets.append(("S7b_rule_attribution", run.load("attribution_rule_failures.csv"),
                       "Failed rules attributed to the violating indicator (R1: cost; R2: GWP; R3: each indicator not reduced; R4: each indicator above 1.10 x FM45) "
                       "and to the ingredient with the largest positive contribution to it; n_cases = number of candidate-indicator cases."))
        sheets.append(("S7c_composition_by_class", run.load("attribution_composition_by_class.csv"),
                       "Mean inclusion (kg/t) of robust and non-robust candidates and of the FM45 benchmark."))
        sheets.append(("S8_contrib_candidates", run.load("attribution_candidates.csv"),
                       "Every candidate: failed rules, indicators above the burden-shifting threshold and, for cost, each impact and the mean impact, the change relative to FM45 (%) "
                       "and the ingredients with the largest positive and negative contributions (percentage points). The full contribution array is produced by attribution.contributions()."))
    sheets.append(("S9a_direct_LP", run.load("benchmark_direct_lp.csv"), "Direct LP optimum of Equation (20) under R1, R2 and R4, without and with an assimilable-energy constraint."))
    sheets.append(("S9b_GL_front", run.load("benchmark_garcia_launay_cp_cap.csv"),
                   "Weighted-sum formulation adapted from Garcia-Launay et al. (2018): min (1 - alpha) C_r + alpha E_m, every indicator <= 105 % of FM45, CP <= CP of FM45."))
    sheets.append(("S9c_GL_no_cp_cap", run.load("benchmark_garcia_launay_no_cp_cap.csv"), "As S9b without the crude-protein cap."))
    sheets.append(("S9d_inclusion_limits", run.load("benchmark_inclusion_limits.csv"), "Direct LP optimum of Equation (20) with altered inclusion limits, and the least-cost FM35 reference diet (Equation 1) re-formulated with the limit on total plant meals at 45 % (base case), 50 % and without limit; all changes relative to the base-case FM45 benchmark."))
    sheets.append(("S10_growth", run.load("growth_summary.csv"), "Predicted production level: growth from 10 g to 50 g at 28 C (days, feed use, FCR) and feed cost for 2,000 fish; impacts per kg of predicted weight gain, equal to the formulation-level impacts multiplied by the FCR, and their change relative to FM45; digestible crude protein with the database ADCs and with the largemouth bass mean ADC-N of Table S11b; share of crude protein from meat-and-bone meal and poultry proteins."))
    if exists("adc_comparison.csv"):
        b = run.cfg["reference"]["benchmark"]
        lit = run.load("adc_literature_largemouth_bass.csv").rename(columns={
            "db_ingredient": "Database ingredient", "source": "Source", "source_ingredient": "Ingredient in source",
            "method": "Method (reference:test diet, marker, faeces collection)", "fish_weight_g": "Fish weight (g)",
            "adc_cp_pct": "ADC crude protein (%)", "adc_lipid_pct": "ADC lipid (%)", "adc_lys_pct": "ADC lysine (%)",
            "adc_total_aa_pct": "ADC total amino acids (%)", "source_type": "Source type"})
        cmp = run.load("adc_comparison.csv").rename(columns={
            "ingredient": "Ingredient", "n_values_N": "n (N)", "basis_N": "Basis of largemouth bass N value",
            "db_adc_N_pct": "Database ADC-N (%)", "lmb_N_min": "Largemouth bass min (%)", "lmb_N_max": "Largemouth bass max (%)",
            "lmb_N_mean": "Largemouth bass mean (%)", "db_minus_lmb_mean_N": "Database - mean (points)",
            "db_N_vs_lmb_range": "Database value vs range", "n_values_L": "n (lipid)", "db_adc_L_pct": "Database ADC-L (%)",
            "lmb_L_min": "Largemouth bass lipid min (%)", "lmb_L_max": "Largemouth bass lipid max (%)",
            "lmb_L_mean": "Largemouth bass lipid mean (%)", "db_minus_lmb_mean_L": "Database - mean, lipid (points)",
            "lmb_lys_min": "Largemouth bass lysine min (%)", "lmb_lys_max": "Largemouth bass lysine max (%)", "sources": "Sources"})
        names = {"database": "database ADCs", "lmb_N": "largemouth bass ADC-N", "lmb_L": "largemouth bass ADC-L",
                 "lmb_N_and_L": "largemouth bass ADC-N and ADC-L"}
        sc = run.load("adc_growth_scenarios.csv")
        ren = {"formula_id": "Diet"}
        for k, v in names.items():
            ren.update({f"ea_{k}": f"E_A, {v} (MJ/kg)", f"days_{k}": f"Days to 50 g, {v}", f"FCR_{k}": f"FCR, {v}",
                        f"days_pct_vs_benchmark_{k}": f"Days to 50 g vs {b}, {v} (%)"})
        sc = sc.rename(columns=ren)
        refs_path = run.root / run.cfg["data"].get("adc_references", "data/adc_largemouth_bass_references.csv")
        sheets.append(("S11a_ADC_LMB_literature", lit,
                       "Apparent digestibility coefficients (%) measured in largemouth bass for ingredients of the database. "
                       "Masagounder et al. (2009) report amino-acid but not crude-protein digestibility; Zou (2021) is cited from Ma et al. (2022)."))
        sheets.append(("S11b_ADC_comparison", cmp,
                       "Database ADC-N (salmonid nitrogen digestibility, INRAE-CIRAD-AFZ feed tables) and ADC-L (salmonid energy digestibility, used because the tables give no lipid digestibility for fish) "
                       "against the largemouth bass values of S11a (range and mean; total amino acids where crude protein was not reported). "
                       "The last column of the N block gives the position of the database value relative to the largemouth bass range."))
        sheets.append(("S11c_ADC_growth", sc,
                       "Assimilable energy (MJ/kg), days to 50 g and FCR of the reference diets and the top 20 robust candidates with the database ADCs, "
                       "with the largemouth bass mean ADC-N of S11b substituted, with the mean ADC-L substituted, and with both substituted."))
        if refs_path.exists():
            sheets.append(("S11d_references", pd.read_csv(refs_path).rename(columns={"source": "Source", "reference": "Reference"}),
                           "References for Tables S11a-S11c."))
    sheets.append(("Data_reference_diets", run.load("reference_diets_composition.csv").rename(columns={"Unnamed: 0": "diet"}), "Reference diets, kg/t."))
    sheets.append(("Data_candidates", run.load("candidates_composition.csv"), f"All candidate formulations, kg/t (premix {run.cfg['nutrients']['premix_kg']:g} kg/t not shown)."))
    sheets.append(("Data_evaluation", ev, "Formulation level: cost, impacts per kg of as-fed feed, change relative to FM45, eligibility rules and cleaner score of all candidates."))
    levels = [("Result levels", "Ingredient level: life-cycle coefficients per kg of as-fed ingredient at the production gate. "
               "Formulation level: upstream impacts of the ingredients in 1 kg of as-fed feed. "
               "Predicted production level: formulation-level impacts multiplied by the predicted FCR, per kg of predicted weight gain.")]
    readme = pd.DataFrame(levels + [(n, d) for n, _, d in sheets], columns=["Sheet", "Content"])
    path = out / name
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        for n, df, _ in sheets:
            df.to_excel(xw, sheet_name=n, index=False)
    wb = load_workbook(path)
    for ws in wb:
        for c in ws[1]:
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="E8EEF6")
        for i, col in enumerate(ws.columns, 1):
            w = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
            ws.column_dimensions[get_column_letter(i)].width = min(max(9, w + 2), 100 if ws.title == "README" else 28)
            if ws.title != "README":
                for c in col[1:]:
                    if isinstance(c.value, float):
                        v = abs(c.value)
                        if ws.title.startswith("S11"):
                            c.number_format = "0.00" if str(col[0].value).startswith("E_A") or "weight (g)" in str(col[0].value) else "0.0"
                        else:
                            c.number_format = "0.00E+00" if 0 < v < 0.01 else ("0.0000" if v < 100 else "0.0")
        for row in ws.iter_rows(min_row=2):
            for c in row:
                if isinstance(c.value, str) and "TO BE COMPLETED" in c.value:
                    c.fill = PatternFill("solid", fgColor="FFF2A8")
        if ws.title == "S11d_references":
            ws.column_dimensions["B"].width = 120
        if ws.title in ("README", "S11d_references"):
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
        else:
            ws.freeze_panes = "B2"
    wb.save(path)
