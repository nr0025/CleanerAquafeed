# Cleaner aquafeed formulation screening

[中文说明](README.zh-CN.md)

Code for the paper *A cleaner aquafeed formulation screening framework under nutritional, economic and environmental constraints for largemouth bass*.

The framework screens feed formulations for largemouth bass fingerlings (*Micropterus salmoides*, 10 to 50 g) against a least-cost benchmark with 45% fishmeal. It

1. formulates least-cost reference diets with 35, 45 and 55% fishmeal by linear programming;
2. generates feasible candidates as LP vertices under varied objective weights and as random convex combinations of these vertices;
3. evaluates cost and six ReCiPe 2016 midpoint (H) indicators relative to the benchmark, applies a burden-shifting penalty and classifies the candidates with four eligibility rules;
4. tests the classification against the rule thresholds, ingredient price uncertainty and LCA coefficient uncertainty;
5. compares the shortlist with a direct LP, an iso-energetic LP and a weighted-sum formulation adapted from Garcia-Launay et al. (2018);
6. predicts the days from 10 to 50 g and the FCR of the reference diets and the robust candidates with a bioenergetic model;
7. splits every change in cost and impacts, and every failed rule, into exact ingredient contributions.

Results are given at three levels:

| level | unit |
|---|---|
| ingredient | life-cycle coefficients per kg of as-fed ingredient at the production gate |
| formulation | upstream impacts of the ingredients in 1 kg of as-fed feed |
| predicted production | formulation-level impacts multiplied by the predicted FCR, per kg of predicted weight gain |

## Installation

Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the example

```bash
python scripts/run_pipeline.py --config config/example.toml --out results_example
python -m pytest -q
```

`config/example.toml` uses the example data in `data/example/` and smaller sample sizes, and finishes in a few minutes. The example ingredient database is synthetic: prices and life-cycle coefficients were randomly scaled and compositions slightly perturbed, so the results do not reproduce the paper.

To run the case study of the paper, place the ingredient database in `data/ingredient_database.csv` (columns as in the example file) and run

```bash
python scripts/run_pipeline.py --config config/default.toml --out results
```

Stages can be run separately with `--stages formulation uncertainty benchmarks growth figures tables`; later stages read the outputs of earlier ones. Every random step has a fixed seed in the configuration file.

## Repository layout

```
config/default.toml          settings of the case study (nutrient limits, rules, seeds, sample sizes)
config/example.toml          the same settings with the example data and smaller samples
data/example/                synthetic example data
src/aquafeed_screen/
    data.py                  loading and validation of the ingredient database
    formulation.py           LP model, reference diets, LP vertices, convex combinations (Eqs. 1-12)
    evaluation.py            ratios, burden-shifting penalty, cleaner score, rules R1-R4 (Eqs. 13-20)
    uncertainty.py           price and LCA Monte Carlo, threshold sensitivity (Eqs. 21-22)
    benchmarks.py            direct LP, iso-energetic LP, weighted-sum front, inclusion limits
    growth.py                bioenergetic growth model (Eqs. 23-25)
    adc_check.py             database ADCs against values measured in largemouth bass
    attribution.py           ingredient contributions and attribution of rule failures (Eqs. 26-27)
    figures.py, reporting.py figures and the supplementary workbook
    pipeline.py              run of all stages
scripts/run_pipeline.py      command-line entry point
tests/                       consistency checks (run on the example data)
docs/methods.md              equations and their implementation
```

## Input data

The ingredient database has one row per ingredient:

| column | unit | meaning |
|---|---|---|
| `price_cny_t`, `price_low_cny_t`, `price_high_cny_t` | CNY t⁻¹ | base price and the range used in the Monte Carlo analysis |
| `lb_kg_t`, `ub_kg_t` | kg t⁻¹ feed | practical inclusion limits |
| `dm_pct` … `avp_pct` | % as-fed | dry matter, crude protein, fibre, lipid, ash, starch, lysine, total and available P |
| `adc_lipid`, `adc_protein` | fraction | digestibility coefficients used by the growth model |
| `de_mj_kg` | MJ kg⁻¹ | digestible energy (reported, not used) |
| `gwp`, `feu`, `meu`, `ta`, `lu`, `wu` | per kg ingredient | kg CO₂-eq, kg P-eq, kg N-eq, kg SO₂-eq, m²a crop-eq, m³ |
| `origin` | text | `animal`, `plant` (plant meals, brans and flours) or `oil`; `plant` ingredients are limited together by `plant_max` |

The loader stops with an error if a required value is missing, an ADC lies outside [0, 1] or a base price lies outside its range. `data/example/adc_largemouth_bass.csv` lists digestibility coefficients measured in largemouth bass, taken from the studies in `adc_largemouth_bass_references.csv`; they are used only to check the database values.

To apply the framework to another species, stage or market, edit the database and the `[nutrients]`, `[reference]`, `[candidates]` and `[growth]` sections of the configuration. The growth parameters are specific to largemouth bass.

## Outputs

| file | content |
|---|---|
| `reference_diets_*.csv` | composition and evaluation of FM35, FM45 and FM55 |
| `candidates_composition.csv`, `candidates_evaluation.csv` | every candidate at the formulation level, its class, P_risk and P_robust |
| `threshold_*.csv`, `lca_uncertainty_summary.csv`, `price_uncertainty_summary.csv` | robustness analyses |
| `benchmark_*.csv` | direct and iso-energetic LP, weighted-sum front, inclusion limits, plant meal limit |
| `growth_summary.csv`, `growth_timeseries.csv` | days to 50 g, FCR, feed cost and impacts per kg of predicted gain |
| `adc_*.csv` | database ADCs against largemouth bass values, and growth with the latter substituted |
| `attribution_*.csv` | ingredient contributions and attribution of rule failures |
| `figures/` | figures (PNG and SVG) |
| `supplementary_tables.xlsx` | all tables in one workbook |
| `summary.json` | main numbers of the run |

Figures use Times New Roman, or Liberation Serif or TeX Gyre Termes where it is not installed.

## Citation

Please cite the paper (see `CITATION.cff`).

## Licence

PolyForm Noncommercial License 1.0.0, see `LICENSE`. Research, teaching and other noncommercial use is permitted. Please open an issue for questions or problems.
