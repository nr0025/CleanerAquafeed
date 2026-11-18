# Methods and implementation

[中文](methods.zh-CN.md)

Equation numbers refer to the paper. Inclusions *x*ᵢ are in kg per tonne of complete feed; a fixed premix of 20 kg t⁻¹ completes the tonne and is not counted in nutrients, dry matter, energy, cost or impacts. Nutrient limits apply to total nutrients on an as-fed basis. All formulations, including the reference diets, also respect a limit on total plant meals.

## Result levels

* Ingredient level: life-cycle coefficients per kg of as-fed ingredient at the production gate (database columns `gwp` … `wu`).
* Formulation level: upstream impacts of the ingredients in 1 kg of as-fed feed (`*_ratio`, `*_change_pct` in `candidates_evaluation.csv`).
* Predicted production level: formulation-level impacts multiplied by the predicted FCR, per kg of predicted weight gain (`*_per_gain_*` in `growth_summary.csv`).

## 1. Linear formulation model — `formulation.py`

| item | equation | implementation |
|---|---|---|
| mass balance | Σ*x*ᵢ + *x*ₚₘ = 1000 | `base_model` |
| nutrient limits (Table 1) | *N*ₙᵐⁱⁿ ≤ Σ*a*ᵢₙ*x*ᵢ/1000 ≤ *N*ₙᵐᵃˣ | `base_model`, limits from `[nutrients]` |
| lysine/CP ratio | Σ*a*ᵢ,Lys*x*ᵢ ≥ ρ Σ*a*ᵢ,CP*x*ᵢ (linear form, Eq. 4) | `base_model` |
| total plant meals | Σ_{*i* ∈ plant} *x*ᵢ ≤ 450 kg t⁻¹ (plant-derived ingredients except oils) | `base_model`, limit `plant_max`, ingredients with `origin = plant` |
| inclusion limits | *LB*ᵢ ≤ *x*ᵢ ≤ *UB*ᵢ | database columns `lb_kg_t`, `ub_kg_t` |
| dry matter | (Σ DMᵢ*x*ᵢ + DMₚₘ*x*ₚₘ)/1000 ≥ 89% | premix dry matter set by `premix_dm` (0 by default) |
| reference diets | min Σ*P*ᵢ*x*ᵢ with *x*_FM fixed (Eqs. 1–6) | `reference_diets`; full ingredient database |
| candidates | Eqs. 2–5, 300 ≤ *x*_FM ≤ 450, GE ≥ max(12.5, 0.97 GE_FM45) (Eqs. 7–9) | `candidate_model` |

All problems are solved with HiGHS through `scipy.optimize.linprog`. Because constraints and objectives are linear, every solve returns a vertex of the feasible polytope and the optimum is global.

### Candidate generation (Eqs. 10–12)

* Normalised coefficients: *p*ᵢ = *P*ᵢ / *C*_FM45 and *e*ᵢₖ = *E*ᵢₖ / Σⱼ *x*ⱼ,FM45 *E*ⱼₖ, so that Σ*x*ᵢ*p*ᵢ = *C*ᵣ and Σ*x*ᵢ*e*ᵢₖ = *R*ₖ (`normalised_coefficients`).
* Named objectives: minimum cost at base and at high prices, minimum of each of the six indicators, and the compromise 0.5 *C*ᵣ + 0.5 *E*ₘ.
* 2,000 random objectives: cost weight *w*₀ ~ U(0, 1); the remaining 1 − *w*₀ is split among the six indicators by a flat Dirichlet draw.
* Tie-breaking perturbation: εᵢ = 10⁻⁴ · mean|*c*| · *u*ᵢ, *u*ᵢ ~ U(0, 1).
* Vertices equal to within 10⁻⁴ kg t⁻¹ are merged.
* Convex combinations: 1,000 draws; each draw takes *k* ~ U{2, …, 5} distinct vertices and weights λ ~ Dirichlet(1, …, 1). Every candidate is checked against all constraints before it is kept.

The candidate set samples the convex hull of the vertices that were found, not the whole feasible polytope, and the share of candidates in each class depends on the sampling design.

## 2. Evaluation and classification — `evaluation.py`

* *C*ᵣ = *C*_f / *C*_FM45 (Eq. 15), *R*ₖ = *I*_f,k / *I*_FM45,k (Eq. 16), *E*ₘ = mean of the six *R*ₖ (Eq. 17).
* Burden-shifting penalty *B* = Σ max(0, *R*ₖ − 1.10) (Eq. 18); score *S* = 0.5 *C*ᵣ + 0.5 *E*ₘ + 2 *B* (Eq. 19).
* Rules: R1 *C*ᵣ ≤ 1.10; R2 *R*_GWP < 1; R3 at least four *R*ₖ < 1; R4 *B* = 0. Robust candidates are ranked by Eq. (20).
* Other classes, in order of precedence: exceeds cost tolerance (not R1); lower GWP with burden shifting (R2 and *B* > 0); partial improvement without GWP reduction (*R*_GWP ≥ 1 and *E*ₘ < 1); otherwise within cost tolerance with limited gain.
* Costs exclude the premix, which is identical in all diets.

## 3. Uncertainty — `uncertainty.py`

* Prices: uniform between the low and high estimates, independent (2,000 samples) or with one common factor through a Gaussian copula with pairwise latent correlation ρ = 0.3, 0.6, 0.9 (2,000 samples). Candidate and benchmark costs use the same sample; *P*_risk is the share of samples with *C*ᵣ > 1.10 (Eq. 22).
* LCA coefficients: each *E*ᵢₖ multiplied by U(0.7, 1.3) or U(0.5, 1.5), independently or with one factor per ingredient shared by the six indicators (2,000 samples); *P*_robust is the share of samples in which R1–R4 hold, with cost at its deterministic value.
* Thresholds: cost tolerance 0–20%, minimum reduced indicators 3–6, burden tolerance 0–20% (80 combinations), and removal of R2.

## 4. Benchmarks — `benchmarks.py`

* Direct LP: minimum of Eq. (20) under the candidate constraints and R1, R2, R4 (R3 is checked afterwards).
* Iso-energetic LP: the same with Σ*x*ᵢ EAᵢ equal to that of FM45. Because diet composition enters the growth model only through *E*_A, this fixes predicted growth at that of the benchmark.
* Weighted-sum formulation after Garcia-Launay et al. (2018, Br. J. Nutr. 120, 1298–1309): min (1 − α) *C*ᵣ + α *E*ₘ under the candidate constraints, every *R*ₖ ≤ 1.05, crude protein ≤ that of FM45, α = 0, 0.01, …, 1. The preferred solution α_lim is the point beyond which the marginal decrease in *E*ₘ is smaller than the marginal increase in *C*ᵣ. Differences from the original: FM45 instead of the least-cost diet as reference and the equally weighted six ReCiPe indicators instead of the original weighted index of four impacts. With both indices normalised by FM45, Eq. (20) equals this objective at α = 0.5.
* Inclusion-limit sensitivity: the direct LP re-solved with altered limits for poultry proteins, meat-and-bone meal and fishmeal, and without either rendered animal protein. The least-cost FM35 diet is also re-formulated with the limit on total plant meals at 45 % (base), 50 % and without limit, relative to the base-case FM45 (Table S9d).

## 5. Growth model — `growth.py`

* *E*_A = Σ*x*ᵢ (23.6 CPᵢ ADC_N,ᵢ + 39.5 Lᵢ ADC_L,ᵢ)/1000, MJ kg⁻¹ (Eq. 23). ADC_N is the salmonid nitrogen digestibility and ADC_L the salmonid energy digestibility of the INRAE-CIRAD-AFZ feed tables; Table S11 compares them with values measured in largemouth bass.
* Intake *C* = min(0.065 *W*⁰·⁷⁵ τ(*T*), 0.03 *W*) g d⁻¹, as-fed (Eq. 24); τ(*T*) is a Kitchell-type function with *T*_max = 37 °C, *T*_opt = 27.5 °C and *Q*₁₀ = 2.65.
* d*W*/d*t* = ((1 − 0.20 − 0.15) *C* *E*_A − *R*_A) / *E*_f with *E*_f = 5.01 *W*⁰·⁰⁴⁶ kJ g⁻¹ (Eq. 25); respiration *R*_A = 13.56 · 0.3393 *W*^−0.3486 e^0.028T e^0.0196·5 *W* · 0.024 kJ d⁻¹.
* From 10 g to 50 g at 28 °C (the simulation stops at 50 g); 2,000 fish scale feed cost only. Integration with RK45 (relative tolerance 10⁻⁸).
* Intake, respiration and energy density do not depend on the diet, so the days needed to reach 50 g decrease monotonically with *E*_A (checked in `tests/`).
* Table S10 also gives, for every simulated diet: impacts per kg of predicted weight gain relative to FM45 (impact ratio per kg feed × FCR/FCR_FM45); digestible crude protein Σ*x*ᵢ CPᵢ ADC_N,ᵢ with the database ADCs and with the largemouth bass mean ADC-N substituted; and the share of crude protein supplied by meat-and-bone meal and poultry proteins.

## 6. Ingredient attribution — `attribution.py`

* Cost and the six impacts are linear in the inclusions, so the change of each ratio relative to FM45 splits exactly into one term per ingredient (Eq. 26): *R*ₖ − 1 = Σᵢ (*x*ᵢ − *x*ᵢ,FM45) *E*ᵢ,ₖ / (1000 *I*_FM45,k) and *C*ᵣ − 1 = Σᵢ (*x*ᵢ − *x*ᵢ,FM45) *P*ᵢ / *C*_FM45. The contribution to *E*ₘ is the mean over the six indicators.
* `contributions()` returns the array formulations × ingredients × (cost, six impacts); `check_exact()` confirms that the sums reproduce `evaluate()` (also checked in `tests/`).
* No model is fitted, so there is no validation step and collinearity between inclusions does not affect the result.
* Rule failures: R1 is attributed to cost, R2 to GWP, R3 to each indicator with *R*ₖ ≥ 1 and R4 to each indicator with *R*ₖ > 1.10, and each of these to the ingredient with the largest positive contribution.
* Outputs: `attribution_diets.csv` (FM35, FM55, the top 20 and their mean; Table 4, Table S6), `attribution_rule_counts.csv` and `attribution_rule_failures.csv` (Table S7a–b), `attribution_composition_by_class.csv` (Table S7c), `attribution_candidates.csv` (Table S8).
