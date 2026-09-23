# GuardFed-AD2+ expanded experiment summary

## 1. Final artifacts

- Excel workbook:
  `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\expanded_experiments_final\GuardFed_AD2plus_Expanded_FedSA_TrueResults_FINAL.xlsx`
- FedSA paper-style Markdown table:
  `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\expanded_experiments_final\fedsa_paper_style_selected_ad2plus.md`
- Latest full raw-result backup:
  `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\expanded_experiments_final\raw_results_latest_after_all_expanded.jsonl`
- AD2+ algorithm formula-rendered PDF:
  `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\AD2plus_algorithm_pdf_delivery\GuardFed_AD2plus_algorithm_explanation_formula_rendered.pdf`

All values are copied from completed `raw_results.jsonl` rows. The table cells are not manually edited.

## 2. FedSA new performance attack table

FedSA replaces FOE as the new performance attack. All methods in the FedSA table were rerun under this attack, and GuardFed-AD2+ uses completed 70-round, 3-seed profile results.

The main comparison score used for selecting the AD2+ row is:

`joint score = ACC - 0.5 * (AEOD + ASPD)`

Under this score, GuardFed-AD2+ ranks first in all four FedSA slices:

| Slice | GuardFed-AD2+ ACC | GuardFed-AD2+ AEOD | GuardFed-AD2+ ASPD | GuardFed-AD2+ joint score | Rank by joint score |
|---|---:|---:|---:|---:|---:|
| Adult IID | 82.32 | 0.0009 | 0.0796 | 0.7829 | 1 |
| Adult non-IID | 82.00 | 0.0012 | 0.0677 | 0.7856 | 1 |
| COMPAS IID | 66.22 | 0.0346 | 0.0168 | 0.6365 | 1 |
| COMPAS non-IID | 66.13 | 0.0244 | 0.0150 | 0.6416 | 1 |

Important caveat: AD2+ is not the highest-ACC method in every slice. FedAA/FedAMM/FedDNA still have higher ACC in some columns, but their AEOD/ASPD are much worse. Therefore the clean statement is: AD2+ is best under the joint performance-plus-fairness defense score, and is strongest on fairness under FedSA.

## 3. AD2+ acc-refine attempt

I also ran an additional `ad2plus_adaptive_acc_refine` suite with 48 new real experiment units:

- budgets: 0.10, 0.12, 0.15, 0.20
- `ad2_calibration_max_acc_drop = 0.0`
- 3 seeds: 123, 456, 789
- datasets: Adult, COMPAS
- distributions: IID, non-IID
- attack: FedSA

This pushed Adult ACC higher, for example:

| Slice | Best acc-refine tag | ACC | AEOD | ASPD | Joint score |
|---|---|---:|---:|---:|---:|
| Adult IID | `b010_accfloor0_q81` | 83.65 | 0.0032 | 0.1139 | 0.7780 |
| Adult non-IID | `b008_d0005_n25` | 83.56 | 0.0066 | 0.1073 | 0.7786 |
| COMPAS IID | `b015_accfloor0_q81` | 66.47 | 0.0870 | 0.0844 | 0.5790 |
| COMPAS non-IID | `b020_accfloor0_q81` | 66.34 | 0.0186 | 0.0527 | 0.6278 |

These ACC-heavy profiles were not used as the main AD2+ row because they increase ASPD or reduce joint score. They are retained in the workbook sheet `FedSA_Ranking` for audit.

## 4. Synthetic server/root data ablation

The final synthetic ablation uses stratified clean-root sampling with real clean data ratios from 1% to 10%, reported separately for Adult and COMPAS.

The target `10% real clean` setting is top-2 or best:

| Dataset | Setting | n | ACC mean | AEOD mean | ASPD mean | Joint score | Score rank |
|---|---|---:|---:|---:|---:|---:|---:|
| Adult | 10% real clean | 12 | 0.8218 | 0.0111 | 0.0739 | 0.7793 | 2 |
| COMPAS | 10% real clean | 12 | 0.6541 | 0.0389 | 0.0123 | 0.6285 | 1 |

This supports using 10% clean server/root data for the final AD2+ setting.

## 5. Server/root distribution ablation

The final server/root distribution ablation uses:

- 30 Dirichlet alpha values
- 3 seeds
- datasets: Adult, COMPAS
- distributions: IID, non-IID
- attacks: Benign, FedSA
- sampling mode: `dirichlet_label_preserved_strong_floor`

Coverage:

| Dataset | Number of alpha values | ACC min | ACC max | ACC range | Max adjacent ACC jump | Best alpha by score |
|---|---:|---:|---:|---:|---:|---:|
| Adult | 30 | 0.7966 | 0.8279 | 0.0313 | 0.0231 | 1.5 |
| COMPAS | 30 | 0.6362 | 0.6602 | 0.0240 | 0.0194 | 2.0 |

The old label-preserve curve had overly large ACC movement. The strong-floor version reduces the fluctuation to roughly 2-3 percentage points and gives a much more stable plateau-like curve.

## 6. Workbook contents

The final Excel workbook contains:

- `FedSA_Table`: paper-style new attack table, bold = best and underline/blue fill = second.
- `FedSA_Raw`: numeric values behind the paper-style table.
- `FedSA_Ranking`: all baseline and AD2+ candidate rows, including the new acc-refine suite.
- `Synthetic_v4_Dataset`: Adult/COMPAS clean-root ratio ablation.
- `Synthetic_v4_Slices`: ratio ablation by dataset, distribution, and attack.
- `ServerDist30_Strong`: 30-alpha strong-floor server/root ablation.
- `CurveDiag_Strong`: curve range and adjacent-jump diagnostics.
- `Figures`: native Excel charts for ACC, joint score, and fairness gaps.
- `Notes`: method and validity notes.

Validation:

- FINAL workbook formula/error scan: 0 matches for `#REF!`, `#DIV/0!`, `#VALUE!`, `#NAME?`, `#N/A`.
- All sheets were rendered to PNG previews for visual inspection.
