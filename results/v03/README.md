# v0.3 results

Everything here comes from one Kaggle run over the full collection (63 runs,
31,864 rows) with the corrected ground class list
(`--extra-ground-labels mountain,rock,hill`). Accuracy is measured on 4,927
held-out rows; the interpretability metrics on 240 samples per arm across 8
held-out experiments.

`../v03_summary.csv` is the table every figure below is drawn from.

| file | what it shows | made by |
|---|---|---|
| `f_tradeoff.png` | The headline figure: distractor share against ground coverage, plus the flat accuracy panel. Up and to the right is "looks at the terrain, and at all of it". | `make_tradeoff_figure.py` |
| `f_lambda_sweep.png` | λ_spread 0 → 1 and λ_rrr 0.5 → 2. The spread term is nearly free and the result is not a knife-edge. | notebook cell, from `artifacts/quantify/focus_sweep_*.json` |
| `f_region_audit.png` | What v1's "top half" penalty region was really made of (19% sky, 24% near ground) and the measured sky share, 9.4% not 50%. | drawn from `artifacts/mask_audit/mask_validation.json` |
| `f_mask_audit.png` | The full four-panel depth audit as `validate_sky_mask.py` writes it. | `validate_sky_mask.py` |
| `f_mask_qualitative.png` | Six frames × five columns: RGB, the v1 top-half region, the v0.3 mask, the depth bands, and the containment check. | `validate_sky_mask.py --qualitative 6` |
| `f_mask_fix.png` | Before / after adding `mountain, rock, hill` to the ground class list, on the two frames where the mud was excluded. | derived from two `validate_sky_mask.py` runs |
| `f_mask_regions.png` | Two frames, RGB against the final ground mask. | crop of the qualitative figure |
| `f_compare_row.png` | One frame, five models side by side, each panel annotated with its distractor share and ground coverage. | crop of `compare_unseen_day_wet.png` |
| `f_compare_unseen_day_wet.png` | The full 4×5 heatmap comparison on the unseen wet day. | `compare_interpret.py` |
| `f_compare_unseen_runs_dry.png` | The same on unseen dry runs. | `compare_interpret.py` |
| `f_compare_unseen_runs_wet.png` | The same on unseen wet runs. | `compare_interpret.py` |

## Reading the heatmap comparisons

Column 1 is the cached RGB frame with the ground mask drawn as a contour; the
remaining columns are the input-gradient saliency of the yaw output for each
model, on the same frame with the same colour scale. Each panel's caption gives
the two numbers that matter: the share of saliency outside the ground, and the
effective share of the ground it covers.

The pattern to look for is in columns 3 and 4 (`v1_tophalf`, `nonground`): the
distractor share is zero and the map is a handful of hot spots. Column 5
(`guided`) has the same distractor share and a map spread across the terrain.
