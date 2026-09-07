# vit_motion — making a motion transformer look at the ground

A ViT-tiny + temporal-fusion transformer predicts the next motion step
`[dx, dy, dyaw]` of a tracked ground vehicle from one cached RGB frame plus six
numeric steps. This repository is the **interpretability** side of that model:
where in the frame it looks, how to measure that, and what happens when you push
it to look at the terrain instead of the sky and the buildings.

Adaptive Photonics Lab, NCTU.

---

## The one-paragraph result

Penalizing the saliency that lands outside the ground works — 19× less attention
on sky and objects — but on its own it **collapses** the remaining attention onto
a handful of ground pixels (`coverage` 0.89 → 0.22). Adding an entropy term that
asks the attention to be *spread out inside* the ground recovers it: the guided
model reaches the same 1% distractor share while covering 87% of the terrain,
slightly more evenly than the unguided model. **Accuracy does not move**: the
whole spread across six arms is 0.011 R², and a no-penalty control with the same
extra epochs scores best of the six. That last sentence is the finding, not a
caveat.

| run | distractor ↓ | concentration ↑ | coverage ↑ | top 10% ↓ | yaw R² |
|---|---:|---:|---:|---:|---:|
| `v03_base` | 0.193 | 1.34 | 0.886 | 0.217 | 0.841 |
| `control` | 0.165 | 1.38 | 0.864 | 0.234 | 0.852 |
| `v1_tophalf` | 0.010 | 1.65 | **0.248** | 0.666 | 0.848 |
| `sky_only` | 0.048 | 1.58 | 0.662 | 0.354 | 0.847 |
| `nonground` | **0.003** | 1.66 | **0.215** | 0.737 | 0.849 |
| **`guided`** | 0.010 | 1.65 | **0.869** | **0.197** | 0.847 |

240 samples per arm over 8 held-out experiments; accuracy pooled over 4,927
held-out rows. `results/v03_summary.csv` is the source of this table.

---

## Method, in three parts

### 1. One quantity, penalized and measured and drawn

```
saliency = | ∂ target / ∂ image |     summed over colour channels, target = yaw
```

The earlier version penalized one quantity and visualized another, so the number
and the heatmap could disagree without either being wrong. Everything here — the
loss term, the reported metric, the figure — is this same scalar.

### 2. The region is the ground, from a segmenter, not "the top half"

v1 used the top half of the frame as a stand-in for "useless". Audited against
the aligned depth frames, that region is only 19% sky and **24% near ground** —
it was penalizing the surface the vehicle drives on. `precompute_sky_masks.py`
replaces it with a per-frame SegFormer-b0 (ADE20K) mask cached as an 8-bit
region-code PNG at the model's input geometry.

The ADE20K class list is diagnosed rather than guessed: `diagnose_ground_classes.py`
samples frames, asks the segmenter what it calls the terrain band versus the top
band, and writes crops with each candidate highlighted. On this collection it
found that `mountain` and `rock` fire on the muddy ruts in front of the vehicle
(0.06% / 0.04% of the upper band — there is no mountain in these scenes), because
ADE20K has no class for churned mud. They are added with
`--extra-ground-labels mountain,rock,hill`, and the list actually used is
recorded in the cache's `index.json`.

### 3. Two loss terms, and no change to the model

```
total_loss = task_loss + λ_rrr · placement + λ_spread · spread

placement = Σ(saliency · distractor_mask) / Σ saliency        → minimize
spread    = 1 − H(saliency | ground) / log N_ground           → minimize
```

The architecture is untouched: zero new parameters, zero new layers, the same
`forward` signature. The terms attach to the **input gradient**, via
`create_graph=True` on `torch.autograd.grad` — so `loss.backward()` computes a
second derivative through attention (double backprop, as in Ross et al. 2017).
That is also why the fine-tune forces the math SDPA kernel: the fused
flash/mem-efficient kernels do not implement double-backward.

`spread` is the loss, `coverage = exp(H)/N` is the reported metric. The split is
deliberate — coverage's gradient is proportional to itself and so vanishes
exactly in the collapsed regime the term exists to escape.

---

## Layout

The root is flat on purpose: the Kaggle pipeline uploads these files as one zip
and runs them from a flat working directory. Do not reorganize without updating
`make_notebook.py`.

```
vit_motion/                  the package
  model.py                   ViT-tiny encoder + temporal fusion + regression head
  dataset.py                 windowed dataset; attaches the cached region masks
  sky_mask.py                region masks, depth bands, the class list, quality test
  attention_focus.py         placement / spread — numpy for reports, torch for the loss
  interpret.py               saliency and heatmap rendering
  manifest.py, config.py, validation.py

train.py                     base model (no masks, no guidance)
finetune_rrr.py              the guided fine-tune — the two loss terms live here
precompute_sky_masks.py      build + screen the mask cache
diagnose_ground_classes.py   which segmenter classes sit on the terrain
validate_sky_mask.py         audit the mask against depth
inspect_dataset.py           manifest + named held-out groups + normalization
interpret_quantify.py        the three headline metrics, per checkpoint
compare_interpret.py         N-way heatmap figure
evaluate_newdata.py          accuracy per experiment / group / condition
make_tradeoff_figure.py      the figure the argument rests on
screen_experiments.py        per-frame image-quality screening
select_experiments.py        coverage-preserving subset selection
prepare_kaggle_dataset.py    24 GB → ~4 GB upload copy
make_notebook.py             generates the Kaggle notebook (edit here, not the .ipynb)

configs/, config_v03.yaml    run configuration
notebooks/                   kaggle_v03_ground_focus.ipynb — the full pipeline
results/                     figures and summary tables
docs/slides/                 the presentation decks
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is not in this repository (~24 GB of PNGs). Place a processed
collection at `data/processed/<experiment_id>/{rgb,depth,samples.csv,samples_full.csv}`
or point `--data-root` at wherever it lives.

## Running the pipeline

```bash
# 1 · manifest and named held-out groups
python inspect_dataset.py --config config_v03.yaml --data-root data/processed \
    --test-group unseen_day_wet=20260812_afternoon_wet_027,... \
    --val-experiments ...

# 2 · which segmenter classes sit on the terrain (look at the crops it writes)
python diagnose_ground_classes.py --data-root data/processed --num 240

# 3 · build and screen the mask cache
python precompute_sky_masks.py --manifest artifacts/manifest_v03/manifest.csv \
    --output-dir artifacts/masks --masker segmentation --size 224 224 \
    --min-ground 0.15 --extra-ground-labels mountain,rock,hill

# 4 · audit it against depth (independent of the segmenter)
python validate_sky_mask.py --data-root data/processed \
    --output-dir artifacts/mask_audit --masker segmentation --qualitative 6

# 5 · base model — no masks, no guidance
python train.py --config config_v03.yaml --balance day_condition

# 6 · the guided fine-tune (and its ablations)
python finetune_rrr.py --config config_v03.yaml \
    --checkpoint artifacts/runs/v03_base/best.pt --sky-mask-dir artifacts/masks \
    --mask-mode nonground --lambda-rrr 1.0 --lambda-spread 0.5 \
    --epochs 3 --target yaw --output artifacts/runs/guided/best.pt

# 7 · measure, compare, plot
python interpret_quantify.py --config config_v03.yaml --checkpoint <ckpt> \
    --sky-mask-dir artifacts/masks --tag guided
python compare_interpret.py --config config_v03.yaml --checkpoints name=path ... \
    --sky-mask-dir artifacts/masks --experiment <exp>
python evaluate_newdata.py --config config_v03.yaml --checkpoints name=path ... \
    --split test --output-dir artifacts/eval_test
python make_tradeoff_figure.py --summary artifacts/v03_summary.csv
```

`notebooks/kaggle_v03_ground_focus.ipynb` runs all of the above end to end on a
Kaggle GPU. Regenerate it with `python make_notebook.py` — do not hand-edit it.

## Guards worth knowing about

These exist because each one has already caught a real mistake:

- **`provides_ground`** — a cache built by the download-free sky detector cannot
  tell grass from a tree, so `--mask-mode nonground` and any `--lambda-spread > 0`
  refuse to run on it. Building a "non-ground" penalty from a sky-only mask
  trains the model to look *away* from the terrain.
- **Mask plausibility** — a frame whose mask claims under 15% ground is a
  segmentation failure, not a real view (on depth-audited frames the ground is
  47% ± 15%). Those frames are re-masked from a white-balanced, CLAHE-equalised
  copy; whatever still fails is marked untrusted in `quality.csv` and contributes
  `has_mask = 0`. It still trains normally — only its guidance is switched off,
  because penalizing "everything except this 4% sliver" is worse than not guiding.
- **Old-checkpoint separation** — a `v03_base` checkpoint from an earlier session
  is not the previous version's model, and the notebook will not score it under
  that name.
- **Normalization per checkpoint** — `evaluate_newdata.py --normalizations` exists
  so a model from an earlier version is scored with its own statistics; scoring it
  with new ones silently rescales its inputs.

## Limitations

Stated in the deck and repeated here:

1. Every wet training row comes from one day, so "wet" and that day cannot be
   told apart by this model.
2. The collection is 82% wet / 18% dry. Dry numbers rest on 966 held-out rows and
   are reported separately, never pooled into a headline.
3. The ground mask still assigns ~2% of the terrain band to `plant`. It was
   deliberately not folded into ground: that class appears just as often at the
   horizon, where it really is vegetation.
4. Three extra epochs is a short fine-tune. Nothing here says the effect survives
   a full retrain from scratch.

## References

- Ross, Hughes & Doshi-Velez, *Right for the Right Reasons*, IJCAI 2017
- Chefer, Schwartz & Wolf, *Optimizing Relevance Maps of Vision Transformers Improves Robustness*, NeurIPS 2022
- Selvaraju et al., *Grad-CAM*, ICCV 2017
- Sundararajan, Taly & Yan, *Axiomatic Attribution for Deep Networks*, ICML 2017
- Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020
- Xie et al., *SegFormer*, NeurIPS 2021 — `nvidia/segformer-b0-finetuned-ade-512-512`

## Status

Internal lab work, not yet released under an open-source licence. Ask before
redistributing.
