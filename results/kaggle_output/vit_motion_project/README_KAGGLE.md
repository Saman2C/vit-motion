# Running ViT-Motion on Kaggle

This setup runs your existing `vit_motion` project on Kaggle — **training,
evaluation, and interpretability** — with the dataset served as a persistent
Kaggle Dataset so you never re-download the 18.4 GB from Hugging Face.

## What's in this bundle

| File | Purpose |
|---|---|
| `vit_motion_kaggle.zip` | Your full project + new interpretability code + `config_kaggle.yaml`. Upload as a **Kaggle Dataset** (this is the "code" input). |
| `kaggle_build_dataset.ipynb` | Run **once** to pull the HF dataset into Kaggle and save it as a Kaggle Dataset. |
| `kaggle_run.ipynb` | The main notebook: manifest → smoke test → train → evaluate → interpret. |

New source files added to the project (also written back into your local project):
`vit_motion/interpret.py`, `interpret_experiment.py`, `config_kaggle.yaml`.

## One-time setup

### 1. Upload the code
Kaggle → **Datasets → New Dataset** → upload `vit_motion_kaggle.zip` (Kaggle
unzips it). Name it e.g. `vit-motion-code`.

### 2. Build the data dataset
Open `kaggle_build_dataset.ipynb` as a new Kaggle notebook, set **Internet = ON**,
Run All. It downloads only `samples.csv` + `rgb/` (skips `depth/` to save space),
then follow the in-notebook step to **Save Version → New Dataset from output**
(name it e.g. `vit-motion-dataset`).

## Every run

Open `kaggle_run.ipynb`, and on the right panel:
- **Add Input** → your code dataset **and** your data dataset.
- **Accelerator = GPU** (T4 ×2 or P100).
- **Internet = ON** (first run downloads the pretrained ViT encoder weights).

Run All. The notebook:
1. Copies the code into `/kaggle/working` (writable — artifacts land there).
2. Auto-detects the dataset root (the exact dataset slug doesn't matter).
3. Builds `artifacts/manifest/` (manifest.csv, normalization.json, splits.json).
4. Runs the smoke test, then `train.py`.
5. Evaluates one experiment and generates interpretability figures.
6. Zips `artifacts/` to `vit_motion_artifacts.zip` for download.

## Notes that matter

- **Paths.** Your configs resolve `manifest_dir` / `output_dir` *relative to the
  config file*. Because the code is copied into `/kaggle/working`, all artifacts
  are writable. The read-only `/kaggle/input` is never written to.
- **`--data-root`.** Only `inspect_dataset.py` reads `data.root`; the notebook
  overrides it with the auto-detected Kaggle path, so the Windows UNC path in
  your configs is ignored. After the manifest is built, `train`/`evaluate`/
  `interpret` use the absolute RGB paths stored inside `manifest.csv`.
- **Session limits.** A full 100-epoch run may exceed one Kaggle session. Set
  `EPOCHS` lower for a first pass, or resume with
  `--resume artifacts/runs/vit_motion_temporal_cr_v0_2_1/last.pt`. Download
  `vit_motion_artifacts.zip` (which contains `best.pt`/`last.pt`) between sessions.
- **Interpretability.** `interpret_experiment.py --target {dx,dy,yaw,norm}`
  produces a 4-panel figure per sample: cached RGB, Grad-CAM overlay, fusion-token
  importance (image vs. the 6 numeric steps, with modality share), and the
  Integrated-Gradients numeric heatmap (6 steps × 5 channels). Because the visual
  encoder is average-pooled before fusion, "image vs numeric" comes from the
  temporal attention / IG, while *where in the image* comes from Grad-CAM — they're
  complementary.

## Methods behind the interpretability module

- Grad-CAM — Selvaraju et al., 2017 (ViT token-grid reshape).
- Integrated Gradients — Sundararajan et al., 2017 (joint image + numeric attribution).
- Attention Rollout — Abnar & Zuidema, 2020 (fusion-encoder token importance).

Depends only on `torch` / `numpy` / `matplotlib` — nothing added to `requirements.txt`.
