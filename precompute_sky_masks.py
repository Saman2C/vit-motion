#!/usr/bin/env python3
"""Pre-compute a per-frame region mask (sky / ground / other) for every image.

The RRR fine-tune and the saliency quantification both need to know, per frame,
which pixels are sky.  Running a segmenter inside the training loop would make
every epoch several times slower for a quantity that never changes, so the masks
are computed once here and cached as tiny 8-bit PNGs of region codes
(0 = sky, 1 = ground, 2 = other) at the model's input geometry.

Part of the collection is washed out and colour-cast, and the segmenter reads the
pale terrain there as anything but ground — 4-8% ground on frames whose lower half
is plainly grass.  Two defences, both recorded:

* any frame whose ground area is below ``--min-ground`` is masked a second time
  from a white-balanced, contrast-equalised copy, and the better of the two is
  kept (the model still sees the untouched frame; only the annotation changes);
* whatever is still implausible is written to ``quality.csv`` as untrusted, and
  the fine-tune and the metrics skip guiding on it.  Penalizing "everything but
  this 4% sliver" is worse than not guiding that frame at all.

Cache layout mirrors the dataset:

    <output-dir>/<experiment_id>/<rgb-stem>.png
    <output-dir>/index.json          masker, size, per-experiment area statistics
    <output-dir>/quality.csv         per-frame areas + the trusted flag

Two input modes:

  --manifest artifacts/manifest/manifest.csv    the training manifest
  --data-root new_data/processed                a raw processed-experiments tree

Example:
    python precompute_sky_masks.py --manifest artifacts/manifest/manifest.csv \
        --output-dir artifacts/sky_masks --masker auto --size 224 224
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from vit_motion.sky_mask import (
    GROUND_LABELS,
    MASK_QUALITY_FILE,
    build_masker,
    enhance_for_segmentation,
    mask_cache_path,
    mask_is_plausible,
    write_mask_png,
)


def rows_from_manifest(path: Path, splits: list[str] | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if splits and "split" in frame.columns:
        frame = frame[frame["split"].isin(splits)]
    frame = frame[["experiment_id", "rgb_abs_path"]].drop_duplicates("rgb_abs_path")
    return frame.reset_index(drop=True)


def rows_from_data_root(root: Path) -> pd.DataFrame:
    records = []
    for csv_path in sorted(root.rglob("samples.csv")):
        exp_dir = csv_path.parent
        frame = pd.read_csv(csv_path)
        column = "rgb_path" if "rgb_path" in frame.columns else frame.columns[0]
        for value in frame[column].astype(str).unique():
            records.append({"experiment_id": exp_dir.name,
                            "rgb_abs_path": str((exp_dir / value).resolve())})
    return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="artifacts/manifest/manifest.csv")
    source.add_argument("--data-root", help="A tree of processed experiment folders.")
    ap.add_argument("--output-dir", default="artifacts/sky_masks")
    ap.add_argument("--splits", default="", help="Comma list, e.g. train,val,test (manifest mode).")
    ap.add_argument("--masker", default="auto", choices=["auto", "segmentation", "energy"])
    ap.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    ap.add_argument("--size", type=int, nargs=2, default=(224, 224),
                    help="Cache geometry (H W) — match data.image_size.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Debug: stop after N images.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--min-ground", type=float, default=0.15,
                    help="Ground area below this is treated as a segmentation failure: the "
                         "frame is retried enhanced, then marked untrusted if it still fails.")
    ap.add_argument("--no-enhance-retry", action="store_true",
                    help="Do not retry implausible frames on an enhanced copy.")
    ap.add_argument("--extra-ground-labels", default="",
                    help="Comma list of segmenter class names to COUNT AS GROUND on top of "
                         "the built-in list, e.g. 'dirt,mud,rock'. Run "
                         "diagnose_ground_classes.py first — add a class because the "
                         "diagnosis showed it sitting on the terrain, not because it sounds "
                         "like terrain. The final list is recorded in index.json.")
    args = ap.parse_args()

    extra = {s.strip().lower() for s in args.extra_ground_labels.split(",") if s.strip()}
    ground_labels = frozenset(GROUND_LABELS | extra)
    if extra:
        print(f"      ground list extended with: {', '.join(sorted(extra))}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (int(args.size[0]), int(args.size[1]))

    if args.manifest:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        frame = rows_from_manifest(Path(args.manifest).expanduser().resolve(), splits or None)
    else:
        frame = rows_from_data_root(Path(args.data_root).expanduser().resolve())
    if args.limit:
        frame = frame.head(args.limit)
    if frame.empty:
        raise SystemExit("No images to mask.")
    print(f"[1/3] {len(frame)} unique frames from "
          f"{frame['experiment_id'].nunique()} experiments")

    print(f"[2/3] Building the '{args.masker}' masker")
    masker = build_masker(args.masker, model_name=args.model,
                          batch_size=args.batch_size, ground_labels=ground_labels)
    masker_name = type(masker).__name__
    print(f"      using {masker_name}, caching at {size[0]}x{size[1]}")

    print("[3/3] Masking")
    started = time.perf_counter()
    areas: list[dict] = []
    provides_ground = True
    written = skipped = rescued = untrusted = 0
    batch_rows: list[tuple[str, Path, Path]] = []

    def flush() -> None:
        nonlocal written, provides_ground, rescued, untrusted
        if not batch_rows:
            return
        images = []
        for _, rgb_path, _ in batch_rows:
            with Image.open(rgb_path) as img:
                images.append(img.convert("RGB").copy())
        results = list(masker.masks(images, size=size))

        # Second pass, only for the frames the segmenter clearly failed on.
        if not args.no_enhance_retry:
            retry = [i for i, m in enumerate(results)
                     if m.provides_ground and not mask_is_plausible(m, args.min_ground)]
            if retry:
                repaired = masker.masks([enhance_for_segmentation(images[i]) for i in retry],
                                        size=size)
                for i, better in zip(retry, repaired):
                    if better.ground.mean() > results[i].ground.mean():
                        results[i] = better
                        if mask_is_plausible(better, args.min_ground):
                            rescued += 1

        for (exp, rgb_path, dest), masks in zip(batch_rows, results):
            write_mask_png(dest, masks)
            trusted = mask_is_plausible(masks, args.min_ground)
            untrusted += (not trusted)
            areas.append({"experiment_id": exp, "stem": Path(rgb_path).stem,
                          "trusted": int(trusted), **masks.area_fractions()})
            # One False anywhere disqualifies the whole cache: a consumer that
            # penalizes "non-ground" must not be fed a mix of real ground masks
            # and sky-only ones.
            provides_ground = provides_ground and bool(masks.provides_ground)
            written += 1
        batch_rows.clear()

    for i, row in enumerate(frame.itertuples(index=False)):
        rgb_path = Path(row.rgb_abs_path)
        dest = mask_cache_path(out_dir, row.experiment_id, rgb_path)
        if args.skip_existing and dest.is_file():
            skipped += 1
            continue
        if not rgb_path.is_file():
            print(f"      missing image, skipped: {rgb_path}")
            continue
        batch_rows.append((str(row.experiment_id), rgb_path, dest))
        if len(batch_rows) >= args.batch_size:
            flush()
        if (i + 1) % 500 == 0:
            rate = written / max(time.perf_counter() - started, 1e-6)
            print(f"      {i + 1}/{len(frame)}  ({rate:.1f} img/s)", flush=True)
    flush()

    stats = pd.DataFrame(areas)
    index = {
        "masker": masker_name,
        "model": args.model if masker_name == "SegformerSkyMasker" else None,
        "size": list(size),
        "n_written": int(written),
        "n_skipped_existing": int(skipped),
        "seconds": round(time.perf_counter() - started, 1),
        "region_codes": {"sky": 0, "ground": 1, "other": 2},
        # Whether the GROUND channel is real. False means the masker could only
        # find the sky, so every non-sky pixel (grass, trees, buildings alike)
        # sits in `other` and no "non-ground" penalty may be built from it.
        "provides_ground": bool(provides_ground and written > 0),
        "ground_labels": sorted(ground_labels),
        "min_ground": float(args.min_ground),
        "enhance_retry": not args.no_enhance_retry,
        "n_rescued_by_enhancement": int(rescued),
        "n_untrusted": int(untrusted),
    }
    if not stats.empty:
        stats.to_csv(out_dir / MASK_QUALITY_FILE, index=False)
        index["untrusted_fraction"] = float(1.0 - stats["trusted"].mean())
        worst = (stats.groupby("experiment_id")["trusted"].mean().sort_values().head(15))
        index["worst_experiments_trusted_fraction"] = worst.round(3).to_dict()
        index["per_experiment_trusted"] = (
            stats.groupby("experiment_id")["trusted"].mean().round(3).to_dict())
        index["area_mean"] = {k: float(stats[k].mean()) for k in ("sky", "ground", "other")}
        index["area_std"] = {k: float(stats[k].std()) for k in ("sky", "ground", "other")}
        index["per_experiment_area"] = (
            stats.groupby("experiment_id")[["sky", "ground", "other"]].mean().round(4)
            .to_dict(orient="index")
        )
        index["area_mean_trusted"] = {
            k: float(stats.loc[stats["trusted"] == 1, k].mean()) for k in ("sky", "ground", "other")
        }
        index["frames_without_sky"] = int((stats["sky"] < 0.005).sum())
        index["frames_without_ground"] = int((stats["ground"] < 0.005).sum())
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    hide = ("per_experiment_area", "ground_labels", "per_experiment_trusted")
    print(json.dumps({k: v for k, v in index.items() if k not in hide}, indent=2))
    if not index["provides_ground"]:
        print("\nWARNING: this cache has NO usable ground channel — it can only support\n"
              "         --mask-mode sky. Rebuild with --masker segmentation for the\n"
              "         non-ground penalty and the ground-spread term.")
    print(f"Saved masks to: {out_dir}")


if __name__ == "__main__":
    main()
