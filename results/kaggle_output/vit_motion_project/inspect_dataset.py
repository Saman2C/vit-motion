#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from vit_motion.config import load_config, resolve_from_config
from vit_motion import manifest as manifest_module
from vit_motion.manifest import (
    assign_experiment_splits,
    calculate_normalization,
    discover_experiments,
    process_experiment,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect all processed experiments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", help="Override data.root")
    args = parser.parse_args()

    started = time.perf_counter()
    print("[1/5] Loading configuration...", flush=True)
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = Path(args.data_root).expanduser().resolve() if args.data_root else Path(
        data_cfg["root"]
    ).expanduser().resolve()
    out_dir = resolve_from_config(data_cfg["manifest_dir"], cfg["_config_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"      Data root: {root}", flush=True)

    print("[2/5] Searching for samples.csv files...", flush=True)
    csv_paths = discover_experiments(root)
    if not csv_paths:
        raise SystemExit(f"No samples.csv found below: {root}")
    print(f"      Found {len(csv_paths)} experiment(s).", flush=True)

    # process_experiment() calls manifest.inspect_rgb() once for every image.
    # Wrap that function so the user sees continuous activity even when an
    # experiment is stored on a slow network drive.
    original_inspect_rgb = manifest_module.inspect_rgb
    image_bar = tqdm(
        desc="[3/5] Verifying RGB images",
        unit="image",
        dynamic_ncols=True,
        mininterval=0.2,
    )

    def inspect_rgb_with_progress(path: Path) -> tuple[bool, str]:
        try:
            return original_inspect_rgb(path)
        finally:
            image_bar.update(1)

    manifest_module.inspect_rgb = inspect_rgb_with_progress
    frames, results = [], []
    try:
        experiment_bar = tqdm(
            csv_paths,
            desc="      Experiments",
            unit="experiment",
            dynamic_ncols=True,
            position=1,
        )
        for csv_path in experiment_bar:
            experiment_bar.set_postfix_str(csv_path.parent.name, refresh=True)
            frame, result = process_experiment(csv_path, data_cfg)
            results.append(result)
            if not frame.empty:
                frames.append(frame)
    finally:
        manifest_module.inspect_rgb = original_inspect_rgb
        image_bar.close()

    if not frames:
        errors = "\n".join(f"- {x.experiment_id}: {x.error}" for x in results)
        raise SystemExit(f"No valid samples found.\n{errors}")

    print("[4/5] Building manifest and normalization statistics...", flush=True)
    manifest = pd.concat(frames, ignore_index=True)
    ratios = tuple(float(x) for x in data_cfg["split_ratios"])
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("data.split_ratios must sum to 1.")
    split_map = assign_experiment_splits(
        manifest["experiment_id"].unique(), ratios, int(cfg["seed"])
    )
    manifest["split"] = manifest["experiment_id"].map(split_map)
    train = manifest[manifest["split"] == "train"]
    normalization = calculate_normalization(train)

    print("[5/5] Writing reports...", flush=True)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    write_json(out_dir / "normalization.json", normalization)
    write_json(out_dir / "splits.json", split_map)

    report = {
        "data_root": str(root),
        "experiments_found": len(csv_paths),
        "experiments_usable": sum(x.error is None and x.valid_rows > 0 for x in results),
        "total_rows": sum(x.total_rows for x in results),
        "valid_rows": len(manifest),
        "split_rows": manifest["split"].value_counts().to_dict(),
        "split_experiments": pd.Series(split_map).value_counts().to_dict(),
        "experiments": [x.__dict__ for x in results],
        "warnings": (
            ["Only one usable experiment: validation/test loaders cannot be created."]
            if len(split_map) == 1
            else []
        ),
    }
    write_json(out_dir / "quality_report.json", report)

    md = [
        "# Dataset quality report",
        "",
        f"- Data root: `{root}`",
        f"- Experiments found: {report['experiments_found']}",
        f"- Total CSV rows: {report['total_rows']}",
        f"- Valid rows after filtering: {report['valid_rows']}",
        f"- Split rows: `{report['split_rows']}`",
        "",
        "| Experiment | Total | Valid | Rejections | Error |",
        "|---|---:|---:|---|---|",
    ]
    for x in results:
        md.append(
            f"| {x.experiment_id} | {x.total_rows} | {x.valid_rows} | "
            f"`{json.dumps(x.rejection_counts, ensure_ascii=False)}` | {x.error or ''} |"
        )
    if report["warnings"]:
        md.extend(["", "## Warnings", *[f"- {x}" for x in report["warnings"]]])
    (out_dir / "quality_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - started
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"PASS: inspection completed in {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
