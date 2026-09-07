#!/usr/bin/env python3
"""Build the manifest / normalization / quality report for a tree of experiments.

Two additions over the original version, both needed to bring a *new collection*
in as a held-out test set without disturbing the trained model:

``--data-root A B``
    Several collection trees in one manifest.  The 2026-08 "clean" collection is 82% wet;
    the older one is 100% dry.  Neither alone gives a model that has seen both conditions,
    so v0.3 trains on both.  List the preferred tree FIRST — a run that appears in more than
    one tree is taken from the first one under ``--on-duplicate first``.

``--on-duplicate {error,first}``
    Nine runs were re-exported into the clean collection and still exist in the old one.
    Counting them twice would silently double their weight in training, so the default is to
    stop; ``first`` keeps the copy from the earliest-listed root and prints what it dropped.

``--test-experiments`` / ``--val-experiments``
    Name the held-out runs instead of hashing them.  Robustness is a claim about
    *which* runs were held out, and a hash cannot express that.  Everything not
    named goes to train.

``--test-group NAME=exp1,exp2``  (repeatable)
    Several held-out sets that are all ``split == test`` but are reported apart,
    recorded in a ``split_group`` column.  "An unseen DAY" and "unseen RUNS from
    days the model trained on" fail for different reasons; pooling them into one
    number hides which one happened.

``--split-override test``
    Put every experiment in one split (for a manifest that exists only to be
    evaluated).

``--normalization-from artifacts/manifest/normalization.json``
    Reuse the statistics a model was trained with instead of recomputing them.
    Recomputing would silently rescale the inputs and make the reported accuracy
    incomparable with that model's earlier numbers.
"""
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
    validate_normalization_schema,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect all processed experiments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", nargs="+",
                        help="One or more collection roots (overrides data.root).")
    parser.add_argument("--test-experiments", default="",
                        help="Comma-separated experiment ids to force into the test split.")
    parser.add_argument("--val-experiments", default="",
                        help="Comma-separated experiment ids to force into the val split.")
    parser.add_argument("--on-duplicate", choices=["error", "first"], default="error",
                        help="What to do when the same experiment folder name appears in two "
                             "roots. 'first' keeps the copy from the earliest-listed root.")
    parser.add_argument("--test-group", action="append", default=[], metavar="NAME=ids",
                        help="A named test set, e.g. unseen_day=a,b. Repeatable; all land in "
                             "split=test and are told apart by the split_group column.")
    parser.add_argument("--split-override", choices=["train", "val", "test"],
                        help="Put every experiment in this split (held-out collections).")
    parser.add_argument("--normalization-from",
                        help="Copy this normalization.json instead of recomputing it.")
    args = parser.parse_args()

    started = time.perf_counter()
    print("[1/5] Loading configuration...", flush=True)
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    roots = ([Path(x).expanduser().resolve() for x in args.data_root] if args.data_root
             else [Path(data_cfg["root"]).expanduser().resolve()])
    out_dir = resolve_from_config(data_cfg["manifest_dir"], cfg["_config_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for root in roots:
        print(f"      Data root: {root}", flush=True)

    print("[2/5] Searching for samples.csv files...", flush=True)
    csv_paths: list[Path] = []
    for root in roots:
        found = discover_experiments(root)
        print(f"      {len(found)} experiment(s) under {root}", flush=True)
        csv_paths.extend(found)
    if not csv_paths:
        raise SystemExit("No samples.csv found below: " + ", ".join(str(r) for r in roots))
    names = [p.parent.name for p in csv_paths]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates and args.on_duplicate == "error":
        raise SystemExit(
            f"{len(duplicates)} experiment folder name(s) appear in more than one root: "
            + ", ".join(duplicates[:10]) + ("..." if len(duplicates) > 10 else "")
            + "\nCounting a run twice doubles its weight in training. Re-run with "
              "--on-duplicate first, listing the preferred root first.")
    if duplicates:
        kept, dropped = [], []
        seen: set[str] = set()
        for path in csv_paths:               # csv_paths is already in root order
            if path.parent.name in seen:
                dropped.append(path)
            else:
                seen.add(path.parent.name)
                kept.append(path)
        csv_paths = kept
        print(f"      {len(dropped)} duplicate experiment(s) dropped, keeping the copy from "
              f"the first root that had them:", flush=True)
        for path in dropped[:12]:
            print(f"        {path.parent.name}  <- dropped {path.parent}", flush=True)
        if len(dropped) > 12:
            print(f"        ... and {len(dropped) - 12} more", flush=True)
    print(f"      Found {len(csv_paths)} experiment(s) in total.", flush=True)

    # process_experiment() calls manifest.inspect_rgb() once for every image.
    # Wrap that function so the user sees continuous activity even when an
    # experiment is stored on a slow network drive.
    original_inspect_rgb = manifest_module.inspect_rgb
    image_bar = tqdm(desc="[3/5] Verifying RGB images", unit="image",
                     dynamic_ncols=True, mininterval=0.2)

    def inspect_rgb_with_progress(path: Path) -> tuple[bool, str]:
        try:
            return original_inspect_rgb(path)
        finally:
            image_bar.update(1)

    manifest_module.inspect_rgb = inspect_rgb_with_progress
    frames, results = [], []
    try:
        experiment_bar = tqdm(csv_paths, desc="      Experiments", unit="experiment",
                              dynamic_ncols=True, position=1)
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
    available = set(manifest["experiment_id"].astype(str).unique())
    named_test = [x.strip() for x in args.test_experiments.split(",") if x.strip()]
    named_val = [x.strip() for x in args.val_experiments.split(",") if x.strip()]

    group_map: dict[str, str] = {}
    for item in args.test_group:
        if "=" not in item:
            raise SystemExit(f"--test-group expects NAME=id1,id2 — got: {item}")
        name, ids = item.split("=", 1)
        members = [x.strip() for x in ids.split(",") if x.strip()]
        if not members:
            raise SystemExit(f"--test-group {name!r} names no experiments.")
        for member in members:
            if member in group_map and group_map[member] != name:
                raise SystemExit(
                    f"{member} is in two test groups ({group_map[member]} and {name}).")
            group_map[member] = name
        named_test.extend(members)
    named_test = sorted(set(named_test))

    overlap = sorted(set(named_test) & set(named_val))
    if overlap:
        raise SystemExit("These runs are named for both test and val: " + ", ".join(overlap))
    unknown = sorted(set(named_test + named_val) - available)
    if unknown:
        raise SystemExit("These experiment ids are not in the data: " + ", ".join(unknown)
                         + "\nAvailable: " + ", ".join(sorted(available)))
    if args.split_override:
        split_map = {x: args.split_override for x in manifest["experiment_id"].unique()}
    elif named_test or named_val:
        split_map = {x: "train" for x in available}
        for x in named_val:
            split_map[x] = "val"
        for x in named_test:
            split_map[x] = "test"
    else:
        ratios = tuple(float(x) for x in data_cfg["split_ratios"])
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("data.split_ratios must sum to 1.")
        split_map = assign_experiment_splits(
            manifest["experiment_id"].unique(), ratios, int(cfg["seed"])
        )
    manifest["split"] = manifest["experiment_id"].map(split_map)
    manifest["split_group"] = (manifest["experiment_id"].map(group_map)
                               .fillna(manifest["split"]))

    if args.normalization_from:
        source = Path(args.normalization_from).expanduser().resolve()
        normalization = json.loads(source.read_text(encoding="utf-8"))
        validate_normalization_schema(normalization)
        print(f"      Reusing normalization from {source}")
    else:
        train = manifest[manifest["split"] == "train"]
        if train.empty:
            raise SystemExit(
                "No training rows, so normalization cannot be computed. Pass "
                "--normalization-from <path to the trained model's normalization.json>."
            )
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
        "split_override": args.split_override,
        "duplicates_dropped": sorted(duplicates) if duplicates else [],
        "split_groups": {
            name: sorted(e for e, g in group_map.items() if g == name)
            for name in sorted(set(group_map.values()))
        },
        "group_rows": manifest.groupby("split_group").size().to_dict(),
        "normalization_source": args.normalization_from or "computed from train split",
        "experiments": [x.__dict__ for x in results],
        "warnings": (
            ["Only one usable experiment: validation/test loaders cannot be created."]
            if len(split_map) == 1 else []
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
        f"- Test groups: `{report['group_rows']}`",
        f"- Normalization: {report['normalization_source']}",
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
    print(json.dumps({k: v for k, v in report.items() if k != "experiments"}, indent=2,
                     ensure_ascii=False))
    print(f"PASS: inspection completed in {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
