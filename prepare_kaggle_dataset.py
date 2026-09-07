#!/usr/bin/env python3
"""Build an upload-ready copy of a collection: small enough to actually transfer.

The full collection is ~24 GB (18 GB RGB + 5.7 GB depth at 640x480). Nothing in
this pipeline ever sees a 640x480 frame:

* the model's transform is ``Resize((224, 224))`` — it squashes 4:3 to square,
  so pre-resizing to a square applies exactly the same distortion, earlier;
* the mask cache is written at 224x224;
* the depth audit compares masks, at whatever size both share.

So a 256x256 copy (a little headroom over 224) carries the same information the
run can use, at roughly a fifth of the bytes. Depth is kept only for the
experiments the audit actually needs — it samples a dozen frames per run, and
5.7 GB of it exists to answer a question a few hundred frames answer.

    python prepare_kaggle_dataset.py --data-root data/processed \
        --output-dir data/kaggle_upload --size 256 --depth-experiments 8

Writes the same tree (``<exp>/rgb``, ``<exp>/depth``, both CSVs, summary.json),
so every script points at it unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image


def resize_rgb(src: Path, dst: Path, size: int) -> None:
    with Image.open(src) as img:
        img.convert("RGB").resize((size, size), Image.BILINEAR).save(dst, optimize=True)


def resize_depth(src: Path, dst: Path, size: int) -> None:
    # NEAREST only: averaging two depths across an edge invents a surface that is
    # at neither distance, and the audit's "is this beyond range" test would then
    # be answered by a pixel that does not exist.
    with Image.open(src) as img:
        array = np.asarray(img)
        out = Image.fromarray(array).resize((size, size), Image.NEAREST)
        out.save(dst, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--depth-experiments", type=int, default=8,
                    help="Keep depth for this many experiments, spread across the "
                         "collection (0 = none, -1 = all).")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    experiments = sorted(p.parent for p in root.glob("*/samples.csv"))
    if not experiments:
        raise SystemExit(f"No experiments under {root}")

    if args.depth_experiments < 0:
        keep_depth = set(e.name for e in experiments)
    elif args.depth_experiments == 0:
        keep_depth = set()
    else:
        picks = np.linspace(0, len(experiments) - 1,
                            min(args.depth_experiments, len(experiments))).round().astype(int)
        keep_depth = {experiments[i].name for i in sorted(set(picks.tolist()))}
    print(f"{len(experiments)} experiments · depth kept for {len(keep_depth)}: "
          f"{', '.join(sorted(keep_depth)) or '(none)'}")

    started = time.perf_counter()
    n_rgb = n_depth = 0
    bytes_in = bytes_out = 0
    for index, exp in enumerate(experiments, 1):
        dest = out_root / exp.name
        (dest / "rgb").mkdir(parents=True, exist_ok=True)
        for name in ("samples.csv", "samples_full.csv", "summary.json"):
            if (exp / name).is_file():
                shutil.copy2(exp / name, dest / name)

        for src in sorted((exp / "rgb").glob("*.png")):
            target = dest / "rgb" / src.name
            if args.skip_existing and target.is_file():
                continue
            bytes_in += src.stat().st_size
            resize_rgb(src, target, args.size)
            bytes_out += target.stat().st_size
            n_rgb += 1

        if exp.name in keep_depth and (exp / "depth").is_dir():
            (dest / "depth").mkdir(parents=True, exist_ok=True)
            for src in sorted((exp / "depth").glob("*.png")):
                target = dest / "depth" / src.name
                if args.skip_existing and target.is_file():
                    continue
                bytes_in += src.stat().st_size
                resize_depth(src, target, args.size)
                bytes_out += target.stat().st_size
                n_depth += 1

        print(f"  [{index}/{len(experiments)}] {exp.name}  "
              f"({n_rgb} rgb, {n_depth} depth, {bytes_out / 1e9:.2f} GB out)", flush=True)

    manifest = {
        "source": str(root),
        "size": args.size,
        "experiments": len(experiments),
        "rgb_frames": n_rgb,
        "depth_frames": n_depth,
        "depth_experiments": sorted(keep_depth),
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "seconds": round(time.perf_counter() - started, 1),
        "note": ("RGB bilinear to a square, matching the model's own Resize((224,224)); "
                 "depth NEAREST so no invented distances. Depth kept only where the audit "
                 "needs it."),
    }
    (out_root / "prepared.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "depth_experiments"}, indent=2))
    print(f"\n{bytes_in / 1e9:.1f} GB -> {bytes_out / 1e9:.1f} GB "
          f"({bytes_out / max(bytes_in, 1):.0%})\nReady to upload: {out_root}")


if __name__ == "__main__":
    main()
