#!/usr/bin/env python3
"""Generate kaggle_v03_ground_focus.ipynb (a script, so the notebook is reviewable)."""
import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md("""
# ViT-Motion v0.3 — make the model use the ground, and all of it

The model predicts the next motion step from one cached RGB frame plus six numeric steps.
The cue it *should* use is the surface the vehicle is driving over — mud, dry dirt, grass,
puddles. Trees, buildings, parked cars, tents and sky carry nothing about egomotion.

Three things change from the version in the deck.

**1. The region is the ground, not the sky.** v1 penalized "the top half of the frame" as a
stand-in for useless. Audited against depth, that region is 29% sky, 51% distant structure,
17% mid-range and 3% near ground — and the sky is 14.6% of a frame, not 50%. Here the
penalty region is `1 - ground`, from a semantic segmentation mask, and the code refuses to
run it on masks that have no real ground channel.

**2. A spread term.** v1 drove sky saliency 45% → 0% and left the attention on two or three
hot spots on the grass. The objective never objected: it only said *don't be outside*, never
*be spread out inside*. v0.3 adds `spread = 1 - H(saliency | ground) / log(N_ground)` and
reports `coverage = exp(H)/N`, the effective share of the terrain the attention covers.

**3. One collection, uniform quality.** 63 runs / ~31.9k rows over four days, every run with
aligned depth. The older collection was tried and dropped: its frames are washed out and
colour-cast, and the segmenter reads their pale terrain as anything but ground — 4-8% ground
where half the image is grass. Mixing it in would have made image quality a confound in every
comparison, and would have left the whole result leaning on the repair machinery.

The price is honest and stated up front: **the collection is 82% wet / 18% dry**, so the dry
numbers rest on 5.9k rows and are reported separately, never pooled into one headline. Wet
runs are *not* pruned to force a balance — run 1 showed the accuracy gain came from data
volume (wet yaw R² 0.753 → 0.829 over v0.2.1), and deleting runs to reach a ratio would throw
that away. Training uses `--balance day_condition` instead, which reweights an epoch without
destroying anything.

Held-out groups, reported apart, because they fail for different reasons:

| group | what it tests |
|---|---|
| `unseen_day_wet` | a whole wet collection day the model never saw |
| `unseen_runs_dry` | dry runs it never saw, from days it did train on |
| `unseen_runs_wet` | the same for wet |

A whole *dry* day is not held out: the only pure dry day is 75% of all the dry data, and
removing it would leave 1.5k dry rows to train on.

**Attach:** the code dataset · the dataset (`processed/`) · optionally the artifacts dataset,
to score the old v0.2.1 checkpoint alongside.  **Settings:** GPU on, Internet on.
"""),

code("""
# --- 1) stage the code, install deps ------------------------------------------------------
# Kaggle mounts inputs either flat (/kaggle/input/<slug>) or nested
# (/kaggle/input/datasets/<slug>), and a dataset still being attached shows up empty.
# This survives both and says which one it hit.
import os, sys, glob, json, shutil, pathlib, subprocess, collections

def show_inputs(depth=2):
    for base in sorted(glob.glob('/kaggle/input/*')):
        print(' ', base)
        if depth > 1:
            for sub in sorted(glob.glob(base + '/*'))[:20]:
                n = len(glob.glob(sub + '/**/*', recursive=True))
                print('     ', os.path.basename(sub), f'({n} entries)')

print('Attached inputs:')
show_inputs()

codes = glob.glob('/kaggle/input/**/precompute_sky_masks.py', recursive=True)
if not codes:
    found = sorted({os.path.basename(p) for p in
                    glob.glob('/kaggle/input/**/*.py', recursive=True)})[:40]
    raise SystemExit(
        'precompute_sky_masks.py is not in any attached dataset.\\n'
        'Python files that ARE attached: ' + (', '.join(found) if found else '(none)') + '\\n\\n'
        'Most likely the CODE dataset is an older upload. Put vit_motion_v03_code.zip in it, '
        'wait for "Adding data..." to finish, then re-run this cell.')

CODE_SRC = str(pathlib.Path(sorted(codes, key=len)[0]).parent)
WORK = '/kaggle/working/vit_motion_project'
if os.path.exists(WORK):
    shutil.rmtree(WORK)
shutil.copytree(CODE_SRC, WORK)
os.chdir(WORK); sys.path.insert(0, WORK)
print('\\ncode ->', WORK)
for required in ['finetune_rrr.py', 'interpret_quantify.py', 'config_v03.yaml',
                 'vit_motion/attention_focus.py', 'vit_motion/sky_mask.py']:
    assert os.path.exists(os.path.join(WORK, required)), (
        f'{required} missing from the code dataset — it is an older upload. '
        f'Replace it with vit_motion_v03_code.zip.')
print('code dataset is the v0.3 one.')

subprocess.run([sys.executable, '-m', 'pip', '-q', 'install',
                'timm>=1.0', 'transformers>=4.40', 'opencv-python-headless>=4.9'], check=False)
"""),

code("""
# --- 2) locate the collections (and the old checkpoint, if attached) ---------------------
counts = collections.Counter(
    str(pathlib.Path(p).parent.parent)
    for p in glob.glob('/kaggle/input/**/samples.csv', recursive=True))
assert counts, 'no samples.csv anywhere — is the dataset still attaching?'

roots = []
for root, n in counts.most_common():
    with_depth = len(glob.glob(f'{root}/*/depth'))
    roots.append({'root': root, 'experiments': n, 'with_depth': with_depth})
roots.sort(key=lambda r: (-r['with_depth'], -r['experiments']))   # depth-bearing root first

print('roots holding experiment folders:')
for r in roots:
    print(f"   {r['experiments']:3d} experiments  ({r['with_depth']} with depth/)  {r['root']}")

DATA_ROOTS = [r['root'] for r in roots]
DEPTH_ROOT = next((r['root'] for r in roots if r['with_depth'] > 0), None)
assert DEPTH_ROOT, 'no collection with depth/ — the mask audit needs one'

ckpts = sorted(glob.glob('/kaggle/input/**/best.pt', recursive=True), key=len)

# A v0.3 base checkpoint from an earlier session is NOT the old v0.2.1 model. Left
# unseparated it would be scored as 'v021_old' and the deck would carry a v0.3
# model under the previous version's name.
def _is_v03_base(p):
    return 'v03_base' in p.replace('\\\\', '/').split('/')

REUSE_BASE = next((p for p in ckpts if _is_v03_base(p)), None)
old_ckpts = [p for p in ckpts if not _is_v03_base(p)]
old_norms = [p for p in sorted(glob.glob('/kaggle/input/**/normalization.json',
                                         recursive=True), key=len)
             if 'v03_base' not in p.replace('\\\\', '/')]
OLD_CKPT = old_ckpts[0] if old_ckpts else None
OLD_NORM = old_norms[0] if old_norms else None
print('REUSE_BASE', REUSE_BASE or '(none — the base model will be trained here)')

print('\\nDATA_ROOTS (preferred first):')
for r in DATA_ROOTS:
    print('  ', r)
print('DEPTH_ROOT', DEPTH_ROOT)
print('OLD_CKPT  ', OLD_CKPT)
print('OLD_NORM  ', OLD_NORM)
if len(DATA_ROOTS) == 1:
    print('\\nWARNING: only one collection is attached. If it is the wet-heavy clean one, the '
          'model will train on ~16% dry data — attach the older dry collection too.')
if OLD_CKPT and not OLD_NORM:
    print('\\nNOTE: the old checkpoint has no normalization.json beside it. It will be SKIPPED '
          'in the accuracy table — scoring it with v0.3 statistics would silently rescale its '
          'inputs and the comparison would be meaningless.')
"""),

md("""
## Step 1 — the split, named run by run

A hash cannot express "a whole unseen wet day, and separately, unseen runs from days we did
train on". Those fail differently, so they are named groups here and separate columns in
every table later. Everything not named goes to train.

Two rules the cell enforces, both learned the hard way:

* a held-out **day** only counts if every run that day is one condition — a day with both
  would leave the model having trained on that site and light anyway;
* the **scarce** condition never gives up a whole day. Here dry is 18% of the rows and its
  only pure day holds 75% of them, so dry contributes held-out *runs* instead, and the cell
  refuses to leave less than `MIN_SCARCE_IN_TRAIN` of the scarce rows in training.
"""),

code("""
exp_ids, seen_root = [], {}
for root in DATA_ROOTS:                       # preferred root first; duplicates keep it
    for p in sorted(glob.glob(f'{root}/*/samples.csv')):
        name = pathlib.Path(p).parent.name
        if name not in seen_root:
            seen_root[name] = root
            exp_ids.append(name)
exp_ids.sort()

def day_of(e):  return e.split('_', 1)[0]
def cond_of(e): return 'wet' if 'wet' in e else 'dry'

rows_of = {}
for e in exp_ids:
    with open(f'{seen_root[e]}/{e}/samples.csv') as handle:
        rows_of[e] = max(sum(1 for _ in handle) - 1, 0)

groups = collections.defaultdict(list)
for e in exp_ids:
    groups[(day_of(e), cond_of(e))].append(e)
print(f'{len(exp_ids)} experiments, {sum(rows_of.values())} rows')
for key in sorted(groups):
    print(f'   {key[0]} {key[1]:3s}  {len(groups[key]):3d} runs  {sum(rows_of[e] for e in groups[key]):6d} rows')

cond_rows = {c: sum(rows_of[e] for e in exp_ids if cond_of(e) == c) for c in ('wet', 'dry')}
ABUNDANT = max(cond_rows, key=cond_rows.get)
SCARCE = 'dry' if ABUNDANT == 'wet' else 'wet'
print(f'\\nabundant = {ABUNDANT} ({cond_rows[ABUNDANT]} rows, {cond_rows[ABUNDANT] / sum(cond_rows.values()):.0%})'
      f' · scarce = {SCARCE} ({cond_rows[SCARCE]} rows)')

MIN_SCARCE_IN_TRAIN = 0.60      # of the scarce condition's rows

def spread_pick(seq, k, taken):
    \"\"\"k evenly spaced picks from seq, skipping anything already taken.\"\"\"
    import numpy as np
    pool = [e for e in seq if e not in taken]
    if not pool or k <= 0:
        return []
    idx = sorted(set(np.linspace(0, len(pool) - 1, min(k, len(pool))).round().astype(int)))
    return [pool[i] for i in idx]

# --- the experiment design; edit HERE and nowhere else -----------------------------------
# 1) a whole day, but only from the abundant condition, and its smallest pure day.
per_day_conditions = collections.defaultdict(set)
for e in exp_ids:
    per_day_conditions[day_of(e)].add(cond_of(e))
pure_days = [d for d, cs in per_day_conditions.items() if cs == {ABUNDANT}]
UNSEEN_DAY = min(pure_days, key=lambda d: sum(rows_of[e] for e in exp_ids if day_of(e) == d))
TEST_DAY = sorted(e for e in exp_ids if day_of(e) == UNSEEN_DAY)
taken = set(TEST_DAY)

# 2) the scarce condition gives up runs, one per day, while training keeps enough of it.
scarce_budget = cond_rows[SCARCE] * (1.0 - MIN_SCARCE_IN_TRAIN)
TEST_SCARCE, VAL_SCARCE, spent = [], [], 0
for target in (TEST_SCARCE, VAL_SCARCE):
    for key in sorted(k for k in groups if k[1] == SCARCE and k[0] != UNSEEN_DAY):
        for pick in spread_pick(sorted(groups[key]), 1, taken):
            if spent + rows_of[pick] > scarce_budget:
                continue
            target.append(pick); taken.add(pick); spent += rows_of[pick]

# 3) the abundant condition gives up a few more runs, from days still in training.
TEST_ABUNDANT, VAL_ABUNDANT = [], []
for target in (TEST_ABUNDANT, VAL_ABUNDANT):
    for key in sorted(k for k in groups if k[1] == ABUNDANT and k[0] != UNSEEN_DAY):
        n = 3 if len(groups[key]) >= 20 else 1
        picked = spread_pick(sorted(groups[key]), n, taken)
        target += picked; taken.update(picked)

VAL_EXPS = sorted(VAL_SCARCE + VAL_ABUNDANT)
TRAIN = [e for e in exp_ids if e not in taken]
TEST_GROUPS = {
    f'unseen_day_{ABUNDANT}': TEST_DAY,
    f'unseen_runs_{SCARCE}': sorted(TEST_SCARCE),
    f'unseen_runs_{ABUNDANT}': sorted(TEST_ABUNDANT),
}

def summarize(label, ids):
    by_cond = collections.Counter(cond_of(e) for e in ids)
    print(f'{label:20s} {len(ids):3d} runs  {sum(rows_of[e] for e in ids):6d} rows  {dict(by_cond)}')
    if 0 < len(ids) <= 6:
        print(f'                     {", ".join(ids)}')

print()
for name, ids in TEST_GROUPS.items():
    summarize('test ' + name, ids)
summarize('val', VAL_EXPS)
summarize('train', TRAIN)

train_scarce = sum(rows_of[e] for e in TRAIN if cond_of(e) == SCARCE)
share = train_scarce / cond_rows[SCARCE]
print(f'\\ntraining keeps {share:.0%} of the {SCARCE} rows ({train_scarce})'
      f' — {train_scarce / sum(rows_of[e] for e in TRAIN):.0%} of the training split')
assert share >= MIN_SCARCE_IN_TRAIN, f'only {share:.0%} of {SCARCE} left in training'
assert all(TEST_GROUPS.values()) and VAL_EXPS and TRAIN
"""),

code("""
CFG = 'config_v03_run.yaml'
import yaml
cfg = yaml.safe_load(open('config_v03.yaml'))
yaml.safe_dump(cfg, open(CFG, 'w'))

groups_arg = ' '.join(f'--test-group {name}={",".join(ids)}'
                      for name, ids in TEST_GROUPS.items() if ids)

!python inspect_dataset.py --config {CFG} --data-root {" ".join(DATA_ROOTS)} \\
    --on-duplicate first {groups_arg} --val-experiments {",".join(VAL_EXPS)}

import pandas as pd
mf = pd.read_csv('artifacts/manifest_v03/manifest.csv')
print(mf.groupby(['split', 'split_group']).size().to_dict())
print(mf['experiment_id'].nunique(), 'experiments ·', len(mf), 'rows')
"""),

md("""
## Step 2 — the region masks

`--masker segmentation` is not optional. Only a semantic segmenter can say "this is drivable
terrain"; a sky detector can only say "this is not sky", which puts trees and buildings in
with the grass — and a non-ground penalty built from that would train the model to look away
from the surface it is supposed to use. The assert below is the guard.

**The failure this step has to catch.** Part of the collection is washed out and colour-cast,
and on those frames the segmenter reads the pale terrain as anything but ground: 4-8% ground
where the lower half of the image is plainly grass. Left alone, that is worse than no
guidance — the penalty would push the attention out of everything except a 4% sliver, which
is exactly how the collapse gets manufactured. Two defences run here: implausible frames are
re-masked from a white-balanced, contrast-equalised copy, and whatever still fails is marked
untrusted and excluded from the guidance (never from training).
"""),

md("""
### Step 2a — which classes is the segmenter putting on the terrain?

Run 1 passed every numeric gate here and still had a defect only the picture showed: on the
wet runs the muddy ruts and puddles were left OUT of the ground mask. That mud is the surface
the vehicle is driving on and the most motion-informative texture in the frame — putting it in
the penalty region would teach the model the opposite of what the whole method is for.

So the class list is not guessed. This samples frames, asks the segmenter what it actually
calls those pixels, and reports every class sitting in the lower part of the frame with the
share it occupies high up as the control. A class is added only if it is big at the bottom,
small at the top, AND looks like terrain in the crops this writes. `hill` and `mountain` fail
the second test, which is what the test is for.
"""),

code("""
!python diagnose_ground_classes.py --data-root "{DATA_ROOTS[0]}" --num 240 \\
    --output-dir artifacts/ground_diagnosis --crops 4

from IPython.display import Image, display
import glob as _glob
for path in sorted(_glob.glob('artifacts/ground_diagnosis/crops/*.png'))[:8]:
    print(path.split('/')[-1])
    display(Image(path, width=260))
"""),

code("""
# Set this from the diagnosis above; '' keeps the built-in list.
# Each name must be one the segmenter itself used, spelled as it prints it.
EXTRA_GROUND = ''
print('ground list extension:', EXTRA_GROUND or '(none)')
"""),

code("""
# No --skip-existing: the cached PNGs hold the OLD class mapping, and skipping them
# would leave the run silently masked by a list this cell no longer uses.
extra_arg = f'--extra-ground-labels \"{EXTRA_GROUND}\"' if EXTRA_GROUND else ''
!python precompute_sky_masks.py --manifest artifacts/manifest_v03/manifest.csv \\
    --output-dir artifacts/masks --masker segmentation --size 224 224 \\
    --batch-size 16 --min-ground 0.15 {extra_arg}

index = json.load(open('artifacts/masks/index.json'))
print(json.dumps({k: v for k, v in index.items()
                  if k not in ('per_experiment_area', 'ground_labels')}, indent=2))
assert index['provides_ground'], (
    'The mask cache has no ground channel — the segmentation weights did not load. '
    'Check Internet is ON, then re-run.')
print('\\nground classes used:', index['ground_labels'])
"""),

code("""
# Sanity-check the masks BEFORE spending GPU hours on them.
q = pd.read_csv('artifacts/masks/quality.csv')
q['day'] = q['experiment_id'].str.split('_').str[0]

print(f"{len(q)} frames · {int((~q.trusted.astype(bool)).sum())} untrusted "
      f"({1 - q.trusted.mean():.1%}) · {index['n_rescued_by_enhancement']} rescued by enhancement")
print()
by_day = q.groupby('day').agg(frames=('trusted', 'size'), trusted=('trusted', 'mean'),
                              ground=('ground', 'mean'), sky=('sky', 'mean'),
                              other=('other', 'mean')).round(3)
print(by_day.to_string())

worst = (q.groupby('experiment_id')['trusted'].mean().sort_values().head(10).round(3))
if (worst < 0.9).any():
    print('\\nexperiments the segmenter struggles with:')
    print(worst[worst < 0.9].to_string())

# Ground should be roughly 40-55% of a frame for a forward-driving ground vehicle.
g = q.loc[q.trusted == 1, 'ground'].mean()
print(f'\\nmean ground area on trusted frames: {g:.1%}')
assert 0.30 <= g <= 0.70, (
    f'ground area {g:.1%} is not plausible for this platform — check the class list in '
    f"index['ground_labels'] before spending GPU time on these masks.")
if 1 - q.trusted.mean() > 0.25:
    print('\\nWARNING: more than a quarter of frames are untrusted. Guidance will only act on '
          'the rest, which is safe but weak — worth looking at a few of those frames.')
"""),

md("""
## Step 3 — audit the masks against depth

Every run here ships an aligned depth frame, which gives a check no learned mask can give
itself: real sky must sit inside the region the camera reports as beyond usable range.
"""),

code("""
!python validate_sky_mask.py --data-root "{DEPTH_ROOT}" --output-dir artifacts/mask_audit \\
    --num-per-exp 12 --masker segmentation --qualitative 6 {extra_arg}
from IPython.display import Image, display
display(Image('artifacts/mask_audit/f_mask_qualitative.png'))
display(Image('artifacts/mask_audit/f_mask_audit.png'))
"""),

md("""
## Step 4 — train the v0.3 base model

No guidance. This is the model that answers "how much of any improvement is just the new
data?", and every guided run starts from it. The long cell — most of the GPU budget.
"""),

code("""
# The base model never sees a mask, so a change to the region masks does not
# invalidate it — a base checkpoint from an earlier session is reused as-is, and
# the ~3 GPU-hours go to the fine-tunes instead.
os.makedirs('artifacts/runs/v03_base', exist_ok=True)
if REUSE_BASE:
    shutil.copy(REUSE_BASE, 'artifacts/runs/v03_base/best.pt')
    for side in ('normalization.json', 'history.json'):
        near = os.path.join(os.path.dirname(REUSE_BASE), side)
        if os.path.exists(near):
            shutil.copy(near, f'artifacts/runs/v03_base/{side}')
    print('REUSING the attached base checkpoint — training skipped.')
    print('   ', REUSE_BASE)
    print('   Only valid because the split is deterministic: same code, same data, same'
          ' seed. Check the split table in Step 1 matches the session that trained it —'
          ' a different split here means this checkpoint has seen the test rows.')
else:
    # --balance day_condition: 82/18 wet/dry AND one day holding 72% of the rows, so
    # balancing on condition alone would still leave that day dominating. It reweights
    # an epoch; it does not invent dry diversity, and the dry numbers are reported
    # separately regardless.
    !python train.py --config {CFG} --balance day_condition

hist = 'artifacts/runs/v03_base/history.json'
print(open(hist).read()[-900:] if os.path.exists(hist) else 'no history.json beside the checkpoint')
assert os.path.exists('artifacts/runs/v03_base/best.pt'), 'no base checkpoint to fine-tune from'
"""),

md("""
## Step 5 — the guided fine-tunes

Five runs, three epochs each, all from the same base checkpoint:

| run | region | λ_rrr | λ_spread | what it isolates |
|---|---|---|---|---|
| `v1_tophalf` | top half of the frame | 1.0 | 0 | the deck's version, reproduced |
| `sky_only` | segmented sky | 1.0 | 0 | a real mask, still only sky |
| `nonground` | sky + structure | 1.0 | 0 | the right region, no spread term |
| `guided` | sky + structure | 1.0 | 0.5 | **the v0.3 proposal** |
| `control` | — | 0 | 0 | same extra training, no guidance |
"""),

code("""
RUNS = {
    'v1_tophalf': ['--mask-mode', 'half',      '--lambda-rrr', '1.0', '--lambda-spread', '0'],
    'sky_only':   ['--mask-mode', 'sky',       '--lambda-rrr', '1.0', '--lambda-spread', '0'],
    'nonground':  ['--mask-mode', 'nonground', '--lambda-rrr', '1.0', '--lambda-spread', '0'],
    'guided':     ['--mask-mode', 'nonground', '--lambda-rrr', '1.0', '--lambda-spread', '0.5'],
    'control':    ['--mask-mode', 'nonground', '--lambda-rrr', '0.0', '--lambda-spread', '0'],
}
BASE = 'artifacts/runs/v03_base/best.pt'
for name, extra in RUNS.items():
    print('\\n===', name, '===')
    subprocess.run(['python', 'finetune_rrr.py', '--config', CFG, '--checkpoint', BASE,
                    '--sky-mask-dir', 'artifacts/masks', '--epochs', '3', '--target', 'yaw',
                    '--max-steps', '200',
                    '--output', f'artifacts/runs/{name}/best.pt'] + extra, check=True)
"""),

md("""
## Step 6 — measure them all with the same three numbers

`distractor_fraction` is what v1 optimized. `ground_coverage` is what it broke. Read them
together: a run can score 0.00 on the first and 0.04 on the second, which is the "three hot
spots on the grass" result written as a number.
"""),

code("""
for name in ['v03_base'] + list(RUNS):
    ck = BASE if name == 'v03_base' else f'artifacts/runs/{name}/best.pt'
    subprocess.run(['python', 'interpret_quantify.py', '--config', CFG, '--checkpoint', ck,
                    '--sky-mask-dir', 'artifacts/masks', '--experiments', 'auto',
                    '--max-exp', '8', '--num-per-exp', '30', '--target', 'yaw',
                    '--saliency', 'inputgrad', '--tag', name], check=True)

rows = []
for tag in ['v03_base'] + list(RUNS):
    d = json.load(open(f'artifacts/quantify/focus_{tag}.json'))
    h = d['headline']
    rows.append({'run': tag,
                 'distractor_fraction': h['distractor_fraction']['mean'],
                 'ground_concentration': h['ground_concentration']['mean'],
                 'ground_coverage': h['ground_coverage']['mean'],
                 'top_decile_share': d['spread_on_ground']['top_decile_share']['mean'],
                 'legacy_top_half': d['legacy_top_half']['fraction']['mean']})
focus = pd.DataFrame(rows).round(3)
focus.to_csv('artifacts/quantify/focus_summary.csv', index=False)
display(focus)
"""),

md("""
## Step 7 — the heatmap comparison

The figure the deck needs: same frames, same saliency, same mask, four models side by side,
each panel annotated with both numbers. One figure per held-out group.
"""),

code("""
for exp in [ids[0] for ids in TEST_GROUPS.values() if ids]:
    out = f'artifacts/interpretability/compare_{exp}.png'
    subprocess.run(['python', 'compare_interpret.py', '--config', CFG,
                    '--checkpoints', f'base={BASE}',
                    'v1_tophalf=artifacts/runs/v1_tophalf/best.pt',
                    'nonground=artifacts/runs/nonground/best.pt',
                    'guided=artifacts/runs/guided/best.pt',
                    '--sky-mask-dir', 'artifacts/masks', '--experiment', exp,
                    '--num-samples', '4', '--target', 'yaw', '--output', out], check=True)
    display(Image(out))
"""),

md("""
## Step 8 — accuracy, per held-out group

`unseen_day` and `unseen_runs` are reported apart. If guidance helps generalization, the gap
should show up on `unseen_day` first — that is the one that needs the model not to have
memorized a particular site's trees and skyline.
"""),

code("""
pairs = [f'v03_base={BASE}'] + [f'{n}=artifacts/runs/{n}/best.pt' for n in RUNS]
norms = []
if OLD_CKPT and OLD_NORM:
    pairs.insert(0, f'v021_old={OLD_CKPT}')
    norms = ['--normalizations', f'v021_old={OLD_NORM}']
    print('scoring the old v0.2.1 checkpoint with its own normalization')

subprocess.run(['python', 'evaluate_newdata.py', '--config', CFG, '--checkpoints', *pairs,
                *norms, '--split', 'test', '--output-dir', 'artifacts/eval_test',
                '--batch-size', '32'], check=True)
display(Image('artifacts/eval_test/f_newdata_accuracy.png'))

m = json.load(open('artifacts/eval_test/newdata_metrics.json'))
acc = pd.DataFrame([
    {'run': k,
     'dx_R2': v['next_body_dx']['r2'],
     'yaw_R2': v['next_delta_yaw']['r2'],
     **{f'{g}_yaw_R2': m['by_group'][k][g].get('next_delta_yaw', {}).get('r2')
        for g in m['groups']}}
    for k, v in m['pooled'].items()]).round(3)
display(acc)
acc.merge(focus, on='run', how='outer').to_csv('artifacts/v03_summary.csv', index=False)
"""),

md("""
### The figure the argument rests on

Every arm lowers `distractor_fraction` — a penalty on a region always does. What separates
them is what that focus COST, and the two numbers sit in different columns of the table where
nobody compares them. On one pair of axes it is a single picture: up and to the right is
"looks at the terrain, and at all of it". The accuracy panel beside it is what lets the claim
be "at no cost" rather than "we hope at no cost".
"""),

code("""
!python make_tradeoff_figure.py --summary artifacts/v03_summary.csv \\
    --output artifacts/f_tradeoff.png
display(Image('artifacts/f_tradeoff.png'))
"""),

md("""
## Step 9 — λ sweep (optional, slow)

Everything above uses λ_rrr = 1, λ_spread = 0.5. This checks the result is not a knife-edge.
"""),

code("""
SWEEP = [(1.0, 0.25), (1.0, 1.0), (2.0, 0.5), (0.5, 0.5)]
for lr, ls in SWEEP:
    name = f'sweep_r{lr}_s{ls}'
    subprocess.run(['python', 'finetune_rrr.py', '--config', CFG, '--checkpoint', BASE,
                    '--sky-mask-dir', 'artifacts/masks', '--mask-mode', 'nonground',
                    '--lambda-rrr', str(lr), '--lambda-spread', str(ls),
                    '--epochs', '3', '--target', 'yaw', '--max-steps', '200',
                    '--output', f'artifacts/runs/{name}/best.pt'], check=True)
    subprocess.run(['python', 'interpret_quantify.py', '--config', CFG,
                    '--checkpoint', f'artifacts/runs/{name}/best.pt',
                    '--sky-mask-dir', 'artifacts/masks', '--experiments', 'auto',
                    '--max-exp', '8', '--num-per-exp', '30', '--target', 'yaw',
                    '--saliency', 'inputgrad', '--tag', name], check=True)
"""),

code("""
shutil.make_archive('/kaggle/working/vit_motion_v03_artifacts', 'zip', 'artifacts')
print('done -> vit_motion_v03_artifacts.zip')
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("kaggle_v03_ground_focus.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote kaggle_v03_ground_focus.ipynb")
