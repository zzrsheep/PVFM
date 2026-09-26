# Independent repository and large assets

This directory is a verified copy of the curated reproduction package,
not a move/delete of the original research workspace. It intentionally
excludes old Git history, duplicate extracted releases, temporary training
checkpoints, caches and logs. Final validation CSV/JSON reports are retained.
Private source-machine paths and archived shell commands are not distributed.
Provenance retains neutral source identifiers and hashes, not personal paths.
Use the portable entry points from the repository root.

## Weights

| Profile | File | Scope |
|---|---|---|
| `all_region` (default) | `checkpoints/pvfm_3_9m_80k_q11.pt` | Target stations excluded, other Australia sites included |
| `australia_holdout` | `checkpoints/pvfm_3_9m_australia_holdout_80k_q11.pt` | All Australia sites excluded |

All active tensors are identical to their respective final 80k Q11 models.
Unused calendar and scalar-head tensors are omitted; see the model card.
Neither includes optimizer state, sampler state or pretraining observations.
`configs/evaluation_profiles.json` binds weights to frozen inputs.
`reference/australia_holdout/common_q9_task_metrics.csv` is the correct
five-task common-Q9 reference for the second profile. Do not substitute the
legacy Q11 hourly AQL/CRPS values in `reference/australia_archived/`.

## Git versus assets

The local directory includes the Australian raw train/val/test data
and both checkpoints, but not prepared evaluation arrays. `.gitignore` excludes
the weights, bulk raw data, binary arrays, runtime outputs and `dist/` from
ordinary Git tracking. The prepared manifests and `SHA256SUMS.json` remain
text metadata. A Git-only checkout is therefore NOT runnable until its
matching assets are installed. No Hong Kong or UK demonstration data is included.
Frozen evaluation manifests describe optional replay arrays that are currently
absent and are not listed as distributed assets in `ASSETS.json`.

For distribution, choose a separately licensed release archive or Git LFS
for these assets. This operation does not choose a remote, upload assets,
add a blanket license, or publish checkpoints. Code, model, AEMO and weather
attribution/permission terms must be confirmed before public release.
The asset list (relative path, bytes, SHA256) is `ASSETS.json`; preserve paths
when installing assets into a code checkout. Do not commit binary files
with `git add -f` merely to bypass the ignore rules.

## Verify after copying

From the directory containing this README and `verify.py`:

```bash
python verify.py
python -m pytest tests -q -p no:cacheprovider --basetemp outputs/test_tmp
python evaluate_australia.py --model-profile australia_holdout \
  --station Bannerton --limit-windows 2 --output outputs/holdout_smoke
```

The evaluation smoke command requires the separate prepared arrays; without
them it fails explicitly without loading weights. Data-dependent tests report
skips when the entire prepared-array set is absent, not successful replay.
With those assets present, the smoke command runs all five tasks on CPU.
For complete scores remove
the station/window limits; use `--device cuda` if available. Test/output
directories must be new. No evaluation command initiates training.
