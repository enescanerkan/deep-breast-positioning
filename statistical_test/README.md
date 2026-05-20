# Statistical Analysis — 10-fold Cross-Validation

This folder contains the analysis script that reproduces the paper's per-image (Wilson 95% CI, McNemar, Cochran's Q) and fold-level (paired Wilcoxon, paired Cohen's d, bootstrap 95% CI) statistics for the Attention U-Net, U-Net, and ResNeXt50 models.

## Files

- `run_cv_statistics.py` — main analysis script.
- `requirements.txt` — `numpy`, `pandas`, `scipy`, `statsmodels`.

> The `output/` folder is intentionally not committed. Generate it locally with the steps below.

## Expected input layout

The script reads per-fold evaluation exports from `<repo_root>/results/`:

```
results/
├── UNet_fold_{1..10}_eval_per_image.csv      # produced by code/regression/main/evaluation.py
├── RAUNet_fold_{1..10}_eval_per_image.csv    # produced by code/regression/main/evaluation.py
├── ResNeXt50_fold_1.csv                       # produced by code/classification/test.py (fold 1)
└── ResNeXt50_fold_{2..10}_eval.json           # produced by code/classification/test.py (folds 2–10)
```

Each `UNet_fold_{k}_eval_per_image.csv` / `RAUNet_fold_{k}_eval_per_image.csv` must include:
- `study_uid`, `sop_uid`
- `mm_perpendicular`, `mm_pec1`, `mm_pec2`, `mm_nipple`, `angular_distance` (landmark errors)
- `automated_label`, `manual_label` (`good` / `bad`)
- `prediction_label` (regression-derived PNL prediction, `good` / `bad`)

Each `ResNeXt50_fold_{k}_eval.json` (folds 2–10) is a JSON object with a `per_image` list:
```json
{
  "per_image": [
    {"StudyInstanceUID": "...", "SOPInstanceUID": "...", "prediction_label": 0},
    {"StudyInstanceUID": "...", "SOPInstanceUID": "...", "prediction_label": 1}
  ]
}
```
where `prediction_label` is `1` (good) or `0` (bad). This is the format produced by `code/classification/test.py --config configs/ResNeXt50_fold_{k}_eval.json`.

`ResNeXt50_fold_1.csv` (fold 1, legacy format) is a flat CSV with columns `index, StudyInstanceUID, SOPInstanceUID, Ground Truth, Prediction` (output of the baseline `test.py` invocation).

## Run

```bash
# From repo root
pip install -r statistical_test/requirements.txt
python3 statistical_test/run_cv_statistics.py
```

Outputs are written to `statistical_test/output/`:

| File | Contents |
|---|---|
| `table1_landmark_unet_vs_raunet.csv` | Pooled landmark mean errors (mm) with bootstrap 95% CI, paired Wilcoxon p, Cohen's d for U-Net vs Attention U-Net. Reproduces manuscript Table 2. |
| `table2_classification_per_image_automated.csv` | Per-image classification metrics (accuracy / sensitivity / specificity) with Wilson 95% CI, vs the automated PNL reference. Reproduces manuscript Table 3. |
| `table2_classification_per_image_manual.csv` | Same, vs the expert qualitative reference. Reproduces manuscript Table 4. |
| `table3_fold_descriptives_{automated,manual}.csv` | Fold-level mean ± std with t-distribution 95% CI (df = 9). |
| `table3_fold_wilcoxon_{automated,manual}.csv` | Paired Wilcoxon p-values across the ten folds. |
| `table3_fold_cohens_d_{automated,manual}.csv` | Paired Cohen's d effect sizes on fold-level differences. |
| `table4_mcnemar_{automated,manual}.csv` | Per-image McNemar p-values for pairwise model comparisons. |
| `table4_cochrans_q_{automated,manual}.csv` | Cochran's Q statistic and p-value for the joint three-model comparison. |
| `SUPPLEMENT_notes.txt` | Run metadata + cross-check between pooled per-image numbers and the manuscript tables. |

## Reproducing the full pipeline (fold splits → training → evaluation → statistics)

1. **Generate 10-fold CSVs** (writes `labels/positioning_labels_fold_{2..10}.csv`; fold 1 is the existing `labels/positioning_labels.csv`):
   ```bash
   python3 scripts/cv_splits.py --mode rotating_9 --write --folds 2-10
   ```

2. **Verify fold integrity** (pairwise disjoint test sets, full 2,000-image test coverage, within-fold Train/Val/Test disjointness):
   ```bash
   python3 scripts/verify_folds.py
   ```

3. **Generate per-fold training/eval configs** under `configs/`:
   ```bash
   python3 scripts/cv_generate_configs.py --folds 2-10
   ```

4. **Train and evaluate each model on each fold**, writing per-image eval exports to `results/`. See `code/regression/main/main.py` and `code/regression/main/evaluation.py` for regression, and `code/classification/main.py` / `test.py` for ResNeXt50.

5. **Run statistics**:
   ```bash
   python3 statistical_test/run_cv_statistics.py
   ```

## Metric definitions

Sensitivity / specificity match `code/regression/main/utils/evaluation_utils.py::calculate_sensitivity_specificity`:
- Binary labels: `good = 1`, `bad = 0` (the manuscript uses the prose term `poor` for the `bad` class; CSV / code identifier is `Bad`).
- **Sensitivity** = recall for the `bad` (positive) class — i.e., the `poor`-positioning detection task.
- **Specificity** = recall for the `good` class.

Landmark Cohen's d uses **U-Net − Attention U-Net** paired error differences (positive d ⇒ U-Net has higher error on that landmark).
