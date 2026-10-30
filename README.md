# Five-year first-onset prediction from synthetic EHR (Synthea / Coherent)

Predict, for each of 40 target conditions, the probability that a patient receives the condition for the **first time**
in the five years after an anchor date (anchor = last recorded encounter − 5 years), using only events before the anchor.
Scored by macro AUROC and mean average precision (mAP) over the 40 conditions on a held-out test split.

**Final model:** three small gradient-boosted members on engineered patient-state features, combined per target by
Caruana greedy selection on rank-transformed out-of-fold scores.

| Split | macro AUROC | mAP | accident / chain / terminal mAP |
| --- | ---: | ---: | --- |
| train, 5-fold cross-fitted OOF (2,791 patients) | 0.786 | 0.306 | 0.098 / 0.298 / 0.632 |
| official validation (365 patients) | 0.787 | 0.350 | 0.102 / 0.332 / 0.748 |

Random (= prevalence) mAP is 0.047; macro AUROC 0.5. Test labels are withheld; `predictions.csv` is the submission.

## Pipeline

```
raw CSV tables ──► analysis/build_clean_dataset.py      2,317 per-patient features (token counts / recency / first, numeric summaries, prior flags, static)
genomics/, imaging/ ──► analysis/build_genomics_features.py   216 features (variant dosages per rsID, gene counts, DICOM-before-anchor flag)
patients.csv cost fields ──► analysis/build_cost_features{,_v2}.py   25 features (lifetime cost, pre-anchor spending, two-stage residuals)
                      ──► analysis/pp_build.py + pp_join_extra.py   person-period dataset: one row per patient per landmark year (56,770 rows)

member A  analysis/pp_screen.py     XGBoost depth 3 on person-period rows, per-target univariate-AP screening (K = 200)      OOF 0.295
member B  analysis/screen_gbm.py    LightGBM on official-anchor rows, per-target screening (k = 200)                          OOF 0.283
member C  analysis/gen_only.py      LightGBM on age / sex / prior flags + genomics + cost only (no history tokens)            OOF 0.260
assembly  analysis/assemble_final.py  per-target greedy weights on rank-transformed OOF, cross-fitted at patient level       OOF 0.306
```

Why this design, in one paragraph: the 40 targets split into 15 accident-type conditions that Synthea generates with a
constant hazard (no history can predict them; AP ≈ prevalence), 15 chain conditions driven by the patient's module state
at the anchor (which precursors happened, how long ago, latest lab values), and 10 terminal conditions that coincide with
death in the window. The signal therefore lives in presence / recency / count features and a few dense fields, not in
event order; with 2,791 training patients, shallow boosted trees with per-target feature screening are far more
sample-efficient than sequence models (a decoder-only Transformer reaches OOF 0.17, tabular MLPs 0.21–0.22; see
`docs/diagnosis_and_improvement_plan.md` §10.6). Member A reads the history, member B re-reads it with a different row
representation and regulariser, member C reads the dense modality fields without dilution; the per-target weights
confirm the division of labour (chain targets → A, terminal targets → C).

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt      # Windows; use .venv/bin/pip elsewhere
# place the task data folder at ./research_intern_coding_test/
python reproduce.py features     # ~5 min, CPU
python reproduce.py members      # member A needs a CUDA GPU (XGBoost), ~10 min; B and C are CPU, ~5 min
python reproduce.py assemble     # writes predictions.csv and validates it
python reproduce.py evaluate     # report/metrics.json, report/per_condition_metrics.csv
```

`WANDB_PROJECT` (default `ehr-forecast-5y`) logs training / validation curves and metrics for the three members and the
assembly; set `WANDB_MODE=offline` without an API key. `EXPORT_DIR` (default `export/`) receives every trained booster,
the screened column indices per target and fold, and the assembly weights; `python predict.py --export export`
regenerates `predictions.csv` from that bundle (this is what the Hugging Face model repository contains) and reports the
validation metrics.

Rules followed: training uses the train split only (vocabularies, screening, models, ensemble weights are fitted inside
train folds); the validation split is used for monitoring; test rows are read only as prediction inputs; no external data
or pretrained weights.

Links: wandb project [ehr-forecast-5y](https://forge.coreweave.com/wandb/chengxianghuang/ehr-forecast-5y) (training / validation curves of the three members and the assembly metrics); model bundle on Hugging Face [Chengxiang1122/synthea-5y-first-onset-forecast](https://huggingface.co/Chengxiang1122/synthea-5y-first-onset-forecast); technical report `report/results.pdf`.

## Repository map

- `reproduce.py`, `predict.py`, `evaluate_submission.py` — entry points.
- `analysis/` — every script used in the project, including the audits and the negative results (Transformer and MLP
  variants, mixtures of experts, stacking, learning curves); `analysis/README.md` lists them by round.
- `docs/data_audit.md` — leak audit of the raw tables and of the optional genomics / imaging / cost fields.
- `docs/diagnosis_and_improvement_plan.md` — the full experimental record (12 rounds) with the per-condition analysis.
- `ehr_forecast/` — the original Transformer baseline package (kept as the sequence-model ablation).
- `report/` — the technical report and its metric tables.
