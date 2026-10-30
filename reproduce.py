"""One-command reproduction of the submitted pipeline (three gradient-boosted members + per-target rank assembly).

    python reproduce.py features      # raw CSV tables -> clean feature matrix, genomics/cost features, person-period dataset
    python reproduce.py members       # train member A (person-period XGBoost, screened), B (official-row LightGBM, screened), C (history-free)
    python reproduce.py assemble      # per-target Caruana greedy rank assembly, cross-fitted OOF, writes predictions.csv
    python reproduce.py all           # everything above in order
    python reproduce.py evaluate      # honest metrics of predictions on the official validation split + train OOF

Environment variables: EXPORT_DIR (default: export/) receives the trained boosters, screened column indices and assembly
weights; WANDB_PROJECT (default: ehr-forecast-5y) enables wandb logging; set WANDB_MODE=offline without an API key.
Data is expected under research_intern_coding_test/ (the task's folder). GPU (CUDA) is used by member A via XGBoost.
"""
import os, sys, subprocess, time, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent; AN = ROOT / "analysis"; PY = sys.executable
os.environ.setdefault("EXPORT_DIR", str(ROOT / "export")); os.environ.setdefault("WANDB_PROJECT", "ehr-forecast-5y")
MEMBERS = "ppds_gcr:screen200_d3_all,screen_gcr:k200,gstack:genc2_only"
def run(cmd, env=None, cwd=AN):
    e = dict(os.environ); e.update(env or {}); t0 = time.time(); print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run([PY, "-X", "utf8", *cmd], cwd=str(cwd), env=e, check=True); print(f"  [{time.time()-t0:.0f}s]", flush=True)
def features():
    run(["build_clean_dataset.py"])                                   # 10 structured tables -> analysis/clean/X.parquet (train-fitted vocabulary)
    run(["build_genomics_features.py"])                               # genomics/*_dna.csv, imaging/*.dcm (pre-anchor only) -> clean/genomics_features.parquet
    run(["death_model.py"])                                           # cross-fitted P(death in window) -> stack/pdeath_*.npy (input of the cost residuals)
    run(["build_cost_features.py"]); run(["build_cost_features_v2.py"])  # lifetime cost fields, pre-anchor spending, two-stage residuals
    run(["pp_build.py"])                                              # person-period (landmark-year) dataset -> analysis/ppds/
    run(["pp_join_extra.py", "ppds", "ppds_gcr", "clean/genomics_features.parquet", "clean/cost_features_v2.parquet"])
def members():
    run(["pp_screen.py", "200", "--ds", "ppds_gcr"])                 # member A
    run(["screen_gbm.py", "200"], env={"EHR_GENOMICS": "3"})          # member B (clean X + genomics + cost v2)
    run(["gen_only.py", "cost2"])                                     # member C
def assemble():
    run(["assemble_final.py", "--rank", "--cf-reps", "5", "--only", MEMBERS])
    shutil.copy(ROOT / "predictions_final.csv", ROOT / "predictions.csv"); (ROOT / "predictions_final.csv").unlink()
    run(["-m", "ehr_forecast", "validate-submission", "--predictions", "predictions.csv"], cwd=ROOT)
    print("\npredictions.csv written and validated")
def evaluate():
    run(["evaluate_submission.py"], cwd=ROOT)
if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"features": features, "members": members, "assemble": assemble, "evaluate": evaluate, "all": lambda: (features(), members(), assemble(), evaluate())}[stage]()
