"""Controlled single-model date-compression and hierarchy study."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ehr_forecast.train import train, write_json

SEEDS = [17, 42, 73]
FAMILIES = {
    "raw_control": ("prepared_raw", "compressed_transformer"),
    "date_recent": ("prepared_date_recent", "compressed_transformer"),
    "date_full_flat": ("prepared_date_full", "compressed_transformer"),
    "date_full_hier": ("prepared_date_full", "hierarchical_transformer"),
    "date_full_overlap": ("prepared_date_full", "hierarchical_transformer"),
    "date_overlap16": ("prepared_date_full", "hierarchical_transformer"),
    "date_local2": ("prepared_date_full", "hierarchical_transformer"),
    "date_sqrt_hier": ("prepared_date_full", "hierarchical_transformer"),
}


def main():
    parser = argparse.ArgumentParser()
    core_families = ["raw_control", "date_recent", "date_full_flat", "date_full_hier", "date_full_overlap"]
    parser.add_argument("--families", nargs="+", choices=list(FAMILIES), default=core_families)
    args = parser.parse_args()
    root = ROOT / "artifacts/hierarchical"
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "seeds": SEEDS, "families": {name: FAMILIES[name] for name in core_families},
        "comparisons": [["raw_control", "date_recent", "compression on the same last 511 events"],
                        ["date_recent", "date_full_flat", "additional historical coverage with the same flat model"],
                        ["date_full_flat", "date_full_hier", "hierarchical topology on exactly identical input"],
                        ["date_full_hier", "date_full_overlap", "overlapping local context with unchanged parameters"]],
        "selection": "Single-model mean across every declared seed. Each checkpoint selected by overall validation mAP only.",
        "primary_evaluation": "Fixed long-history validation cohort: full deduplicated pre-anchor event count >255 (133 patients)",
        "secondary_evaluation": "overall/short mAP, AUROC, Brier, per-target support, runtime and memory",
        "bootstrap": "Shared patient resampling across both architectures and all seeds; metrics computed per seed then averaged. No ensemble predictions.",
        "fixed_budget": "d128, 4heads, 3layers, position capacity512, batch16, AdamW3e-4 wd0.01, dropout0.15, max35epochs, patience8, no pretraining",
        "representation": "Same-day event mean/count embedding, date/time features. 512 units x <=128 event bag. No within-day event truncation.",
        "hierarchy": "1 local layer in32-date windows with stride32/16 controls, pooled then2 global layers; Hi-BEHRT-inspired, not exact replication",
        "current_submission": "Retained; this study evaluates single models without promoting a seed or ensemble",
        "test_selection": False,
    }
    if not (root / "plan.json").exists():
        write_json(root / "plan.json", plan)
    refinements = [name for name in args.families if name in {"date_overlap16", "date_local2", "date_sqrt_hier"}]
    if refinements and not (root / "refinement_plan.json").exists():
        write_json(root / "refinement_plan.json", {
            "families": refinements, "seeds": SEEDS, "adaptive": True,
            "reason": "Raw-long validation patients have a median74 dates, yielding only3-4 global windows atW32. Trained mean-bag vectors also shrink for large date bags (observational, not yet causal evidence). Coarse pooling and clinical-signal dilution are concerns after the core results.",
            "contrasts": [["date_full_overlap", "date_overlap16", "window32/stride16 to window16/stride8, still1local+2global"],
                          ["date_overlap16", "date_local2", "same16/8 windows, redistribute3layers from1local+2global to2local+1global"],
                          ["date_local2", "date_sqrt_hier", "same input/parameters/initialization, replace event mean with sum/sqrt(count)"]],
            "controls": "Same data/parameters/initial layers/seeds/training budget; no seed or ensemble selection",
            "scope": "Adaptive sensitivity analysis; not a pre-registered confirmation and no claim of independent validation",
        })
    import torch
    for family in args.families:
        cache, model_type = FAMILIES[family]
        for seed in SEEDS:
            config = {
                "name": f"{family}-seed{seed}", "family": family, "seed": seed,
                "prepared_dir": f"artifacts/hierarchical/{cache}",
                "output_dir": f"artifacts/hierarchical/runs/{family}-seed{seed}",
                "model_type": model_type, "d_model": 128, "n_heads": 4, "n_layers": 3,
                "max_unit_events": 128, "window_size": 32, "local_layers": 1, "global_layers": 2,
                "window_stride": 16 if family == "date_full_overlap" else 32,
                "model_max_events": 512, "dropout": 0.15, "batch_size": 16,
                "epochs": 35, "pretrain_epochs": 0, "learning_rate": 3e-4,
                "weight_decay": 0.01, "patience": 8, "device": "auto", "amp": True,
                "wandb_mode": "offline", "wandb_project": "patient-timeline-forecast",
                "validation_groups_csv": "artifacts/optimization/history_patient_groups.csv",
            }
            if family in {"date_overlap16", "date_local2", "date_sqrt_hier"}:
                config.update(window_size=16, window_stride=8)
            if family in {"date_local2", "date_sqrt_hier"}:
                config.update(local_layers=2, global_layers=1)
            if family == "date_sqrt_hier":
                config.update(bag_normalization="sqrt")
            write_json(ROOT / "configs/hierarchical" / f"{config['name']}.json", config)
            directory = ROOT / config["output_dir"]
            if (directory / "summary.json").exists():
                saved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
                if saved != config:
                    raise ValueError(f"Run configuration differs: {directory}")
                print(f"SKIP {config['name']}", flush=True)
                continue
            print(f"BEGIN {config['name']}", flush=True)
            summary = train(config)
            print(json.dumps({"completed": config["name"], **summary}), flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
