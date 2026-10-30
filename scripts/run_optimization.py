"""Small, declared validation-only experiments; one GPU run at a time."""

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
OUTPUT = ROOT / "artifacts/optimization"


def experiment_config(family, seed, prepared, model_type="transformer", pretrain_epochs=4, **updates):
    config = {
        "name": f"{family}-seed{seed}", "family": family,
        "prepared_dir": str(prepared.relative_to(ROOT)),
        "output_dir": f"artifacts/optimization/runs/{family}-seed{seed}",
        "model_type": model_type, "seed": seed,
        "d_model": 128, "n_heads": 4, "n_layers": 3, "dropout": 0.15,
        "batch_size": 16, "epochs": 35, "pretrain_epochs": pretrain_epochs,
        "learning_rate": 3e-4, "pretrain_learning_rate": 5e-4,
        "weight_decay": 0.01, "patience": 8,
        "device": "auto", "amp": True, "wandb_mode": "offline",
        "wandb_project": "patient-timeline-forecast",
    }
    config.update(updates)
    return config


def run_family(configs):
    import torch
    for config in configs:
        config_path = ROOT / "configs/optimization" / f"{config['name']}.json"
        write_json(config_path, config)
        output = ROOT / config["output_dir"]
        if (output / "summary.json").exists():
            saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
            if saved != config:
                raise ValueError(f"Existing configuration differs: {output}")
            print(f"SKIP completed {config['name']}", flush=True)
            continue
        print(f"BEGIN {config['name']}", flush=True)
        summary = train(config)
        print(json.dumps({"completed": config["name"], **summary}), flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("original", "length", "summary", "summary-clip", "no-pretrain", "pooling", "weighted", "followups", "refinements"), required=True)
    parser.add_argument("--length", type=int, choices=(256, 512, 1024), default=512)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "seeds": SEEDS, "lengths": [256, 512, 1024],
        "selection": "family mean validation mAP; never best seed, no hidden test labels",
        "length_controls": "same full-training vocabulary/bins/static, same model position capacity 1024, batch16 and training schedule",
        "diagnostic_followups": ["full-history summary linear and MLP", "remove next-event pretraining", "mean+last pooling with nonlinear static fusion"],
        "uncertainty": "seed sample std separately from paired patient bootstrap; validation selection bias remains",
    }
    if not (OUTPUT / "plan.json").exists():
        write_json(OUTPUT / "plan.json", plan)
    configs = []
    stages = (["original", "summary", "no-pretrain", "pooling"] if args.stage == "followups" else
              ["weighted", "summary-clip"] if args.stage == "refinements" else [args.stage])
    for stage in stages:
        if stage == "original":
            base = json.loads((ROOT / "configs/transformer.json").read_text(encoding="utf-8"))
            # Seed 42 was already fitted in the first iteration; preserve it.
            for seed in (17, 73):
                config = {**base, "name": f"original-seed{seed}", "seed": seed,
                          "output_dir": f"artifacts/optimization/runs/original-seed{seed}"}
                configs.append(config)
        elif stage == "length":
            for length in (256, 512, 1024):
                for seed in SEEDS:
                    configs.append(experiment_config(f"length-l{length}", seed, OUTPUT / f"prepared_l{length}", model_max_events=1024))
        elif stage in {"summary", "summary-clip"}:
            for kind in (("mlp",) if stage == "summary-clip" else ("linear", "mlp")):
                for seed in SEEDS:
                    family = "summary-clip5" if stage == "summary-clip" else f"summary-{kind}"
                    cache = "prepared_summary_clip5" if stage == "summary-clip" else "prepared_summary"
                    configs.append(experiment_config(family, seed, OUTPUT / cache, model_type=f"summary_{kind}",
                                                     pretrain_epochs=0, dropout=0.35, batch_size=64, epochs=70,
                                                     learning_rate=1e-3, weight_decay=0.03, patience=12))
        else:
            for seed in SEEDS:
                family = f"{stage}-l{args.length}"
                configs.append(experiment_config(family, seed, OUTPUT / f"prepared_l{args.length}",
                                                 model_type="pooled_transformer" if stage in {"pooling", "weighted"} else "transformer",
                                                 pretrain_epochs=4 if stage in {"pooling", "weighted"} else 0,
                                                 model_max_events=1024,
                                                 **({"loss_positive_weight": "sqrt"} if stage == "weighted" else {})))
    run_family(configs)


if __name__ == "__main__":
    main()
