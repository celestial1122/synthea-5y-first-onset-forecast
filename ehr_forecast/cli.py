"""Commands are deliberately explicit so test labels never enter training."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Synthetic patient timeline forecasting")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build audited train-only preprocessing and official splits")
    prepare.add_argument("--data-dir", type=Path, default=Path("research_intern_coding_test"))
    prepare.add_argument("--output-dir", type=Path, default=Path("artifacts/prepared"))
    prepare.add_argument("--max-events", type=int, default=256)
    prepare.add_argument("--min-token-count", type=int, default=3)
    prepare.add_argument("--n-value-bins", type=int, default=8)
    train_parser = commands.add_parser("train", help="Train a PyTorch model; select checkpoint on validation mAP")
    train_parser.add_argument("--config", type=Path, required=True)
    predict_parser = commands.add_parser("predict", help="Reload weights and reproduce predictions")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared"))
    predict_parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    predict_parser.add_argument("--device", default="auto")
    predict_parser.add_argument("--split", choices=("val", "test"), default="test")
    validate_parser = commands.add_parser("validate-submission", help="Validate exact test coverage and probabilities")
    validate_parser.add_argument("--predictions", type=Path, default=Path("predictions.csv"))
    validate_parser.add_argument("--data-dir", type=Path, default=Path("research_intern_coding_test"))
    export_parser = commands.add_parser("export", help="Package weights and sources locally for a later Hub upload")
    export_parser.add_argument("--run-dir", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_bundle"))
    experiment_prepare = commands.add_parser("prepare-experiment", help="Prepare common-encoder length variants or full-history summaries")
    experiment_prepare.add_argument("--mode", choices=("length", "summary"), required=True)
    experiment_prepare.add_argument("--data-dir", type=Path, default=Path("research_intern_coding_test"))
    experiment_prepare.add_argument("--output-root", type=Path, default=Path("artifacts/optimization"))
    experiment_prepare.add_argument("--min-token-count", type=int, default=3)
    experiment_prepare.add_argument("--n-value-bins", type=int, default=8)
    experiment_prepare.add_argument("--summary-clip", type=float, default=None)
    ensemble_predict = commands.add_parser("predict-ensemble", help="Reproduce an exported equal-weight ensemble")
    ensemble_predict.add_argument("--manifest", type=Path, required=True)
    ensemble_predict.add_argument("--prepared-root", type=Path, default=Path("artifacts/optimization"))
    ensemble_predict.add_argument("--output", type=Path, default=Path("predictions.csv"))
    ensemble_predict.add_argument("--device", default="auto")
    ensemble_predict.add_argument("--split", choices=("val", "test"), default="test")
    ensemble_prepare = commands.add_parser("prepare-ensemble", help="Rebuild and verify every encoder recipe in an exported ensemble")
    ensemble_prepare.add_argument("--manifest", type=Path, required=True)
    ensemble_prepare.add_argument("--data-dir", type=Path, default=Path("research_intern_coding_test"))
    ensemble_prepare.add_argument("--output-root", type=Path, default=Path("artifacts/optimization"))
    visits_prepare = commands.add_parser("prepare-visits", help="Prepare frozen-encoder raw/date controls for hierarchical models")
    visits_prepare.add_argument("--data-dir", type=Path, default=Path("research_intern_coding_test"))
    visits_prepare.add_argument("--encoder-dir", type=Path, default=None)
    visits_prepare.add_argument("--output-root", type=Path, default=Path("artifacts/hierarchical"))
    visits_prepare.add_argument("--max-units", type=int, default=512)
    visits_prepare.add_argument("--max-unit-events", type=int, default=128)
    visits_prepare.add_argument("--raw-event-limit", type=int, default=511)
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare_dataset
        result = prepare_dataset(args.data_dir, args.output_dir, max_events=args.max_events,
                                 min_token_count=args.min_token_count, n_value_bins=args.n_value_bins)
    elif args.command == "prepare-experiment":
        from .experiments_data import prepare_length_ablation, prepare_summary_dataset
        if args.mode == "length":
            result = prepare_length_ablation(args.data_dir, args.output_root, (256, 512, 1024),
                                            min_token_count=args.min_token_count, n_value_bins=args.n_value_bins)
        else:
            cache_name = "prepared_summary" if args.summary_clip is None else f"prepared_summary_clip{args.summary_clip:g}"
            result = prepare_summary_dataset(args.data_dir, args.output_root / cache_name,
                                             min_token_count=args.min_token_count, n_value_bins=args.n_value_bins,
                                             standardized_clip=args.summary_clip)
    elif args.command == "predict-ensemble":
        from .ensemble import predict_ensemble
        from .ensemble_cli import load_members
        result = predict_ensemble(load_members(args.manifest, args.prepared_root), args.output,
                                  device=args.device, split=args.split)
    elif args.command == "prepare-ensemble":
        from .ensemble_cli import prepare_inputs
        result = prepare_inputs(args.manifest, args.data_dir, args.output_root)
    elif args.command == "prepare-visits":
        from .visit_data import prepare_visit_datasets
        encoder = args.encoder_dir
        if encoder is None:
            from .experiments_data import prepare_length_ablation
            encoder_root = args.output_root / "base_encoder"
            encoder = encoder_root / "prepared_l512"
            if not (encoder / "preprocessing.json").is_file():
                prepare_length_ablation(args.data_dir, encoder_root, lengths=(512,))
        result = prepare_visit_datasets(args.data_dir, encoder, args.output_root,
                                        args.max_units, args.max_unit_events, args.raw_event_limit)
    elif args.command == "train":
        from .train import train
        result = train(json.loads(args.config.read_text(encoding="utf-8")))
    elif args.command == "predict":
        from .train import predict
        result = predict(args.checkpoint, args.prepared_dir, args.output, args.device, args.split)
    elif args.command == "export":
        from .export import export_model
        result = export_model(args.run_dir, args.output_dir)
    else:
        import csv
        from .submission import validate_predictions
        with (args.data_dir / "target_conditions.csv").open(encoding="utf-8-sig", newline="") as handle:
            codes = [row["CODE"] for row in csv.DictReader(handle)]
        with (args.data_dir / "test_anchors.csv").open(encoding="utf-8-sig", newline="") as handle:
            patients = [row["Id"] for row in csv.DictReader(handle)]
        result = validate_predictions(args.predictions, codes, patients)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
