"""Inspect or prepare frozen-encoder date-bag controls for hierarchical models."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ehr_forecast.visit_data import analyze_date_units, prepare_visit_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("research_intern_coding_test"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("artifacts/optimization/prepared_l512"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/hierarchical"))
    parser.add_argument("--max-units", type=int, default=512)
    parser.add_argument("--max-unit-events", type=int, default=128)
    parser.add_argument("--raw-event-limit", type=int, default=511)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    result = (prepare_visit_datasets(args.data_dir, args.encoder_dir, args.output_root,
                                    args.max_units, args.max_unit_events, args.raw_event_limit)
              if args.prepare else analyze_date_units(args.data_dir, args.output_root))
    if args.prepare:
        result = {key: value for key, value in result.items() if key != "date_diagnostics"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
