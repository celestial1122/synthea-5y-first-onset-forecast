"""Inspect history truncation; optionally prepare controlled ablation arrays."""

import argparse
import json
from pathlib import Path

from ehr_forecast.experiments_data import analyze_histories, prepare_length_ablation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("research_intern_coding_test"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/optimization"))
    parser.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--prepare", action="store_true", help="Also create length variants with a common fitted encoder")
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare_length_ablation(args.data_dir, args.output_root, args.lengths)
        print(json.dumps(manifest, indent=2))
        return
    result = analyze_histories(args.data_dir, args.output_root, args.lengths)
    print(json.dumps({split: {"patients": data["patients"], "events": data["full_event_counts"],
                              "truncation": {length: item["truncated_fraction"] for length, item in data["lengths"].items()},
                              "long_256": data["among_long_256"]}
                      for split, data in result["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
