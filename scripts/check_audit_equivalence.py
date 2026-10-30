"""Real-input CPU equivalence gates, without training performance selection."""

from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ehr_forecast.experimental_bridge import ControlledTimelineTransformer
from ehr_forecast.models import TimelineTransformer
from ehr_forecast.train import seed_everything, state_fingerprint, write_json


def main():
    root = ROOT / "artifacts/experimental_audit"
    prepared = ROOT / "artifacts/optimization/prepared_l512"
    metadata = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    with np.load(prepared / "train.npz", allow_pickle=False) as arrays:
        tokens = torch.from_numpy(arrays["tokens"][:4]).long()
        times = torch.from_numpy(arrays["time_days"][:4]).float()
        static = torch.from_numpy(arrays["static"][:4]).float()
        labels = torch.from_numpy(arrays["labels"][:4]).float()
        eligible = ~torch.from_numpy(arrays["prior_conditions"][:4]).bool()
    config = dict(vocab_size=metadata["vocab_size"], n_targets=len(metadata["target_codes"]),
                  n_static=metadata["n_static"], max_events=1024, d_model=128, n_heads=4,
                  n_layers=3, dropout=.15)
    rows = []
    for seed in (17, 42, 73):
        seed_everything(seed)
        original = TimelineTransformer(**config).eval()
        initial_state = {key: tensor.clone() for key, tensor in original.state_dict().items()}
        rng = torch.get_rng_state().clone()
        seed_everything(seed)
        bridge = ControlledTimelineTransformer(**config).eval()
        if not torch.equal(rng, torch.get_rng_state()):
            raise AssertionError("Legacy adapter changes initial RNG state")
        if any(not torch.equal(initial_state[key], tensor) for key, tensor in bridge.state_dict().items()):
            raise AssertionError("Legacy adapter changes initial parameters")
        first, second = original(tokens, times, static), bridge(tokens.unsqueeze(-1), times, static)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        losses = [(criterion(value, labels) * eligible).sum() / eligible.sum() for value in (first, second)]
        losses[0].backward()
        losses[1].backward()
        for (name_a, p), (name_b, q) in zip(original.named_parameters(), bridge.named_parameters()):
            if name_a != name_b or (p.grad is None) != (q.grad is None):
                raise AssertionError("Gradient parameter coverage differs")
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        for model in (original, bridge):
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01).step()
        if state_fingerprint(original.state_dict()) != state_fingerprint(bridge.state_dict()):
            raise AssertionError("Adapter changes one-step AdamW result")

        # Capacity is a functional no-op if all used position rows are copied.
        smaller = TimelineTransformer(**{**config, "max_events": 512}).eval()
        copied = dict(original.state_dict())
        copied["position_embedding.weight"] = copied["position_embedding.weight"][:512].clone()
        smaller.load_state_dict(copied)
        with torch.no_grad():
            capacity_error = float((original(tokens, times, static) - smaller(tokens, times, static)).abs().max())
        if capacity_error != 0:
            raise AssertionError("Copied used position rows failed the capacity identity")

        common_states, rngs = [], []
        for attention in ("causal", "bidirectional"):
            for readout in ("legacy", "pooled_mlp"):
                seed_everything(seed)
                model = ControlledTimelineTransformer(**config, attention_mode=attention, readout_mode=readout)
                state = {key: value for key, value in model.state_dict().items()
                         if not key.startswith(("classification_head.", "readout."))}
                common_states.append(state_fingerprint(state))
                rngs.append(torch.get_rng_state().clone())
        if len(set(common_states)) != 1 or any(not torch.equal(rngs[0], value) for value in rngs[1:]):
            raise AssertionError("Factorial cells do not share initialization/RNG")
        rows.append({"seed": seed, "adapter_logit_loss_grad_and_step_identical": True,
                     "position_capacity_max_error": capacity_error,
                     "four_cell_common_state_sha256": common_states[0], "four_cell_initial_rng_identical": True})
    report = {"passed": True, "device": "CPU", "patients": 4, "seeds": rows,
              "scope": "Function/gradient/step/initialization identity only. Different attention/dropout shapes can diverge during training."}
    write_json(root / "gate_b_equivalence.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
