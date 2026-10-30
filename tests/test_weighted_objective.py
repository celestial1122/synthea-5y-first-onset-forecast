import numpy as np
import pytest
import torch

from ehr_forecast.models import build_model
from ehr_forecast.train import predict_probabilities


def test_weighted_bce_posterior_offset_roundtrip(tmp_path):
    probabilities = torch.tensor([.02, .2, .7])
    weights = torch.tensor([8., 3., 1.1])
    tilted = torch.logit(probabilities) + weights.log()
    recovered = torch.sigmoid(tilted - weights.log())
    torch.testing.assert_close(recovered, probabilities)
    config = dict(model_type="summary_linear", vocab_size=12, n_targets=3, n_static=2,
                  max_events=4, prediction_logit_offset=weights.log().tolist())
    model = build_model(config)
    with torch.no_grad():
        model.classification_head.weight.zero_()
        model.classification_head.bias.copy_(tilted)
    checkpoint = tmp_path / "weighted.pt"
    torch.save({"config": config, "weights": model.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    restored = build_model(saved["config"])
    restored.load_state_dict(saved["weights"])
    batch = {"tokens": torch.tensor([[2, 4, 0, 0]]), "time_days": torch.tensor([[0., 10., 0., 0.]]),
             "static": torch.zeros(1, 2), "prior_conditions": torch.tensor([[False, True, False]])}
    actual = predict_probabilities(restored, [batch], torch.device("cpu"))
    np.testing.assert_allclose(actual, [[.02, 0., .7]], rtol=1e-6)


@pytest.mark.parametrize("offset", [[0.], [0., float("nan")], [0., float("inf")]])
def test_invalid_prediction_offsets_rejected(offset):
    with pytest.raises(ValueError, match="one finite value per target"):
        build_model(dict(model_type="summary_linear", vocab_size=12, n_targets=2,
                         n_static=2, max_events=4, prediction_logit_offset=offset))
