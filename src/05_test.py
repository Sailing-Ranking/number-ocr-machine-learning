import pytest
import tensorflow as tf

from utils import load_yaml

model = tf.keras.models.load_model("data/experiment/cnn_classifier")
metrics = load_yaml(path="data/experiment/metrics.json")


@pytest.mark.parametrize("metric", metrics.values())
def test_metrics(metric):
    assert metric > 0.9
