import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from utils import load_yaml

params = load_yaml(path="params.yaml")

model = tf.keras.models.load_model("data/experiment/cnn_classifier")

(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

pred = model.predict(X_test)

y_pred = list()
for prediction in pred:
    print(prediction)
    y_pred.append(np.argmax(prediction))

df = pd.DataFrame(data={"y_pred": y_pred, "y_true": y_test})
df.to_csv("data/experiment/predictions.csv")

acc = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average=params["evaluate"]["average"])
precision = precision_score(y_test, y_pred, average=params["evaluate"]["average"])
recall = recall_score(y_test, y_pred, average=params["evaluate"]["average"])


with open("data/experiment/metrics.json", "w") as f:
    json.dump(
        {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        },
        f,
    )
