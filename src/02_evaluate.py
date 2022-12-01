import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

model = tf.keras.models.load_model("./data/model/cnn_classifier")

(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

pred = model.predict(X_test)

y_pred = list()
for prediction in pred:
    y_pred.append(np.argmax(prediction))

acc = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="micro")
recall = recall_score(y_test, y_pred, average="micro")


with open("./data/model/metrics.json", "w") as f:
    json.dump(
        {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "precision": precision,
            "recall": recall,
        },
        f,
    )
