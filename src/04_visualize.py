import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


df_predictions = pd.read_csv("data/experiment/predictions.csv")

y_pred = df_predictions["y_pred"].to_numpy()
y_true = df_predictions["y_true"].to_numpy()

confusion = confusion_matrix(y_true, y_pred)

with open("data/experiment/history.json") as f:
    history = json.load(f)

    # Accuracy vs Validation Accuracy
    plt.figure(figsize=(7, 7))
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("data/experiment/plots/acc_vs_val_acc.png", dpi=120)
    plt.close()

    # Loss vs Validation Loss
    plt.figure(figsize=(7, 7))
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("data/experiment/plots/loss_vs_val_loss.png", dpi=120)
    plt.close()

    plot_confusion_matrix(confusion, figsize=(7, 7), show_normed=True)
    plt.savefig("data/experiment/plots/confusion_matrix.png", dpi=120)
    plt.close()
