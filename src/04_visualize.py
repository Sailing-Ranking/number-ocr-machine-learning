import json

import matplotlib.pyplot as plt

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
