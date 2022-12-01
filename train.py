import numpy as np

import tensorflow as tf

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


model = tf.keras.Sequential(layers=[
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.AveragePooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.AveragePooling2D((2, 2)),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

history = model.fit(X_train, y_train, validation_split=.25, epochs=1)


# Evaluation
pred = model.predict(X_test)

y_pred = list()
for prediction in pred:
    y_pred.append(np.argmax(prediction))

acc = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="micro")
recall = recall_score(y_test, y_pred, average="micro")

print(acc, bacc, precision, recall)

with open("results.txt", 'w') as f:
    f.write(f"Accuracy: {acc}\nBalanced Accuracy: {bacc}\nPrecision: {precision}\nRecall: {recall}\n")

# Accuracy vs Validation Accuracy
plt.figure(figsize=(7, 7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("acc_vs_val_acc.png", dpi=120)
plt.close()

# Loss vs Validation Loss
plt.figure(figsize=(7, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("loss_vs_val_loss.png", dpi=120)
plt.close()