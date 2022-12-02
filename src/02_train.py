import json

import tensorflow as tf

from utils import load_params

params = load_params()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


model = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Conv2D(
            params["model"]["filters"],
            params["model"]["kernel_size"],
            activation=params["model"]["activation"],
            kernel_initializer=params["model"]["kernel_initializer"],
            input_shape=params["model"]["input_shape"],
        ),
        tf.keras.layers.AveragePooling2D(params["model"]["pooling"]),
        tf.keras.layers.Conv2D(
            params["model"]["filters"] * 2,
            params["model"]["kernel_size"],
            activation=params["model"]["activation"],
        ),
        tf.keras.layers.AveragePooling2D(params["model"]["pooling"]),
        tf.keras.layers.Conv2D(
            params["model"]["filters"] * 2,
            params["model"]["kernel_size"],
            activation=params["model"]["activation"],
        ),
        tf.keras.layers.AveragePooling2D(params["model"]["pooling"]),
        tf.keras.layers.Dropout(params["model"]["dropout"]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(params["model"]["filters"] * 4, activation="relu"),
        tf.keras.layers.Dense(params["model"]["filters"] * 2, activation="relu"),
        tf.keras.layers.Dense(10, activation=params["model"]["out_activation"]),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[params["compile"]["metrics"]],
)

history = model.fit(
    X_train, y_train, validation_split=params["train"]["validation_split"], epochs=2
)


model.save("data/experiment/cnn_classifier")

with open("data/experiment/history.json", "w") as f:
    json.dump(history.history, f)
