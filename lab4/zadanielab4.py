import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras as k


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = k.models.Sequential([
        k.layers.Conv2D(24, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        k.layers.MaxPool2D(pool_size=(2, 2)),
        k.layers.Conv2D(36, (3, 3), activation="relu"),
        k.layers.MaxPool2D(pool_size=(2, 2)),
        k.layers.Flatten(),
        k.layers.Dense(900, activation="relu"),
        k.layers.Dense(128, activation="relu"),
        k.layers.Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("test_accuracy:", float(test_acc))
