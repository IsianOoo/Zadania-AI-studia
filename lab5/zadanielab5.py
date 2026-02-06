import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def filter_classes_and_limit(X, y, classes, limit_per_class=None, total_limit=None, seed=42):
    set_seed(seed)
    y = y.reshape(-1)
    mask = np.isin(y, classes)
    Xf = X[mask]
    yf = y[mask]

    class_to_new = {c: i for i, c in enumerate(classes)}
    yf_new = np.array([class_to_new[v] for v in yf], dtype=int)

    idx_all = []
    for new_c, orig_c in enumerate(classes):
        idx = np.where(yf_new == new_c)[0]
        np.random.shuffle(idx)
        if limit_per_class is not None:
            idx = idx[:limit_per_class]
        idx_all.append(idx)

    idx_all = np.concatenate(idx_all) if len(idx_all) else np.array([], dtype=int)
    np.random.shuffle(idx_all)

    if total_limit is not None:
        idx_all = idx_all[:total_limit]

    return Xf[idx_all], yf_new[idx_all]


def normalize_train_test(X_train, X_test):
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-7

    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    return X_train_n, X_test_n, mean, std


def build_cnn(num_filters, n_classes, drop_dense=0.5, drop_conv=0.0, lr=1e-3):
    model = k.models.Sequential()
    model.add(k.layers.Conv2D(num_filters, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same"))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same"))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(drop_conv))

    model.add(k.layers.Conv2D(2 * num_filters, (3, 3), activation="relu", padding="same"))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.Conv2D(2 * num_filters, (3, 3), activation="relu", padding="same"))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(drop_conv))

    model.add(k.layers.Conv2D(4 * num_filters, (3, 3), activation="relu", padding="same"))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.Conv2D(4 * num_filters, (3, 3), activation="relu", padding="same"))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(drop_conv))

    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(512, activation="relu"))
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Dropout(drop_dense))
    model.add(k.layers.Dense(n_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=k.optimizers.Adam(learning_rate=lr),
        metrics=["accuracy"],
    )
    return model


def make_datagen(rotation_range, width_shift_range, height_shift_range, zoom_range):
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
    )


def kfold_score_with_aug(X, y_onehot, n_splits, aug_params, num_filters, batch_size, epochs, seed=42):
    set_seed(seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_onehot[train_idx], y_onehot[val_idx]

        model = build_cnn(num_filters=num_filters, n_classes=y_onehot.shape[1])

        datagen = make_datagen(**aug_params)
        datagen.fit(X_tr)

        steps = max(1, int(np.ceil(len(X_tr) / batch_size)))
        history = model.fit(
            datagen.flow(X_tr, y_tr, batch_size=batch_size),
            steps_per_epoch=steps,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=0,
        )

        best_val_acc = float(np.max(history.history.get("val_accuracy", [0.0])))
        accs.append(best_val_acc)

        k.backend.clear_session()

    return float(np.mean(accs)), accs


def random_search_aug(X, y_onehot, trials, n_splits, num_filters, batch_size, epochs, seed=42):
    set_seed(seed)
    best = None
    results = []

    for t in range(trials):
        aug_params = {
            "rotation_range": int(np.random.choice([0, 5, 10, 15, 20])),
            "width_shift_range": float(np.random.choice([0.0, 0.05, 0.1, 0.15])),
            "height_shift_range": float(np.random.choice([0.0, 0.05, 0.1, 0.15])),
            "zoom_range": float(np.random.choice([0.0, 0.05, 0.1, 0.15])),
        }

        mean_acc, fold_accs = kfold_score_with_aug(
            X=X,
            y_onehot=y_onehot,
            n_splits=n_splits,
            aug_params=aug_params,
            num_filters=num_filters,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed + t + 1,
        )

        results.append((mean_acc, aug_params, fold_accs))
        if best is None or mean_acc > best[0]:
            best = (mean_acc, aug_params, fold_accs)

        print(f"Trial {t+1}/{trials} | mean val acc: {mean_acc:.4f} | params: {aug_params} | folds: {[round(a,4) for a in fold_accs]}")

    results.sort(key=lambda x: x[0], reverse=True)
    return best, results


if __name__ == "__main__":
    set_seed(42)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    classes_orig = [0, 1, 2, 3]
    total_train_limit = 2000
    total_test_limit = 800

    X_train_s, y_train_s = filter_classes_and_limit(
        X_train, y_train, classes=classes_orig, total_limit=total_train_limit, seed=42
    )
    X_test_s, y_test_s = filter_classes_and_limit(
        X_test, y_test, classes=classes_orig, total_limit=total_test_limit, seed=43
    )

    X_train_n, X_test_n, mean, std = normalize_train_test(X_train_s, X_test_s)

    n_classes = len(classes_orig)
    y_train_oh = to_categorical(y_train_s, n_classes)
    y_test_oh = to_categorical(y_test_s, n_classes)

    print("Dane:")
    print("X_train:", X_train_n.shape, "y_train:", y_train_oh.shape)
    print("X_test:", X_test_n.shape, "y_test:", y_test_oh.shape)
    print("Klasy (oryginalne):", classes_orig, "-> nowe etykiety 0..", n_classes - 1)
    print()

    trials = 6
    n_splits = 3
    num_filters = 32
    batch_size = 64
    cv_epochs = 3

    print("Random Search + KFold (augmentacja jako hiperparametry):")
    best, all_results = random_search_aug(
        X=X_train_n,
        y_onehot=y_train_oh,
        trials=trials,
        n_splits=n_splits,
        num_filters=num_filters,
        batch_size=batch_size,
        epochs=cv_epochs,
        seed=100,
    )

    best_mean_acc, best_aug, best_folds = best
    print()
    print("Najlepsze parametry augmentacji:")
    print("mean val acc:", round(best_mean_acc, 4))
    print("params:", best_aug)
    print()

    final_epochs = 8
    val_split = 0.2

    idx = np.random.permutation(len(X_train_n))
    split = int((1 - val_split) * len(X_train_n))
    tr_idx, val_idx = idx[:split], idx[split:]

    X_tr, y_tr = X_train_n[tr_idx], y_train_oh[tr_idx]
    X_val, y_val = X_train_n[val_idx], y_train_oh[val_idx]

    model = build_cnn(num_filters=num_filters, n_classes=n_classes)

    datagen = make_datagen(**best_aug)
    datagen.fit(X_tr)

    steps = max(1, int(np.ceil(len(X_tr) / batch_size)))
    history = model.fit(
        datagen.flow(X_tr, y_tr, batch_size=batch_size),
        steps_per_epoch=steps,
        epochs=final_epochs,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test_n, y_test_oh, verbose=0)
    print()
    print("Finalny wynik na tescie:")
    print("test_acc:", float(test_acc))