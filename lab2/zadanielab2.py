import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

try:
    from PIL import Image
    PIL_OK = True
except:
    PIL_OK = False


class Perceptron:
    def __init__(self, n, bias=True):
        self.w = np.random.randn(n).astype(np.float32)
        self.b = float(np.random.randn()) if bias else 0.0

    def predict(self, x):
        s = float(np.dot(self.w, x) + self.b)
        return 1 if s > 0 else 0

    def train(self, xx, d, eta=0.01, epochs=30, tol=0.0, shuffle=True, verbose=True):
        xx = np.asarray(xx, dtype=np.float32)
        d = np.asarray(d, dtype=np.int32)

        n = len(d)
        errors = []

        for ep in range(1, epochs + 1):
            if shuffle:
                idx = np.random.permutation(n)
                xx_ep = xx[idx]
                d_ep = d[idx]
            else:
                xx_ep = xx
                d_ep = d

            wrong = 0
            for x, y_t in zip(xx_ep, d_ep):
                y = self.predict(x)
                if y != y_t:
                    self.w = self.w + eta * (y_t - y) * x
                    self.b = self.b + eta * (y_t - y)
                    wrong += 1

            err = wrong / n
            errors.append(err)
            if verbose:
                print(f"Epoka {ep}: blad = {err:.4f}")

            if err <= tol:
                break

        return errors

    def evaluate_test(self, xx, d):
        xx = np.asarray(xx, dtype=np.float32)
        d = np.asarray(d, dtype=np.int32)
        preds = np.array([self.predict(x) for x in xx], dtype=np.int32)
        err = float(np.mean(preds != d)) if len(d) else 0.0
        return err, preds


def load_mnist_csv(path):
    df = pd.read_csv(path, header=None)
    data = df.to_numpy()
    y = data[:, 0].astype(np.int32)
    X = data[:, 1:].astype(np.float32)
    return X, y


def filter_two_digits(X, y, d0, d1):
    mask = (y == d0) | (y == d1)
    Xf = X[mask]
    yf = y[mask]
    yb = (yf == d1).astype(np.int32)
    return Xf, yb


def show_examples(X, y_bin, d0, d1, n=8):
    plt.figure()
    idx = np.random.permutation(len(X))[:n]
    for i, j in enumerate(idx, 1):
        img = X[j].reshape(28, 28)
        label = d1 if y_bin[j] == 1 else d0
        plt.subplot(2, (n + 1) // 2, i)
        plt.imshow(img, cmap="gray")
        plt.title(str(label))
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def load_custom_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    x = arr.reshape(-1).astype(np.float32)
    return x, arr


if __name__ == "__main__":
    np.random.seed(42)

    train_path = "mnist_train.csv"
    test_path = "mnist_test.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Brak plikow mnist_train.csv / mnist_test.csv w tym folderze.")
        raise SystemExit

    X_train, y_train = load_mnist_csv(train_path)
    X_test, y_test = load_mnist_csv(test_path)

    d0, d1 = 0, 1

    X_train_f, y_train_b = filter_two_digits(X_train, y_train, d0, d1)
    X_test_f, y_test_b = filter_two_digits(X_test, y_test, d0, d1)

    print("Wybrane cyfry:", d0, "vs", d1)
    print("Train:", X_train_f.shape, y_train_b.shape)
    print("Test:", X_test_f.shape, y_test_b.shape)
    print()

    show_examples(X_train_f, y_train_b, d0, d1, n=8)

    X_train_n = X_train_f / 255.0
    X_test_n = X_test_f / 255.0

    p = Perceptron(n=X_train_n.shape[1])

    errors = p.train(X_train_n, y_train_b, eta=0.01, epochs=30, tol=0.0, verbose=True)

    plt.figure()
    plt.plot(range(1, len(errors) + 1), errors)
    plt.xlabel("Epoka")
    plt.ylabel("Blad")
    plt.title("Blad po epokach")
    plt.show()

    test_err, y_pred = p.evaluate_test(X_test_n, y_test_b)
    wrong = int(test_err * len(y_test_b))

    print()
    print("Blednie sklasyfikowane probki (test):", wrong, "/", len(y_test_b))
    print("Blad test:", test_err)

    cm = confusion_matrix(y_test_b, y_pred)
    print()
    print("Confusion matrix:")
    print(cm)

    acc = accuracy_score(y_test_b, y_pred)
    prec = precision_score(y_test_b, y_pred, zero_division=0)
    rec = recall_score(y_test_b, y_pred, zero_division=0)
    f1 = f1_score(y_test_b, y_pred, zero_division=0)

    print()
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)

    custom_path = "my_digit.png"
    if PIL_OK and os.path.exists(custom_path):
        x_custom, img_custom = load_custom_image(custom_path)
        pred = p.predict(x_custom)

        print()
        print("Test obrazka:", custom_path)
        print("Predykcja binarna:", pred, "(1 oznacza cyfre", d1, ", 0 oznacza cyfre", d0, ")")

        plt.figure()
        plt.imshow(img_custom, cmap="gray")
        plt.title(f"Pred: {pred} (1={d1}, 0={d0})")
        plt.axis("off")
        plt.show()
    else:
        print()
        print("Aby zrobic pkt 10: wrzuc obrazek 'my_digit.png' do folderu (28x28 lub dowolny, zostanie przeskalowany).")
        if not PIL_OK:
            print("Brak PIL. Zainstaluj: pip install pillow")
