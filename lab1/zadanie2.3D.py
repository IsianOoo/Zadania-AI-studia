import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron:
    def __init__(self, n, bias=True):
        self.w = np.random.randn(n)
        self.b = float(np.random.randn()) if bias else 0.0

    def predict(self, x):
        s = float(np.dot(self.w, x) + self.b)
        return 1 if s > 0 else 0

    def train(self, xx, d, eta, tol):
        xx = np.asarray(xx, dtype=float)
        d = np.asarray(d, dtype=int)

        n = len(d)
        if n == 0:
            return

        max_epochs = 200000
        for _ in range(max_epochs):
            idx = np.random.permutation(n)
            correct = 0
            for i in idx:
                x = xx[i]
                y_t = d[i]
                y = self.predict(x)
                if y != y_t:
                    self.w = self.w + eta * (y_t - y) * x
                    self.b = self.b + eta * (y_t - y)
                else:
                    correct += 1
            acc = correct / n
            if acc >= tol:
                break

    def evaluate_test(self, xx, d):
        xx = np.asarray(xx, dtype=float)
        d = np.asarray(d, dtype=int)
        preds = np.array([self.predict(x) for x in xx], dtype=int)
        err = float(np.mean(preds != d)) if len(d) else 0.0
        return err, preds


if __name__ == "__main__":
    df = pd.read_csv("lab1/3D.csv", skiprows=1, delimiter=";", names=["X1", "X2", "X3", "L"], decimal=",")
    X = df[["X1", "X2", "X3"]].to_numpy(dtype=float)
    y = df["L"].to_numpy(dtype=int)

    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test, y_test = X[idx[split:]], y[idx[split:]]

    p = Perceptron(3)
    p.train(X_train, y_train, eta=0.01, tol=1.0)

    train_err, _ = p.evaluate_test(X_train, y_train)
    test_err, _ = p.evaluate_test(X_test, y_test)

    print("Zadanie 2 (3D)")
    print("blad train:", train_err)
    print("blad test:", test_err)
    print("w:", p.w)
    print("b:", p.b)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, marker="o")
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, marker="x")

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    X1, X2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 15),
        np.linspace(x2_min, x2_max, 15)
    )

    if abs(p.w[2]) > 1e-12:
        X3 = -(p.w[0] * X1 + p.w[1] * X2 + p.b) / p.w[2]
        ax.plot_surface(X1, X2, X3, alpha=0.3)
    else:
        print("Nie da sie narysowac plaszczyzny jako X3=..., bo w3 ~ 0")

    ax.set_title("3D: train (o) i test (x) + plaszczyzna separujaca")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3")
    plt.show()
