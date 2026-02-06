import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    df = pd.read_csv("lab1/2D.csv", skiprows=1, delimiter=";", names=["X1", "X2", "L"], decimal=",")

    X = df[["X1", "X2"]].to_numpy(dtype=float)
    y = df["L"].to_numpy(dtype=int)

    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test, y_test = X[idx[split:]], y[idx[split:]]

    p = Perceptron(2)
    p.train(X_train, y_train, eta=0.01, tol=1.0)

    train_err, _ = p.evaluate_test(X_train, y_train)
    test_err, _ = p.evaluate_test(X_test, y_test)

    print("Zadanie 2 (2D)")
    print("blad train:", train_err)
    print("blad test:", test_err)
    print("w:", p.w)
    print("b:", p.b)

    plt.figure()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker="o")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker="x")

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    xs = np.linspace(x_min, x_max, 200)

    if abs(p.w[1]) > 1e-12:
        ys = -(p.w[0] * xs + p.b) / p.w[1]
        plt.plot(xs, ys)
    else:
        if abs(p.w[0]) > 1e-12:
            x0 = -p.b / p.w[0]
            plt.axvline(x0)

    plt.title("2D: train (o) i test (x) + linia separujaca")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
