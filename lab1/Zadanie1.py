import numpy as np

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
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([0, 0, 0, 1], dtype=int)

    p = Perceptron(2)
    p.train(X, y, eta=0.1, tol=1.0)

    err, preds = p.evaluate_test(X, y)

    print("Zadanie 1")
    print("w:", p.w)
    print("b:", p.b)
    print("pred:", preds.tolist())
    print("blad:", err)
