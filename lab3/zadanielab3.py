import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z, beta):
    return 1.0 / (1.0 + np.exp(-beta * z))


def d_sigmoid(y, beta):
    return beta * y * (1.0 - y)


def tanh_act(z, beta):
    return np.tanh(beta * z)


def d_tanh(y, beta):
    return beta * (1.0 - y * y)


def forward(x, w_h, w_o, beta):
    h_in = w_h @ x
    h = tanh_act(h_in, beta)
    h_b = np.concatenate(([1.0], h))
    o_in = w_o @ h_b
    y = sigmoid(o_in, beta)
    return float(y[0]), h_b


def ok(y, target):
    return (target == 1 and y > 0.9) or (target == 0 and y < 0.1)


def init_weights():
    w_h = np.random.uniform(-0.5, 0.5, (2, 3))
    w_o = np.random.uniform(-0.5, 0.5, (1, 3))
    return w_h, w_o


def train_online(X, y, lr, beta, max_epochs=100000):
    w_h, w_o = init_weights()
    curve = []

    for epoch in range(max_epochs):
        sse = 0.0
        hits = 0

        for x_i, t in zip(X, y):
            y_hat, h_b = forward(x_i, w_h, w_o, beta)
            e = y_hat - float(t)
            sse += e * e

            if ok(y_hat, t):
                hits += 1

            delta_out = e * d_sigmoid(y_hat, beta)
            grad_o = delta_out * h_b

            h_no_bias = h_b[1:]
            delta_h = (delta_out * w_o[0, 1:]) * d_tanh(h_no_bias, beta)
            grad_h = np.outer(delta_h, x_i)

            w_o -= lr * grad_o
            w_h -= lr * grad_h

        curve.append(0.5 * sse / len(X))
        if hits == len(X):
            print(f"Koniec (próbka) w epoce: {epoch}")
            break

    return curve


def train_batch(X, y, lr, beta, max_epochs=100000):
    w_h, w_o = init_weights()
    curve = []

    for epoch in range(max_epochs):
        sse = 0.0
        hits = 0

        g_h = np.zeros_like(w_h)
        g_o = np.zeros_like(w_o)

        for x_i, t in zip(X, y):
            y_hat, h_b = forward(x_i, w_h, w_o, beta)
            e = y_hat - float(t)
            sse += e * e

            if ok(y_hat, t):
                hits += 1

            delta_out = e * d_sigmoid(y_hat, beta)
            g_o += delta_out * h_b

            h_no_bias = h_b[1:]
            delta_h = (delta_out * w_o[0, 1:]) * d_tanh(h_no_bias, beta)
            g_h += np.outer(delta_h, x_i)

        w_o -= lr * g_o
        w_h -= lr * g_h

        curve.append(0.5 * sse / len(X))
        if hits == len(X):
            print(f"Koniec (epoka) w epoce: {epoch}")
            break

    return curve


if __name__ == "__main__":
    X = np.array([
        [1, -1, -1],
        [1, -1,  1],
        [1,  1, -1],
        [1,  1,  1]
    ], dtype=float)

    y = np.array([0, 1, 1, 0], dtype=int)

    eta = 0.1
    beta = 1.0

    e1 = train_online(X, y, eta, beta)
    e2 = train_batch(X, y, eta, beta)

    plt.figure(figsize=(10, 6))
    plt.plot(e1, label="Metoda próbki")
    plt.plot(e2, label="Metoda epoki", linestyle="--")
    plt.xlabel("Epoka")
    plt.ylabel("Błąd MSE")
    plt.legend()
    plt.grid(True)
    plt.show()
