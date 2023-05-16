import numpy as np
import matplotlib.pyplot as plt

P = np.arange(-2, 2.1, 0.1).reshape((1, 41))
T = P * P + (np.random.rand(P.shape[0]) - 0.5)

S1 = 100
W1 = np.random.rand(S1, 1) - 0.5
B1 = np.random.rand(S1, 1) - 0.5
W2 = np.random.rand(1, S1) - 0.5
B2 = np.random.rand(1, 1) - 0.5
lr = 0.001


for epoch in range(200):
    A1 = np.tanh(W1 @ P + B1 * np.ones(P.shape[0]))
    A2 = W2 @ A1 + B2
    E2 = T - A2
    E1 = W2.T @ E2
    dW2 = lr * E2 @ A1.T
    dB2 = lr * E2 * np.ones(E2.shape[0]).T
    dW1 = lr * (1 - np.multiply(A1, A1)) * E1 @ P.T
    dB1 = lr * (1 - np.multiply(A1, A1)) * E1 * np.ones(P.shape[0]).T

    W2 = W2 + dW2
    B2 = B2 + dB2
    W1 = W1 + dW1
    B1 = B1 + dB1
    if np.mod(epoch, 5) == 0:
        plt.clf()
        plt.plot(P, T, 'r*')
        plt.plot(P, A2, 'g.')
        plt.show()
