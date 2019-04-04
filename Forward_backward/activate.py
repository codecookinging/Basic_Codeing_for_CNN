import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    A = 1/(1+np.exp(-x))
    return A, x


def relu(x):

    A = np.maximum(0, x)
    return A, x


def leaky_relu(x):
    A = np.maximum(0.01*x, x)
    return A, x


def tanh(x):
    A = np.tanh(x)
    return A, x

x = np.linspace(-10, 10, 100)
A_sigmoid, x = sigmoid(x)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x, A_sigmoid, label="function")
plt.plot(x, A_sigmoid*(1-A_sigmoid), label="Derivation")
plt.legend(loc="upper left")

plt.xlabel("x")
plt.ylabel(r"$\frac{1}{1 + e^{-x}}$")

A_relu, x = relu(x)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 2)
plt.plot(x, A_relu, label="function")
plt.xlabel("x")
plt.ylabel(r"$max\{0, z\}$")

A_tanh, x = tanh(x)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 3)
plt.plot(x, A_tanh, label="function")
plt.xlabel("x")
plt.ylabel(r"$max\{0, z\}$")

plt.tight_layout()
plt.show()









