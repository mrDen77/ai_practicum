import numpy as np
import matplotlib.pyplot as plt
import math

def leaky_relu(x, alpha=0.03):
    return np.where(x >= 0, x, alpha * x)

x_vals = np.linspace(-5, 5, 400)
y_vals = leaky_relu(x_vals)
plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals)
plt.title("Leaky ReLU (α=0.01)")
plt.xlabel("x")
plt.ylabel("LeakyReLU(x)")
plt.grid(True)
plt.show() 