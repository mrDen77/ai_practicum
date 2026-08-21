import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return x * sigmoid(beta * x)

# Диапазон для визуализации
x_vals = np.linspace(-5, 5, 400)
y_vals = swish(x_vals)

plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals)
plt.title("Swish (β=1)")
plt.xlabel("x")
plt.ylabel("Swish(x)")
plt.grid(True)
plt.show()