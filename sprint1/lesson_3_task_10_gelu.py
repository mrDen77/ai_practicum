import numpy as np
import matplotlib.pyplot as plt
import math

def gelu(x: np.ndarray) -> np.ndarray:
    # Приближение через tanh
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715 * x**3)))

# Диапазон для визуализации
x_vals = np.linspace(-5, 5, 400)
y_vals = gelu(x_vals)

plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals)
plt.title("GELU (приближение через tanh)")
plt.xlabel("x")
plt.ylabel("GELU(x)")
plt.grid(True)
plt.show()