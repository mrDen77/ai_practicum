import numpy as np
import matplotlib.pyplot as plt
import math

def relu(x):
    return np.maximum(0, x)

x_vals = np.linspace(-5, 5, 400)
y_vals = relu(x_vals)
plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals)
plt.title("ReLU")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid(True)
plt.show() 