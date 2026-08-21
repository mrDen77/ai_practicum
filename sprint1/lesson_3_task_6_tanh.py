import numpy as np
import matplotlib.pyplot as plt
import math

def tanh(x: float) -> float:
    return math.tanh(x)

x_vals = [i * 0.1 for i in range(-50, 51)]
y_vals = [tanh(x) for x in x_vals]
plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals)
plt.title("Tanh")
plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.grid(True)
plt.show() 