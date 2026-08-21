import numpy as np
import matplotlib.pyplot as plt

def relu(value):
    return np.maximum(0, value)

def neuron(x, weight, bias):
    z = x*weight + bias
    return relu(z) 
    # Допишите код как реализацию нейрона с весами и смещением

# диапазон входов
x = np.linspace(-10, 10, 400)
bias = 0
weights = [0.5, 1.0, 2.0]

plt.figure(figsize=(8, 5))
for w in weights:
    y = neuron(x, w, bias)
    plt.plot(x, y, label=f"weight = {w}")

plt.title("Влияние веса (bias = 0)")
plt.xlabel("x")
plt.ylabel("y = ReLU(w·x)")
plt.legend()
plt.grid(True)
plt.show()