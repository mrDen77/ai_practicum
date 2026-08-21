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
weight = 1.0
biases = [-5, 0, 5]

plt.figure(figsize=(8, 5))
for b in biases:
    y = neuron(x, weight, b)
    plt.plot(x, y, label=f"bias = {b}")

plt.title("Влияние смещения (weight = 1.0)")
plt.xlabel("x")
plt.ylabel("y = ReLU(x + b)")
plt.legend()
plt.grid(True)
plt.show()