import numpy as np
import matplotlib.pyplot as plt
import math

def relu(value):
    return np.maximum(0, value)

def neuron(x, weight, bias):
    z = x*weight + bias
    return relu(z) 
    # Допишите код как реализацию нейрона с весами и смещением

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

x_vals = [i * 0.1 for i in range(-50, 51)] # Генерируем значения от -5 до 5 с шагом 0.1
y_vals = [sigmoid(x) for x in x_vals] # Применяем функцию sigmoid к каждому значению

plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals)
plt.title("Sigmoid")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.grid(True)
plt.show()