import numpy as np
import matplotlib.pyplot as plt
import math

# 1) Входные данные
x = np.linspace(-3, 3, 300)

# 2) Реализация функций активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh_act(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)


# 3) Вычисление значений
y_sig = sigmoid(x)
y_tanh = tanh_act(x)
y_relu = relu(x)
y_leaky = leaky_relu(x)

# 4) Визуализация
plt.figure(figsize=(8, 5))
plt.plot(x, y_sig, label='Sigmoid')
plt.plot(x, y_tanh, label='Tanh')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_leaky, label='Leaky ReLU')
plt.title('Кривые основных функций активации')
plt.xlabel('x')
plt.ylabel('activation(x)')
plt.legend()
plt.grid(True)
plt.show()