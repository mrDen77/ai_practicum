import numpy as np
import matplotlib.pyplot as plt

def relu(value):
    return np.maximum(0, value)

def neuron(x, weight, bias):
    z = x*weight + bias
    return relu(z) 
    # Допишите код как реализацию нейрона с весами и смещением

# Генерируем входные значения с помощью функции linspace в диапазоне от -5 до 5 с помощью numpy
x_vals = np.linspace(-5, 5, 200)

v_bias = -3
# Вычисляем выход нейрона со смещением (weight=1.0, bias=3)
y_vals = neuron(x_vals, weight=1.0, bias=v_bias)

# Строим график
plt.plot(x_vals, y_vals)
plt.title('Работа нейрона со смещением bias={v_bias}')
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.xlabel('x')
plt.ylabel('y = neuron(x, 1.0, 0)')
plt.show()