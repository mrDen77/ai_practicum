import numpy as np
import matplotlib.pyplot as plt
import math

def softmax(z):
    # 1) для стабильности вычитаем максимум
    e_z = np.exp(z - np.max(z))
    # 2) нормируем экспоненты, чтобы сумма = 1
    return e_z / e_z.sum()

# диапазон значений второго логита
x_vals = np.linspace(-5, 5, 400)
# для каждого x считаем только P(class=1) в Softmax([0, x])
p1 = [softmax([0, x])[1] for x in x_vals]

plt.figure(figsize=(6,4))
plt.plot(x_vals, p1)
plt.title("Softmax для двух классов: P(class=1 | [0, x])")
plt.xlabel("x (логит второго класса)")
plt.ylabel("P(class=1)")
plt.grid(True)
plt.show() 