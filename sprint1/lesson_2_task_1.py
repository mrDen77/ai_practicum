from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Загружаем изображение и преобразуем его в чёрно-белое
img = Image.open("/data/image-3.png").convert('L')
img = img.resize((28, 28))

# Нормализуем пиксели в диапазон от 0 до 1
pixels = np.array(img) / 255.0

# Преобразуем в одномерный вектор
inputs = pixels.flatten()

print("Количество входов нейрона:", len(inputs))
print("Входные данные нейрона:", inputs.tolist())

# Отобразим изображение
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()