from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("/data/image-3.png").convert('L')
img = img.resize((28, 28))
pixels = np.array(img) / 255.0
inputs = pixels.flatten()

np.random.seed(42)

weights = np.random.randn(len(inputs))
bias = np.random.randn()    


print("Количество входов:", len(inputs))
print("Форма weights:", weights.shape)
print("Пример первых 5 весов:", weights[:5])
print("Bias:", bias)