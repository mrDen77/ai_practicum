
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) 
img = Image.open("/data/image-3.png").convert('L')
img = img.resize((28, 28))
pixels = np.array(img) / 255.0
weights = np.random.randn(28 * 28)
bias = np.random.randn()

def relu(x): 
    return np.maximum(0, x)


def pixelwise_neuron(pixels, weights, bias):
    flat = pixels.flatten()
    activated = relu(flat * weights + bias)
    return activated.reshape(pixels.shape)

processed = pixelwise_neuron(pixels, weights, bias)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(pixels, cmap='gray')
axs[0].set_title('Оригинальное изображение')
axs[0].axis('off')

axs[1].imshow(processed, cmap='gray')
axs[1].set_title('После одного нейрона')
axs[1].axis('off')

plt.show()