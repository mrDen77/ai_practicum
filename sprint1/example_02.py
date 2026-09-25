import os

import torch
from torchvision import models, transforms
from PIL import Image
import requests

imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
classes = requests.get(imagenet_labels_url).json() 

# Загрузка предобученной модели ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval() # Переводит модель в режим инференса (inference)

# Создаём конвейер преобразований
transform = transforms.Compose([
    transforms.Resize(256),           # Изменение размера: меньшая сторона 256 пикселей
    transforms.CenterCrop(224),        # Центрированная обрезка до 224x224
    transforms.ToTensor(),             # Преобразование изображения в тензор
    transforms.Normalize(              # Стандартизация по статистике ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Загрузка и преобразование изображения (уже описано на этапе препроцессинга)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image = Image.open(os.path.join(BASE_DIR, "wqa.jpg")).convert('RGB')

# Даже если у вас одно изображение, модель ожидает вход размера [batch_size, channels, height, width]. 
# Для этого используется метод unsqueeze(0) чтобы добавить измерение батча
tensor_image = transform(image)
image_tensor = tensor_image.unsqueeze(0)

with torch.no_grad():
    outputs = model(image_tensor)  # Получаем "сырые" выходы модели
    _, predicted_idx = torch.max(outputs, 1)  # Находим индекс класса с наибольшей вероятностью
    predicted_label = classes[predicted_idx.item()]  # Преобразуем индекс в читаемую метку
    
print(f"Предсказанный класс: {predicted_label}")