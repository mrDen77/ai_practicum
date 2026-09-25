import os

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Читаем изображение
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image = Image.open(os.path.join(BASE_DIR, "IMG-20240606-WA0000.jpg"))

resize_transform = transforms.Resize(256)
resized_image = resize_transform(image)
center_crop = transforms.CenterCrop(224)
cropped_image = center_crop(resized_image)
to_tensor = transforms.ToTensor()
tensor_image = to_tensor(cropped_image)

# Применяем стандартизацию по статистике ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalized_tensor = normalize(tensor_image)

print("Нормализация прошла успешно!")