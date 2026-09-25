import os, torch
from torchvision import models, transforms
from PIL import Image, ImageFilter
import requests
import matplotlib.pyplot as plt

imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
classes = requests.get(imagenet_labels_url).json()

# Загрузка предобученной модели ResNet и перевод её в режим инференса
model = models.resnet18(pretrained=True) 
model.eval()


# Определение преобразований для изображения
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_top3(image):
    image_tensor = transform(image).unsqueeze(0)  # Добавляем размерность батча
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top3_probs, top3_indices = torch.topk(probabilities, 3)
    
    results = []
    for i in range(3):
        label = classes[top3_indices[i].item()]
        prob = top3_probs[i].item() * 100  # Проценты
        results.append(f"{label}: {prob:.2f}%")
    
    return results

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image = Image.open(os.path.join(BASE_DIR, "bag.jpg")).convert('RGB')

# Добавляем размытие с помощью фильтра GaussianBlur (radius=10 задаёт сильное размытие)
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))

# Сравниваем предсказания для оригинального изображения и изображения с шумом
original_pred = predict_top3(image)
blurred_pred = predict_top3(blurred_image)

print("Оригинал:", original_pred)
print("С шумом:", blurred_pred)

# Отображаем оригинальное и размытое изображение рядом
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image)
axes[0].set_title("Оригинальное изображение")
axes[0].axis("off")

axes[1].imshow(blurred_image)
axes[1].set_title("Изображение с GaussianBlur (radius=15)")
axes[1].axis("off")

plt.suptitle("Сравнение исходного изображения и изображения с шумом", fontsize=16)
plt.tight_layout()
plt.show()