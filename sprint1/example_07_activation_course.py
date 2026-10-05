import os
from PIL import Image
from torchvision import models, transforms
import torch
import matplotlib.pyplot as plt

# Загрузка предобученной модели ResNet и перевод её в режим инференса
model = models.resnet18(pretrained=True) 
model.eval()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image = Image.open(os.path.join(BASE_DIR, "guitar.jpg")).convert('RGB')


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = preprocess(image).unsqueeze(0)

# Функция для визуализации активаций
def visualize_layer_activations(model, image_tensor, layer_name='layer1'):
    activations = {}
    
    def hook_fn(module, input, output):
        activations[layer_name] = output
    
    # Выбираем слой для визуализации
    if layer_name == 'layer1':
        model.layer1.register_forward_hook(hook_fn)
    elif layer_name == 'layer2':
        model.layer2.register_forward_hook(hook_fn)
    elif layer_name == 'layer3':
        model.layer3.register_forward_hook(hook_fn)
    elif layer_name == 'layer4':
        model.layer4.register_forward_hook(hook_fn)
    
    # Прямой проход
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Получаем активации выбранного слоя
    layer_activation = activations[layer_name][0].cpu().numpy()
    
    # Визуализируем первые 16 карт активации
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < layer_activation.shape[0]:
            ax.imshow(layer_activation[i], cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(f'Активации слоя {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
visualize_layer_activations(model, input_tensor, 'layer4')