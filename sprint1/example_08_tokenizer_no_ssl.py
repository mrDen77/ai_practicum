from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import ssl
import sys


# Отключаем проверку SSL
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Для библиотеки requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Для huggingface_hub
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Глобально отключаем проверку SSL в ssl-модуле
ssl._create_default_https_context = ssl._create_unverified_context

local_model_path = "/models/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)


# Перевод модели в режим оценки
model.eval()

print(f"Кодировка stdout: {sys.stdout.encoding}")
print("Модель и токенизатор успешно загружены!")

# Пример текста на русском языке
text = "Всё говно, всё переделать!!!!"

# Токенизация с использованием padding и truncation
inputs = tokenizer(
    text,
    padding='max_length',   # Все последовательности дополняются специальными токенами до заданной длины. Это необходимо, чтобы обеспечить одинаковый размер входа для всех примеров.
    truncation=True,        # Обрезка, если текст длиннее max_length
    max_length=64,          # Если текст превышает 64 токена, он обрезается до нужного размера, чтобы избежать ошибок при обработке.
    return_tensors="pt"     # Результат возвращается в формате тензоров PyTorch, что удобно для дальнейшей работы с моделью.
)

print("Текст до токенизации:")
print(text)
print("Текст после токенизации:")
print(inputs)

# Выполнение предсказания без вычисления градиентов
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Применяем softmax для получения вероятностей
probabilities = torch.nn.functional.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities).item()

# Выводим результат
print(f"\nПредсказанный класс: {predicted_class}")
print(f"Вероятности классов: {probabilities}") 