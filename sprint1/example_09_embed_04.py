from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 1) Загрузка токенизатора и модели RuBERT
tokenizer = AutoTokenizer.from_pretrained("/models/rubert-base-cased-sentiment")
model = AutoModel.from_pretrained("/models/rubert-base-cased-sentiment")
model.eval()

# 2) Определяем функцию embed(texts)
def embed(texts):
    # Токенизация с паддингом и усечением
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state  # [batch, seq_len, hidden]
    # mean-pooling по последовательности
    emb = out.mean(dim=1).cpu().numpy()  # [batch, hidden]
    # L2-нормализация
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)

print(embed('Как быстро доставляется заказ?'))