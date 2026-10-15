import os

from gensim.models import KeyedVectors

print(os.listdir('.')) 

model_ft = KeyedVectors.load_word2vec_format(
    "/models/fasttext/cc.ru.300.vec.gz", 
    binary=False,
    limit=50000
)  # fastText бинарная модель
print("Модель загружена, словарь содержит", len(model_ft), "слов")