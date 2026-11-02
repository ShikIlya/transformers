from transformers import AutoModel, AutoTokenizer
import torch

from src.embeddings import get_embeddings
from src.cosine_similarity import similarity

model_name = "distilbert-base-uncased"

model = AutoModel.from_pretrained(model_name)
model.eval()

print(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "This movie was absolutely amazing!"
tokens = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**tokens)

print('type(outputs): ', type(outputs))
print('outputs.last_hidden_state.shape: ', outputs.last_hidden_state.shape)

cls_embedding = outputs.last_hidden_state[:, 0, :]
print(f'CLS embedding shape: {cls_embedding.shape}')
print(f'CLS embedding: {cls_embedding[0][:5]}...')  # первые 5 значений

texts = [
    "This movie was absolutely amazing!",
    "Terrible movie, waste of time.",
    "Pretty good, I liked it.",
    "Boring and too long."
]

# 2. Получите эмбеддинги:
embeddings = get_embeddings(texts, tokenizer, model)
print(f'Embeddings shape: {embeddings.shape}')
print(f'Ожидается: (4, 768) для DistilBERT')

sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)
sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)

print(f'Сходство похожих: {sim1:.3f}')
print(f'Сходство разных: {sim2:.3f}')