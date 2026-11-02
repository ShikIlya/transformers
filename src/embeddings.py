import torch
import numpy as np

from .tokenization import tokenize_texts

def get_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        tokens = tokenize_texts(
            texts=batch_texts,
            tokenizer=tokenizer,
            max_length=128,
        )

        # Получаем hidden states
        with torch.no_grad():
            outputs = model(**tokens)

        # Извлекаем CLS-токены
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    # Объединяем все батчи
    return np.vstack(all_embeddings)