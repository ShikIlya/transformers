from transformers import AutoModel, AutoTokenizer
import torch
from src.visualize_attention import visualize_attention

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(
    model_name,
    output_attentions=True
)
model.eval()

# text = "The amazing movie won many awards"
# tokens = tokenizer(text, return_tensors="pt")
#
# with torch.no_grad():
#     outputs = model(**tokens)
#
# print(type(outputs.attentions))
# print(f'Количество слоёв: {len(outputs.attentions)}')
# print(f'Форма attention для слоя 0: {outputs.attentions[0].shape}')

# attention = outputs.attentions[0]
# print(f'Attention shape: {attention.shape}')
#
# attn_single = attention[0, 0]  # [seq_len, seq_len]
# print(f'Single head shape: {attn_single.shape}')

# Первый слой
# visualize_attention(tokens, outputs.attentions, tokenizer=tokenizer, layer=0, head=0)

# Средний слой
# visualize_attention(tokens, outputs.attentions, tokenizer=tokenizer, layer=3, head=0)

# Последний слой
# visualize_attention(tokens, outputs.attentions, tokenizer=tokenizer, layer=5, head=0)

# for head in range(8):  # DistilBERT имеет 8 голов
#     visualize_attention(tokens, outputs.attentions, tokenizer=tokenizer, layer=0, head=head)

text = "This movie was absolutely terrible and I hated it"
tokens = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**tokens)

visualize_attention(tokens, outputs.attentions, tokenizer=tokenizer, layer=5, head=0)