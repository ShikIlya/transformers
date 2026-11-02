from transformers import AutoTokenizer
import torch
import transformers
from src.tokenization import tokenize_texts

print('torch version ', torch.__version__)
print('transformers version ', transformers.__version__)

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

print('vocab_size ', tokenizer.vocab_size)
print('model max length ', tokenizer.model_max_length)

text = "This movie was absolutely amazing!"

tokens = tokenizer(text)

print('tokens ', tokens)

input_ids = tokens['input_ids']

print(f'Количество токенов: {len(input_ids)}')

decoded = tokenizer.decode(input_ids)

print(f'Декодировано {decoded}')

texts = [
    "This movie was great!",
    "Terrible movie, waste of time."
]

tokens = tokenize_texts(texts=texts, tokenizer=tokenizer)

print(f'Shape: {tokens["input_ids"].shape}')
print(f'Attention mask:\n{tokens["attention_mask"]}')


print(f'CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})')
print(f'SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})')
print(f'PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})')

single = tokenizer(text, return_tensors="pt")
print(f'Input IDs: {single["input_ids"]}')
print(f'Decoded: {tokenizer.decode(single["input_ids"][0])}')

def explain_tokenization(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f'Исходный текст: {text}')
    print(f'Токены: {tokens}')
    print(f'IDs: {ids}')
    print(f'Количество: {len(tokens)}')

explain_tokenization("Transformers are amazing!", tokenizer)