from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import get_embeddings

def similarity(text1, text2, tokenizer, model):
    emb = get_embeddings([text1, text2], tokenizer, model)
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
    return sim