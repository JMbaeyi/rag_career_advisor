import faiss
import numpy as np

class FAISSRetriever:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)

    def add_embeddings(self, embeddings):
        self.index.add(np.array(embeddings, dtype='float32'))

    def search(self, query_vector, k=5):
        D, I = self.index.search(np.array([query_vector], dtype='float32'), k)
        return D, I
