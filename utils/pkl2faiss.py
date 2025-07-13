import os
import pickle
import erniebot

from tqdm import tqdm
from typing import List
from langchain.schema.embeddings import Embeddings
from langchain_community.vectorstores import FAISS




class ErniebotEmbeddings(Embeddings):
    def __init__(self, access_token: str, api_type: str = 'aistudio', model: str = 'ernie-text-embedding', batch_size: int = 16, embedding_size: int = 384):
        erniebot.api_type = api_type
        erniebot.access_token = access_token
        self.model = model
        self.batch_size = batch_size
        self.embedding_size = embedding_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_texts = [texts[i: i+self.batch_size]
                       for i in range(0, len(texts), self.batch_size)]
        embeddings = []
        for batch_text in tqdm(batch_texts):
            try:
                response = erniebot.Embedding.create(
                    model=self.model,
                    input=batch_text
                )
                embeddings += [item['embedding'] for item in response['data']]
            except:
                for text in tqdm(batch_text):
                    try:
                        response = erniebot.Embedding.create(
                            model=self.model,
                            input=[text]
                        )
                        embeddings.append(response['data'][0]['embedding'])
                    except:
                        embeddings.append([0.0] * self.embedding_size)
                        continue
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = erniebot.Embedding.create(
            model=self.model,
            input=[text]
        )
        return response['data'][0]['embedding']

def load_pkl(token):
    data_dir = r'/home/aistudio/index'

    embedding = ErniebotEmbeddings(token)

    with open(os.path.join(data_dir, 'data.pkl'), 'rb') as f:
        data = pickle.load(f)
        
    index = FAISS._FAISS__from(
        texts=data['texts'], embeddings=data['embeddings'], embedding=embedding, metadatas=data['metadatas']
    )
    index.save_local(data_dir)