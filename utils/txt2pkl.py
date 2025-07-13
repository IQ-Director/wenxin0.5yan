import os
import time
import pickle
import erniebot

from tqdm import tqdm

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_docs(token:str,docs_dir = r'/home/aistudio/data',data_dir = r'/home/aistudio/index'):
    #assert not os.path.exists(os.path.join(data_dir, 'data.pkl')), '如需重新编码，请删除这段代码'
    exts = ['txt']
    batch_size = 16
    chunk_size = 384
    loader_cls = TextLoader

    docs = []
    embeddings = []
    texts = []
    metadatas = []

    for ext in exts:
        loader = DirectoryLoader(
            docs_dir,
            glob='*.%s' % ext,
            recursive=True,
            show_progress=True,
            silent_errors=True,
            loader_cls=loader_cls
        )
        docs += loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    docs = splitter.split_documents(docs)

    batch_docs = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]

    for batch_doc in tqdm(batch_docs):
        try:
            response = erniebot.Embedding.create(
                model='ernie-text-embedding',
                input=[item.page_content for item in batch_doc]
            )
            embeddings += [item['embedding'] for item in response['data']]
            texts += [item.page_content for item in batch_doc]
            metadatas += [item.metadata for item in batch_doc]
            time.sleep(1)
        except:
            for text in tqdm(batch_doc):
                try:
                    response = erniebot.Embedding.create(
                        model='ernie-text-embedding',
                        input=[text.page_content]
                    )
                    embeddings.append(response['data'][0]['embedding'])
                    texts.append(text.page_content)
                    metadatas.append(text.metadata)
                except:
                    continue

    data = {
        "embeddings": embeddings,
        "texts": texts,
        'metadatas': metadatas,
    }

    with open(os.path.join(data_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)