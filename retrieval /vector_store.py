import chromadb
from langchain_community.vectorstores import Chroma
from config.settings import CHROMA_DB_DIR, COLLECTION_NAME
class VectorStore:
def __init__(self, embedding_model):
self.embedding_model = embedding_model
self.db = None
def add_chunks(self, chunks: list[dict]):
texts = [c['text'] for c in chunks]
metadatas = [{'source': c['source'], 'id': c['id']} for c in chunks]
self.db = Chroma.from_texts(
texts=texts,
embedding=self.embedding_model,
metadatas=metadatas,
persist_directory=CHROMA_DB_DIR,
collection_name=COLLECTION_NAME
)
def persist(self):
if self.db:
self.db.persist()
print(f' DB persisted to {CHROMA_DB_DIR}')
def load(self):
self.db = Chroma(
persist_directory=CHROMA_DB_DIR,
embedding_function=self.embedding_model,
collection_name=COLLECTION_NAME
)
return self.db
def as_retriever(self, k: int = 10):
return self.db.as_retriever(search_kwargs={'k': k})