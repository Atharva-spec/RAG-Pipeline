from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL
def get_embedding_model() -> HuggingFaceEmbeddings:
return HuggingFaceEmbeddings(
model_name=EMBEDDING_MODEL,
model_kwargs={'device': 'cpu'},
encode_kwargs={'normalize_embeddings': True}
)