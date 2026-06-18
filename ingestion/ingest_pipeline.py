from ingestion.document_loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import get_embedding_model
from retrieval.vector_store import VectorStore
from config.settings import DATA_RAW_DIR
def run_ingestion(data_dir: str = DATA_RAW_DIR):
print('\n=== Starting ingestion ===')
print('\n[1/4] Loading documents...')
documents = load_documents(data_dir)
print(f' {len(documents)} document(s) loaded')
print('\n[2/4] Chunking...')
chunks = chunk_documents(documents)
print(f' {len(chunks)} chunks created')
print('\n[3/4] Loading embedding model...')
model = get_embedding_model()
print('\n[4/4] Storing in ChromaDB...')
store = VectorStore(model)
store.add_chunks(chunks)
store.persist()
print(f'\nIngestion complete! {len(chunks)} chunks stored.')
if __name__ == '__main__':
run_ingestion()