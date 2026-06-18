from config.settings import TOP_K
def retrieve_chunks(vector_store, query: str, k: int = TOP_K) -> list[dict]:
results = vector_store.db.similarity_search_with_score(query, k=k)
chunks = []
for doc, score in results:
chunks.append({
'text': doc.page_content,
'source': doc.metadata.get('source', 'unknown'),
'score': round(score, 4)
})
return chunks