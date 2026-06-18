from sentence_transformers import CrossEncoder
from config.settings import RERANK_MODEL, RERANK_TOP_N
_model = None # lazy-loaded once
def _get_model() -> CrossEncoder:
global _model
if _model is None:
_model = CrossEncoder(RERANK_MODEL)
return _model
def rerank_chunks(
query: str,
chunks: list[dict],
top_n: int = RERANK_TOP_N
) -> list[dict]:
model = _get_model()
pairs = [(query, c['text']) for c in chunks]
scores = model.predict(pairs)
for chunk, score in zip(chunks, scores):
chunk['rerank_score'] = round(float(score), 4)
ranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
return ranked[:top_n]