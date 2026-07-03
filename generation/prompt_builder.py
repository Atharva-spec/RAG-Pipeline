SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user question "
    "using ONLY the context provided below. "
    "If the answer is not in the context, say "
    "\"I don't have enough information to answer that.\" "
    "Do not make up information."
)


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f'[{i}] Source: {chunk["source"]}\n{chunk["text"]}'
        )

    context = "\n\n".join(context_parts)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
