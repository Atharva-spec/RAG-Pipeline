from typing import List, Optional, Union
from generation.memory import ConversationMemory

SYSTEM_PROMPT = (
    "You are a helpful, enterprise-grade AI assistant. Answer the user question accurately "
    "using ONLY the provided document context below.\n\n"
    "Guidelines:\n"
    "1. Ground your answer strictly in the provided Context chunks.\n"
    "2. If the context does not contain enough information to answer the question, clearly state: "
    '"I don\'t have enough information in the provided documents to answer that."\n'
    "3. Maintain a natural, professional tone. When relevant, cite the source document names (e.g. [Source: filename.pdf]).\n"
    "4. Do not speculate or invent facts."
)


def build_prompt(
    question: str,
    chunks: list[dict],
    memory: Optional[Union[ConversationMemory, List[dict]]] = None,
) -> list[dict]:
    """
    Constructs the message payload for the LLM, combining system instructions,
    retrieved document context, chat history, and the user question.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        context_parts.append(
            f'[{i}] Source: {source} (ID: {chunk_id})\n{chunk["text"]}'
        )

    context_block = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # If chat history is provided, append past dialog turns
    if memory is not None:
        if isinstance(memory, ConversationMemory):
            history_messages = memory.get_history_messages()
        elif isinstance(memory, list):
            history_messages = memory
        else:
            history_messages = []

        messages.extend(history_messages)

    user_message = f"Context:\n{context_block}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_message})

    return messages

