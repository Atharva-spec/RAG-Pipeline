from typing import Optional, Union

import ollama

from config.settings import LLM_MODEL
from generation.memory import ConversationMemory

CONDENSER_SYSTEM_PROMPT = (
    "You are an expert search query refiner. Given a conversation history and a follow-up user question, "
    "rephrase the follow-up question into a single, complete, standalone search query that includes all "
    "necessary context, names, and subjects mentioned in the history.\n\n"
    "Rules:\n"
    "1. Do NOT answer the question.\n"
    "2. If the user question is already standalone and does not refer to prior context, return it unchanged.\n"
    "3. Output ONLY the rephrased standalone query with no explanation, markdown formatting, or quotes."
)


def condense_query(
    question: str,
    memory: Optional[Union[ConversationMemory, str]] = None,
) -> str:
    """
    Transforms a conversational follow-up question into a standalone search query
    using chat history context. If no history is present, returns original query with 0ms latency.
    """
    if memory is None:
        return question.strip()

    if isinstance(memory, ConversationMemory):
        if memory.is_empty():
            return question.strip()
        history_text = memory.get_history_text()
    else:
        history_text = str(memory).strip()

    if not history_text:
        return question.strip()

    user_prompt = f"Chat History:\n{history_text}\n\nFollow-up Question: {question.strip()}\n\nStandalone Search Query:"

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": CONDENSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.0},
        )
        condensed = response["message"]["content"].strip()
        # Clean up any surrounding quotes if generated
        if (condensed.startswith('"') and condensed.endswith('"')) or (
            condensed.startswith("'") and condensed.endswith("'")
        ):
            condensed = condensed[1:-1].strip()

        if condensed:
            print(f"  [Query Condenser] '{question}' ──► '{condensed}'")
            return condensed
        return question.strip()
    except Exception as e:
        print(f"  [Query Condenser Warning] Failed to condense query ({e}), using raw question.")
        return question.strip()
