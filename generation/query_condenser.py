import re
from typing import Optional, Union

from config.settings import FAST_QUERY_CONDENSER
from generation.llm_client import get_answer
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

_REFERENTIAL_REGEX = re.compile(
    r"\b(it|its|they|them|their|theirs|this|that|these|those|he|him|his|she|her|the former|the latter|the previous|the above)\b",
    re.IGNORECASE,
)
_SHORT_FOLLOWUP_REGEX = re.compile(
    r"^(what about|how about|why\?|and\?|more\?|what else|tell me more|explain more|how so)",
    re.IGNORECASE,
)


def _needs_condensation(question: str) -> bool:
    """
    Fast heuristic check: determines in ~0.01ms if the question actually refers
    to prior context and needs condensation.
    """
    clean_q = question.strip()
    if _SHORT_FOLLOWUP_REGEX.search(clean_q):
        return True
    if _REFERENTIAL_REGEX.search(clean_q):
        return True
    return False


def _heuristic_condense(question: str, history_text: str, memory: Optional[ConversationMemory] = None) -> str:
    """
    Fast 0ms context stitching if LLM call is bypassed or for instant resolution.
    """
    if memory and hasattr(memory, "history") and memory.history:
        last_turn = memory.history[-1]
        last_query = last_turn.get("user", "")
        # Remove common question words to extract subject keywords
        cleaned_last = re.sub(
            r"^(what is|what are|how to|how does|why is|explain|tell me about|describe)\s+",
            "",
            last_query,
            flags=re.IGNORECASE,
        ).rstrip("?. ")
        if cleaned_last:
            return f"{question.strip()} (Regarding: {cleaned_last})"
    return question.strip()


def condense_query(
    question: str,
    memory: Optional[Union[ConversationMemory, str]] = None,
) -> str:
    """
    Transforms a conversational follow-up question into a standalone search query.
    Uses fast heuristic pruning to avoid unnecessary LLM latency when the question
    is already standalone.
    """
    clean_q = question.strip()
    if not clean_q:
        return clean_q

    if memory is None:
        return clean_q

    if isinstance(memory, ConversationMemory):
        if memory.is_empty():
            return clean_q
        history_text = memory.get_history_text()
    else:
        history_text = str(memory).strip()

    if not history_text:
        return clean_q

    # Fast heuristic check: If query does not contain referential pronouns or follow-up markers, return instantly
    if FAST_QUERY_CONDENSER and not _needs_condensation(clean_q):
        return clean_q

    user_prompt = f"Chat History:\n{history_text}\n\nFollow-up Question: {clean_q}\n\nStandalone Search Query:"

    try:
        raw_response = get_answer(
            messages=[
                {"role": "system", "content": CONDENSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        condensed = raw_response.strip()
        # Clean up any surrounding quotes if generated
        if (condensed.startswith('"') and condensed.endswith('"')) or (
            condensed.startswith("'") and condensed.endswith("'")
        ):
            condensed = condensed[1:-1].strip()

        if condensed:
            print(f"  [Query Condenser] '{clean_q}' ──► '{condensed}'")
            return condensed
        return clean_q
    except Exception as e:
        print(f"  [Query Condenser Warning] Fast LLM condensation failed ({e}), using heuristic.")
        mem_obj = memory if isinstance(memory, ConversationMemory) else None
        return _heuristic_condense(clean_q, history_text, mem_obj)
