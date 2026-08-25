from typing import Optional, Union

from generation.memory import ConversationMemory
from generation.rephraser_agent import rephrase_query


def normalize_query_terms(text: str) -> str:
    """
    Normalizes query terms using the Rephraser Agent.
    """
    return rephrase_query(text.strip())


def condense_query(
    question: str,
    memory: Optional[Union[ConversationMemory, str]] = None,
) -> str:
    """
    Transforms a raw or follow-up question into an optimized standalone search query
    using the Rephraser Agent.
    """
    return rephrase_query(question, memory=memory)

