"""
Deprecated: Replaced by generation.rephraser_agent.RephraserAgent.
This module is maintained for backwards compatibility.
"""
from typing import Optional
from generation.rephraser_agent import rephrase_query


def correct_query_spelling(query: str) -> str:
    """
    Compatibility wrapper: delegates query spelling correction & normalization
    to the Rephraser Agent.
    """
    return rephrase_query(query)
