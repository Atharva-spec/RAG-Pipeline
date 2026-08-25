import logging
import re
from typing import Optional, Union

from config.settings import FAST_REPHRASER, REPHRASER_AGENT_ENABLED
from generation.llm_client import get_answer
from generation.memory import ConversationMemory

logger = logging.getLogger(__name__)

REPHRASER_SYSTEM_PROMPT = (
    "You are an expert search query rephraser and query reformulation agent for a hybrid RAG system.\n"
    "Your job is to transform raw, noisy, conversational, or typo-filled user questions into clean, standalone, high-precision search queries optimized for dense vector and keyword search.\n\n"
    "Guidelines:\n"
    "1. Correct all spelling mistakes, typos, phonetic errors, and malformed compounds (e.g., 'suprvised lerning' -> 'supervised learning', 'pre procesing' -> 'preprocessing', 'descision tree classifcation' -> 'decision tree classification').\n"
    "2. If chat history is provided, resolve all pronouns (e.g., 'it', 'its', 'they', 'them', 'this', 'that', 'the former') and follow-up references into a complete, standalone query.\n"
    "3. Strip conversational filler (e.g., 'can you please explain', 'i want to know about', 'tell me') while preserving all core technical terms, domain entities, and search intent.\n"
    "4. Do NOT answer the question.\n"
    "5. Output ONLY the rephrased search query without quotes, markdown formatting, or explanations."
)

_REFERENTIAL_REGEX = re.compile(
    r"\b(it|its|they|them|their|theirs|this|that|these|those|he|him|his|she|her|the former|the latter|the previous|the above)\b",
    re.IGNORECASE,
)

_SHORT_FOLLOWUP_REGEX = re.compile(
    r"^(what about|how about|why\?|and\?|more\?|what else|tell me more|explain more|how so)",
    re.IGNORECASE,
)

_CONVERSATIONAL_FILLER_REGEX = re.compile(
    r"^(can you please |could you please |please |can you |could you |tell me |i want to know |i would like to know |help me understand |explain to me )\s*",
    re.IGNORECASE,
)


class RephraserAgent:
    """
    Agentic Query Rephraser that refines raw user input into optimal standalone search queries.
    Handles spelling corrections, entity disambiguation, noise reduction, and conversational
    context resolution.
    """

    def __init__(self, system_prompt: str = REPHRASER_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def _needs_llm_processing(self, question: str, has_history: bool) -> bool:
        """
        Fast heuristic check to determine if the query requires LLM rephrasing
        or can be returned / lightly normalized directly.
        """
        clean = question.strip()
        if not clean:
            return False

        # 1. If conversational context exists and query refers to prior context
        if has_history:
            if _SHORT_FOLLOWUP_REGEX.search(clean) or _REFERENTIAL_REGEX.search(clean):
                return True

        # 2. Check for conversational filler prefixes
        if _CONVERSATIONAL_FILLER_REGEX.search(clean):
            return True

        # 3. Check for potential typos: repeated unusual chars, non-alphanumeric noise
        if re.search(r"([a-zA-Z])\1{2,}", clean):  # e.g., 'whaaat', 'soooo'
            return True

        # 4. Check for obvious misspellings / malformed question starts (e.g., 'waht', 'explian', 'suprvised')
        clean_lower = clean.lower()
        if any(typo in clean_lower for typo in ["waht", "whar", "explian", "suprvised", "lerning", "procesing", "descision", "classifcation", "modling"]):
            return True

        return False


    def _heuristic_fallback(
        self,
        question: str,
        history_text: str = "",
        memory: Optional[ConversationMemory] = None,
    ) -> str:
        """
        Instant fallback rephrasing using heuristic pattern matching when LLM is unavailable.
        """
        clean = question.strip()
        # Strip conversational filler
        clean = _CONVERSATIONAL_FILLER_REGEX.sub("", clean).strip()

        # Stitch context if available and referential
        if memory and hasattr(memory, "history") and memory.history:
            last_turn = memory.history[-1]
            last_query = last_turn.get("user", "")
            cleaned_last = re.sub(
                r"^(what is|what are|how to|how does|why is|explain|tell me about|describe)\s+",
                "",
                last_query,
                flags=re.IGNORECASE,
            ).rstrip("?. ")
            if cleaned_last and (_REFERENTIAL_REGEX.search(clean) or _SHORT_FOLLOWUP_REGEX.search(clean)):
                return f"{clean} (Regarding: {cleaned_last})"

        return clean

    def rephrase(
        self,
        question: str,
        memory: Optional[Union[ConversationMemory, str]] = None,
        force_llm: bool = False,
    ) -> str:
        """
        Transforms a raw question into a clean, standalone search query using LLM reasoning
        with fast heuristic fallback.
        """
        if not question or not question.strip():
            return ""

        raw_q = question.strip()

        if not REPHRASER_AGENT_ENABLED:
            return raw_q

        # Extract history text if provided
        history_text = ""
        has_history = False
        mem_obj: Optional[ConversationMemory] = None

        if memory is not None:
            if isinstance(memory, ConversationMemory):
                mem_obj = memory
                if not memory.is_empty():
                    history_text = memory.get_history_text()
                    has_history = bool(history_text)
            else:
                history_text = str(memory).strip()
                has_history = bool(history_text)

        # Fast heuristic bypass for clean standalone single queries
        if not force_llm and FAST_REPHRASER and not self._needs_llm_processing(raw_q, has_history):
            return raw_q

        # Construct prompt for Rephraser Agent
        if has_history:
            user_prompt = f"Chat History:\n{history_text}\n\nUser Question: {raw_q}\n\nOptimized Standalone Search Query:"
        else:
            user_prompt = f"User Question: {raw_q}\n\nOptimized Search Query:"

        try:
            raw_response = get_answer(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            rephrased = raw_response.strip()

            # Clean quotes if generated
            if (rephrased.startswith('"') and rephrased.endswith('"')) or (
                rephrased.startswith("'") and rephrased.endswith("'")
            ):
                rephrased = rephrased[1:-1].strip()

            if rephrased:
                print(f"  [Rephraser Agent] '{raw_q}' ──► '{rephrased}'")
                return rephrased
            return raw_q

        except Exception as e:
            logger.warning(f"Rephraser Agent LLM call failed ({e}), using heuristic fallback.")
            return self._heuristic_fallback(raw_q, history_text, mem_obj)


# Global singleton instance
_default_rephraser = RephraserAgent()


def rephrase_query(
    question: str,
    memory: Optional[Union[ConversationMemory, str]] = None,
    force_llm: bool = False,
) -> str:
    """
    Public entrypoint for the Rephraser Agent.
    """
    return _default_rephraser.rephrase(question, memory=memory, force_llm=force_llm)

