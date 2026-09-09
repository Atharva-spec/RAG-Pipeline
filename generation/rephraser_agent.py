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
    "2. Normalize colloquial acronym expansion queries (e.g., 'full form of X', 'X full form', 'what is X in full', 'what does X stand for') into search phrases like 'What does X stand for? (X acronym expansion and definition)'.\n"
    "3. If chat history is provided, resolve all pronouns (e.g., 'it', 'its', 'they', 'them', 'this', 'that', 'the former') and follow-up references (e.g. 'full form?', 'advantages?', 'types?') into a complete, standalone query integrating the previous topic.\n"
    "4. Strip conversational filler (e.g., 'can you please explain', 'i want to know about', 'tell me', 'can you tell me about') while preserving all core technical terms, domain entities, and search intent.\n"
    "5. CRITICAL: Do NOT invent, append, or hallucinate extra search keywords, metadata terms, or encyclopedic terms not implied by the user (e.g., do NOT expand 'neem' into 'Neem definition, properties, uses, and benefits'; keep entity searches concise and focused).\n"
    "6. Do NOT answer the question.\n"
    "7. Output ONLY the rephrased search query without quotes, markdown formatting, or explanations."
)

_REFERENTIAL_REGEX = re.compile(
    r"\b(it|its|they|them|their|theirs|this|that|these|those|he|him|his|she|her|the former|the latter|the previous|the above)\b",
    re.IGNORECASE,
)

_SHORT_FOLLOWUP_REGEX = re.compile(
    r"^(what about|how about|why\?|and\?|more\?|what else|tell me more|explain more|how so)",
    re.IGNORECASE,
)

_TRAILING_FILLER_REGEX = re.compile(
    r"[\s,]+(?:to me|for me|please|can you|could you|if possible|as well|right now|currently)\s*\??$",
    re.IGNORECASE,
)

_LEADING_CONVERSATIONAL_FRAMES = [
    r"^(?:can you (?:please )?(?:tell|explain|describe|give|share|provide)(?: to)?(?: me)?(?: more)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?)\s+",
    r"^(?:could you (?:please )?(?:tell|explain|describe|give|share|provide)(?: to)?(?: me)?(?: more)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?)\s+",
    r"^(?:please (?:tell|explain|describe|give|share|provide)(?: to)?(?: me)?(?: more)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?)\s+",
    r"^(?:tell me(?: more)?(?: about| regarding| on)?)\s+",
    r"^(?:talk (?:to me )?about)\s+",
    r"^(?:explain to me(?: about)?)\s+",
    r"^(?:describe to me(?: about)?)\s+",
    r"^(?:what can you tell me(?: about| regarding)?)\s+",
    r"^(?:what do you know about)\s+",
    r"^(?:what about|how about)\s+",
    r"^(?:i (?:just )?(?:want|would like|wish) to know(?: more)?(?: about| on| regarding)?)\s+",
    r"^(?:give me(?: some)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?)\s+",
    r"^(?:information (?:about|on|regarding))\s+",
    r"^(?:details (?:about|on|regarding))\s+",
    r"^(?:brief (?:me on|about))\s+",
    r"^(?:help me understand(?: about)?)\s+",
    r"^(?:(?:the )?(?:meaning|definition|concept|overview|summary|details)(?: and (?:uses?|applications?|benefits?))? (?:of|for|about))\s+",
    r"^(?:(?:the )?(?:uses?|applications?|benefits?)(?: and (?:meaning|definition|concept))? (?:of|for|about))\s+",
    r"^(?:what is the (?:meaning|definition|use|purpose) of)\s+",
]

_CONVERSATIONAL_FILLER_REGEX = re.compile(
    r"^(can you (?:please )?(?:tell|explain|describe|give|share|provide)|could you (?:please )?(?:tell|explain|describe|give|share|provide)|please (?:tell|explain|describe|give|share|provide)|tell me|talk about|explain to me|i want to know|i would like to know|help me understand|what can you tell me|what do you know about)\s*",
    re.IGNORECASE,
)

def strip_conversational_frames(query: str) -> str:
    """
    Strips leading and trailing conversational wrapper phrases, politeness markers,
    and filler (e.g., 'tell me about neem', 'can you explain neem to me please')
    leaving the core domain question or entity.
    """
    res = query.strip()
    res = _TRAILING_FILLER_REGEX.sub("", res).strip()
    for pattern in _LEADING_CONVERSATIONAL_FRAMES:
        res = re.sub(pattern, "", res, flags=re.IGNORECASE).strip()
    res = _TRAILING_FILLER_REGEX.sub("", res).strip()
    if len(res.split()) <= 4:
        res = res.rstrip("?. ")
    return res

_STOPWORDS = {"its", "it", "the", "this", "that", "their", "a", "an", "what", "which", "is", "of", "in"}


def _extract_last_subject(memory: Optional[ConversationMemory]) -> Optional[str]:
    """
    Extracts the core topic or entity from the most recent conversational turn.
    """
    if not memory or not hasattr(memory, "turns") or not memory.turns:
        return None
    last_turn = memory.turns[-1]
    last_query = getattr(last_turn, "user_message", "")
    if not last_query:
        return None
    # Strip conversational frames: "tell me about", "meaning and use of", etc.
    cleaned = strip_conversational_frames(last_query.strip())
    # Strip common question frames: "what is", "explain", etc.
    cleaned = re.sub(
        r"^(?:what is(?: the)?|what are(?: the)?|how to|how does|why is|explain(?: the)?|tell me about|describe(?: the)?)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).rstrip("?. ")
    return cleaned.strip() if cleaned else None


def normalize_query_heuristics(
    question: str,
    memory: Optional[ConversationMemory] = None,
) -> str:
    """
    Deterministic query normalization for acronym inquiries, regional idioms ('full form'),
    conversational frame stripping, and multi-turn conversational follow-ups.
    """
    clean = question.strip()
    if not clean:
        return ""

    # Strip conversational filler prefixes & wrappers
    clean = strip_conversational_frames(clean)

    # 1. Pattern: "full form of <TERM>" or "what is the full form of <TERM>"
    m_term1 = re.search(
        r"(?i)\b(?:what is (?:the )?)?full[-\s]*form of\s+([a-zA-Z0-9_\-]+)\b",
        clean,
    )
    if m_term1 and m_term1.group(1).lower() not in _STOPWORDS:
        term = m_term1.group(1).upper()
        return f"what does {term} stand for ({term} acronym meaning)"

    # 2. Pattern: "<TERM> full form" (e.g., 'peas full form', 'SVM fullform')
    m_term2 = re.search(
        r"(?i)\b([a-zA-Z0-9_\-]+)\s+full[-\s]*form\b",
        clean,
    )
    if m_term2 and m_term2.group(1).lower() not in _STOPWORDS:
        term = m_term2.group(1).upper()
        return f"what does {term} stand for ({term} acronym meaning)"

    # 3. Pattern: Elliptical "full form" or "what does it stand for" referencing chat history
    if re.search(r"(?i)\bfull[-\s]*form\b", clean) or re.search(r"(?i)\bwhat does it stand for\b", clean):
        last_subj = _extract_last_subject(memory)
        if last_subj:
            return f"what does {last_subj.upper()} stand for ({last_subj.upper()} acronym meaning)"
        return re.sub(r"(?i)\bfull[-\s]*form\b", "acronym meaning definition", clean)

    # 4. Multi-turn context resolution for short follow-ups (e.g. 'advantages', 'types', 'explain it')
    if memory and hasattr(memory, "turns") and memory.turns:
        last_subj = _extract_last_subject(memory)
        if last_subj:
            # Referential pronouns
            if _REFERENTIAL_REGEX.search(clean) or _SHORT_FOLLOWUP_REGEX.search(clean):
                if last_subj.lower() not in clean.lower():
                    return f"{clean} (Regarding: {last_subj})"

            # Very short follow-ups (<= 4 words) that are elliptical fragments, NOT standalone questions
            words = clean.split()
            if len(words) <= 4 and last_subj.lower() not in clean.lower():
                question_or_greeting = {
                    "hi", "hello", "thanks", "bye", "ok", "yes", "no",
                    "what", "who", "where", "when", "how", "why", "which",
                    "is", "are", "was", "were", "do", "does", "did", "can",
                    "could", "should", "would", "tell", "explain", "describe",
                }
                if not any(w.lower() in question_or_greeting for w in words):
                    return f"{clean} of {last_subj}"

    return clean


class RephraserAgent:
    """
    Agentic Query Rephraser that refines raw user input into optimal standalone search queries.
    Handles spelling corrections, entity disambiguation, noise reduction, and conversational
    context resolution.
    """

    def __init__(self, system_prompt: str = REPHRASER_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

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
        clean = strip_conversational_frames(clean)

        # Stitch context if available and referential
        last_subj = _extract_last_subject(memory)
        if last_subj and (_REFERENTIAL_REGEX.search(clean) or _SHORT_FOLLOWUP_REGEX.search(clean)):
            if last_subj.lower() not in clean.lower():
                return f"{clean} (Regarding: {last_subj})"

        return normalize_query_heuristics(clean, memory)

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

        # 2. Check for potential typos: repeated unusual chars, non-alphanumeric noise
        if re.search(r"([a-zA-Z])\1{2,}", clean):  # e.g., 'whaaat', 'soooo'
            return True

        # 3. Check for obvious misspellings / malformed question starts (e.g., 'waht', 'explian', 'suprvised')
        clean_lower = clean.lower()
        if any(typo in clean_lower for typo in ["waht", "whar", "explian", "suprvised", "lerning", "procesing", "descision", "classifcation", "modling"]):
            return True

        # 4. Check for conversational filler prefixes that cannot be cleanly stripped
        if _CONVERSATIONAL_FILLER_REGEX.search(clean):
            stripped = strip_conversational_frames(clean)
            if not stripped or stripped.lower() == clean.lower():
                return True

        return False

    def rephrase(
        self,
        question: str,
        memory: Optional[Union[ConversationMemory, str]] = None,
        force_llm: bool = False,
    ) -> str:
        """
        Transforms a raw question into a clean, standalone search query using deterministic
        normalization and LLM reasoning.
        """
        if not question or not question.strip():
            return ""

        raw_q = question.strip()

        if not REPHRASER_AGENT_ENABLED:
            return raw_q

        # Extract history text & memory object if provided
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

        # 1. Deterministic heuristic normalization (instant 0ms, strips filler & normalizes acronyms)
        normalized = normalize_query_heuristics(raw_q, mem_obj)
        if normalized != raw_q and not force_llm:
            if not self._needs_llm_processing(normalized, has_history):
                print(f"  [Rephraser Agent (Heuristic)] '{raw_q}' ──► '{normalized}'")
                return normalized

        if re.search(r"(?i)\bfull[-\s]*form\b", raw_q):
            if normalized != raw_q:
                print(f"  [Rephraser Agent (Heuristic)] '{raw_q}' ──► '{normalized}'")
                return normalized

        # 2. Fast heuristic bypass for clean standalone single queries
        if not force_llm and FAST_REPHRASER and not self._needs_llm_processing(raw_q, has_history):
            return normalized if normalized else raw_q

        # 3. Construct prompt for Rephraser Agent LLM
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
                print(f"  [Rephraser Agent (LLM)] '{raw_q}' ──► '{rephrased}'")
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


