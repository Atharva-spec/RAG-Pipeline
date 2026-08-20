import re
from typing import Tuple

from config.settings import (
    GUARDRAIL_BLOCK_PROMPT_INJECTION,
    GUARDRAIL_CONFIDENCE_THRESHOLD,
    GUARDRAILS_ENABLED,
)

# Known prompt injection & jailbreak patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+)?(previous|prior)\s+(instructions|prompts)",
    r"(reveal|print|show|dump)\s+(your\s+)?(system\s+prompt|instructions|initial\s+prompt)",
    r"you\s+are\s+now\s+(in\s+)?(dan|developer|jailbreak|unfiltered)\s+mode",
    r"bypass\s+(all\s+)?(safety|security|content\s+filters)",
    r"system\s*:\s*override",
    r"<\s*script[^>]*>",
]

# Common conversational intents (chit-chat, greetings, bot identity)
GREETINGS_PATTERN = r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening|day))(\s+(there|everyone|folks|assistant|bot|friend))?[\s!\.\?]*$"
THANKS_PATTERN = r"^(thank\s+you|thanks|thx|appreciate\s+it|many\s+thanks)(\s+(very\s+much|so\s+much|a\s+lot|for\s+the\s+help|for\s+your\s+help))?[\s!\.\?]*$"
BOT_IDENTITY_PATTERN = r"^(who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|help(\s+me)?|what\s+is\s+this)[\s!\.\?]*$"



def check_input_safety(query: str) -> Tuple[bool, str | None]:
    """
    Evaluates query for prompt injections and malicious jailbreak attempts.
    Returns (is_safe, refusal_reason).
    """
    if not GUARDRAILS_ENABLED or not GUARDRAIL_BLOCK_PROMPT_INJECTION:
        return True, None

    clean_query = query.strip().lower()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, clean_query):
            return False, "Security violation: Potential prompt injection or system override attempt detected."

    return True, None


def detect_conversational_intent(query: str) -> str | None:
    """
    Detects if the query is a greeting, polite closure, or bot inquiry.
    Returns an instant conversational reply if matched (0ms vector retrieval), or None.
    """
    clean = query.strip().lower()

    if re.match(GREETINGS_PATTERN, clean):
        return "Hello! I am your Enterprise Document Assistant. How can I help you with your documents today?"

    if re.match(THANKS_PATTERN, clean):
        return "You're welcome! Let me know if you have any more questions about your documents."

    if re.match(BOT_IDENTITY_PATTERN, clean):
        return (
            "I am a native Enterprise RAG assistant. I search and synthesize answers "
            "directly from your uploaded documents with verifiable citations and guardrails."
        )

    return None


def validate_retrieval_confidence(
    ranked_chunks: list[dict],
    threshold: float = GUARDRAIL_CONFIDENCE_THRESHOLD,
) -> Tuple[bool, list[dict]]:
    """
    Zero-Hallucination Gate: Evaluates cross-encoder rerank confidence scores.
    Returns (is_confident, filtered_chunks).
    """
    if not GUARDRAILS_ENABLED:
        return bool(ranked_chunks), ranked_chunks

    if not ranked_chunks:
        return False, []

    # Filter chunks that meet the minimum confidence threshold
    confident_chunks = [
        c for c in ranked_chunks if c.get("confidence_score", 1.0) >= threshold
    ]

    if not confident_chunks:
        top_score = ranked_chunks[0].get("confidence_score", 0.0) if ranked_chunks else 0.0
        print(f"  [Guardrail] Refused: Top confidence ({top_score:.3f}) below threshold ({threshold:.3f})")
        return False, []

    return True, confident_chunks


def safe_refusal_response() -> str:
    """
    Standardized, professional refusal response when documents lack sufficient context.
    """
    return (
        "I could not find sufficient information in your uploaded documents to answer that question accurately. "
        "Please make sure the relevant document is uploaded or try refining your question."
    )


def sanitize_output(answer: str) -> str:
    """
    Ensures generated output does not leak system prompts or internal debug markers.
    """
    # Remove any accidental system prompt echoes
    sanitized = re.sub(r"(?i)system\s*prompt\s*:\s*", "", answer)
    return sanitized.strip()
