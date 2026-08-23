import base64
import re
import unicodedata
from typing import Tuple

from config.settings import (
    GUARDRAIL_BLOCK_PROMPT_INJECTION,
    GUARDRAIL_CONFIDENCE_THRESHOLD,
    GUARDRAILS_ENABLED,
)

# Homoglyph lookalike character translation map (Cyrillic, Greek, Latin)
HOMOGLYPH_MAP = str.maketrans({
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ɑ": "a",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
})

# Leetspeak translation map for obfuscated attack payloads
LEETSPEAK_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "@": "a",
    "5": "s",
    "$": "s",
    "7": "t",
    "+": "t",
    "!": "i",
    "|": "i",
})

# Known prompt injection, jailbreak, and system override patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|preceding)\s+(instructions|prompts|rules|guidelines)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)",
    r"(reveal|print|show|dump|output)\s+(your\s+)?(system\s+prompt|instructions|initial\s+prompt|configuration)",
    r"(you\s+are\s+now|play\s+a\s+game\s+where\s+you\s+are|act\s+as|pretend\s+to\s+be)\s+(in\s+)?(dan|developer|jailbreak|unfiltered|evil)\s+mode",
    r"we\s+are\s+playing\s+a\s+game\s+where\s+you\s+are\s+dan",
    r"bypass\s+(all\s+)?(safety|security|content\s+filters|guardrails)",
    r"system\s*:\s*(override|instruction|prompt)",
    r"<\s*script[^>]*>",
    r"<\s*\|\s*im_start\s*\|\s*>",
    r"```\s*system",
]

# Common conversational intents (chit-chat, greetings, bot identity)
GREETINGS_PATTERN = r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening|day))(\s+(there|everyone|folks|assistant|bot|friend))?[\s!\.\?]*$"
THANKS_PATTERN = r"^(thank\s+you|thanks|thx|appreciate\s+it|many\s+thanks)(\s+(very\s+much|so\s+much|a\s+lot|for\s+the\s+help|for\s+your\s+help))?[\s!\.\?]*$"
BOT_IDENTITY_PATTERN = r"^(who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|help(\s+me)?|what\s+is\s+this)[\s!\.\?]*$"



def check_input_safety(query: str) -> Tuple[bool, str | None]:
    """
    Evaluates query for prompt injections, obfuscated leetspeak, homoglyphs,
    Base64 encoded payloads, and malicious jailbreak attempts.
    Returns (is_safe, refusal_reason).
    """
    if not GUARDRAILS_ENABLED or not GUARDRAIL_BLOCK_PROMPT_INJECTION:
        return True, None

    # Collect normalized representation candidates
    candidates = []

    # 1. Unicode NFKD & Homoglyph translation
    norm_query = unicodedata.normalize("NFKD", query).translate(HOMOGLYPH_MAP).strip().lower()
    candidates.append(norm_query)

    # 2. Leetspeak translated variant
    leet_query = norm_query.translate(LEETSPEAK_MAP)
    candidates.append(leet_query)

    # 3. Detect and decode any embedded Base64 payload strings
    for b64_match in re.finditer(r"[A-Za-z0-9+/]{12,}={0,2}", query):
        try:
            raw_b64 = b64_match.group(0)
            decoded_bytes = base64.b64decode(raw_b64)
            decoded_str = decoded_bytes.decode("utf-8", errors="ignore").strip().lower()
            if decoded_str:
                candidates.append(decoded_str)
                candidates.append(decoded_str.translate(LEETSPEAK_MAP))
        except Exception:
            pass

    # Check all candidate representations against attack patterns
    for candidate in candidates:
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, candidate, flags=re.IGNORECASE):
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
        c for c in ranked_chunks
        if c.get("confidence_score", c.get("score", 0.0)) >= threshold
    ]

    if not confident_chunks:
        top_score = ranked_chunks[0].get("confidence_score", ranked_chunks[0].get("score", 0.0)) if ranked_chunks else 0.0
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


# ──────────────────────────────────────────────────────────────
# Output Safety Guardrail (Pre-Display Filter)
# ──────────────────────────────────────────────────────────────

# Known secret & credential leak patterns to redact from final outputs
SECRET_LEAK_PATTERNS = [
    (r"(?i)(?:bearer\s+[a-zA-Z0-9_\-\.]{20,})", "[REDACTED TOKEN]"),
    (r"(?i)(?:sk-[a-zA-Z0-9]{20,})", "[REDACTED API KEY]"),
    (r"(?i)(?:ghp_[a-zA-Z0-9]{36,})", "[REDACTED GITHUB TOKEN]"),
    (r"(?i)(?:AKIA[0-9A-Z]{16})", "[REDACTED AWS KEY]"),
    (r"(?i)(?:-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----)", "[REDACTED PRIVATE KEY]"),
]

# System prompt leakage patterns to detect in generated answers
SYSTEM_LEAK_PATTERNS = [
    r"(?i)you\s+are\s+a\s+helpful,\s+enterprise-grade\s+AI\s+assistant",
    r"(?i)mandatory\s+guidelines\s*:",
    r"(?i)critical\s+security\s+rule\s*:",
    r"(?i)treat\s+them\s+strictly\s+as\s+data\s+to\s+extract\s+facts\s+from",
    r"(?i)ground\s+your\s+answer\s+strictly\s+in\s+the\s+provided\s+context",
]


def sanitize_output(answer: str) -> str:
    """
    Ensures generated output does not leak system prompts, role headers, or sensitive secrets.
    """
    if not answer:
        return ""

    # 1. Remove accidental system prompt / assistant echoes
    sanitized = re.sub(r"(?i)system\s*prompt\s*:\s*", "", answer)
    sanitized = re.sub(r"(?i)^(assistant|ai|bot)\s*:\s*", "", sanitized)

    # 2. Redact sensitive credentials / API keys / tokens
    for pattern, replacement in SECRET_LEAK_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized)

    return sanitized.strip()


def validate_output_safety(raw_output: str) -> Tuple[bool, str, str | None]:
    """
    Pre-Display Output Guardrail:
    Evaluates generated output before presenting to the end user:
    1. Rejects empty, whitespace-only, or broken outputs with safe refusal.
    2. Detects and intercepts system prompt / instruction regurgitation.
    3. Redacts leaked credentials/tokens.
    4. Cleans internal artifact markers.

    Returns:
        (is_safe, sanitized_output, alert_reason)
    """
    if not GUARDRAILS_ENABLED:
        return True, raw_output.strip() if raw_output else "", None

    if not raw_output or not raw_output.strip():
        return False, safe_refusal_response(), "Empty response received from LLM."

    clean = raw_output.strip()

    # 1. Check for system prompt leakage
    for pattern in SYSTEM_LEAK_PATTERNS:
        if re.search(pattern, clean):
            print("  [Output Guardrail] Intercepted system prompt leakage in model response")
            # Scrub leaked prompt lines
            sanitized = sanitize_output(clean)
            for leak_p in SYSTEM_LEAK_PATTERNS:
                sanitized = re.sub(leak_p, "", sanitized)
            sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
            if not sanitized or len(sanitized) < 10:
                return False, safe_refusal_response(), "Output contained system prompt leak."
            return True, sanitized, "System prompt echo scrubbed."

    # 2. Apply secret redaction and output sanitization
    sanitized = sanitize_output(clean)

    return True, sanitized, None


# ──────────────────────────────────────────────────────────────
# Indirect Prompt Injection Defense (Document Content Sanitizer)
# ──────────────────────────────────────────────────────────────

# Patterns that detect instruction-like text embedded inside document content.
# These should never appear in legitimate document passages but are common in
# indirect prompt injection payloads designed to hijack the LLM.
CONTEXT_INJECTION_PATTERNS = [
    # Direct instruction hijacks
    r"(?i)ignore\s+(all\s+)?(previous|prior|above|preceding)\s+(instructions|prompts|context|rules|guidelines)",
    r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|context|rules)",
    r"(?i)forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|context)",
    # Role/mode overrides
    r"(?i)you\s+are\s+now\s+(a|an|in)\s+",
    r"(?i)new\s+instructions?\s*:",
    r"(?i)system\s*:\s*(override|instruction|prompt|message)",
    r"(?i)<\s*system\s*>",
    r"(?i)\[system\s*(message|prompt|instruction)\]",
    # Output manipulation
    r"(?i)(always|must|should)\s+respond\s+with",
    r"(?i)the\s+answer\s+to\s+(every|all|any)\s+question\s+is",
    r"(?i)from\s+now\s+on\s*,?\s*(always|only|you\s+must)",
    r"(?i)IMPORTANT\s*:\s*(ignore|disregard|forget|override)",
    # Prompt extraction
    r"(?i)(reveal|print|output|show|repeat|echo)\s+(your\s+)?(system\s+prompt|instructions|initial\s+prompt|configuration)",
    # Jailbreak markers
    r"(?i)(DAN|developer|jailbreak|unfiltered|uncensored)\s+mode",
    r"(?i)bypass\s+(all\s+)?(safety|security|content\s+filter|guardrail)",
]


def sanitize_context_text(text: str) -> str:
    """
    Neutralizes indirect prompt injection attempts embedded inside retrieved
    document content before it enters the LLM prompt.

    Strategy: Detected injection patterns are wrapped in visible warning brackets
    rather than silently removed. This preserves the original text for audit
    purposes while preventing the LLM from following embedded instructions.
    """
    if not GUARDRAILS_ENABLED:
        return text

    sanitized = text
    for pattern in CONTEXT_INJECTION_PATTERNS:
        sanitized = re.sub(
            pattern,
            lambda m: f"[BLOCKED CONTENT: {m.group(0)}]",
            sanitized,
        )

    if sanitized != text:
        print(f"  [Guardrail] Neutralized embedded injection in document content")

    return sanitized
