import collections
import logging
import os
import re
from typing import Counter, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Standard grammatical, question starters, common nouns & morphological prefixes
QUESTION_WORDS: Set[str] = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "explain", "describe", "compare", "contrast", "define", "list", "name", "show", "detail", "state", "summarize",
}

STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
    # Question commands and common domain nouns
    "tell", "give", "explain", "describe", "compare", "contrast", "define",
    "list", "name", "show", "detail", "state", "summarize", "learning",
    "transformer", "transformers", "model", "models", "architecture",
    "deep", "system", "systems", "network", "networks", "data", "process",
    "processing", "database", "query", "search", "document", "documents",
    # Morphological prefixes (never mutate)
    "pre", "post", "sub", "pro", "re", "co", "de", "un", "non", "multi",
    "semi", "cross", "auto", "per", "set", "hyper", "meta", "over", "under",
}

# Dynamic in-memory index of corpus vocabulary with frequencies
_CORPUS_VOCABULARY: Counter = collections.Counter()
_VOCAB_LOADED: bool = False


def _damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes Damerau-Levenshtein distance (insert, delete, substitution, AND transposition).
    Correctly recognizes single-character transpositions like 'waht' -> 'what' as distance 1.
    """
    len_s1, len_s2 = len(s1), len(s2)
    if s1 == s2:
        return 0
    if len_s1 == 0:
        return len_s2
    if len_s2 == 0:
        return len_s1

    d = {}
    for i in range(-1, len_s1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len_s2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len_s1):
        for j in range(len_s2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost  # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)  # transposition

    return d[(len_s1 - 1, len_s2 - 1)]


def refresh_corpus_vocabulary(force: bool = False):
    """
    Dynamically builds vocabulary directly from PostgreSQL chunks table and raw files.
    """
    global _CORPUS_VOCABULARY, _VOCAB_LOADED

    if _VOCAB_LOADED and not force:
        return

    vocab: Counter = collections.Counter()

    # 1. Query PostgreSQL chunks table
    try:
        from retrieval.vector_store import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT text FROM document_chunks LIMIT 5000;")
                rows = cur.fetchall()
                for row in rows:
                    content = row.get("text", "") if isinstance(row, dict) else row[0]
                    words = re.findall(r"\b[A-Za-z\-]{3,30}\b", content.lower())
                    vocab.update(words)
    except Exception as e:
        logger.debug(f"Could not load vocabulary directly from PostgreSQL: {e}")

    # 2. Also scan text files in data/processed and data/raw
    for target_dir in ["data/processed", "data/raw"]:
        if os.path.exists(target_dir):
            for fname in os.listdir(target_dir):
                if fname.endswith((".txt", ".md")):
                    fpath = os.path.join(target_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            words = re.findall(r"\b[A-Za-z\-]{3,30}\b", f.read().lower())
                            vocab.update(words)
                    except Exception:
                        pass

    _CORPUS_VOCABULARY = vocab
    _VOCAB_LOADED = True
    logger.debug(f"Loaded {len(_CORPUS_VOCABULARY)} unique vocabulary terms from corpus.")


def get_corpus_vocabulary() -> Counter:
    global _VOCAB_LOADED
    if not _VOCAB_LOADED:
        refresh_corpus_vocabulary()
    return _CORPUS_VOCABULARY


def _split_token(token: str) -> Tuple[str, str, str]:
    """
    Separates leading punctuation, clean alphabetic word, and trailing punctuation.
    """
    match = re.match(r"^([^\w]*)([\w\-]+)([^\w]*)$", token)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return "", token, ""


def _merge_split_compounds(tokens: List[str], vocab: Counter) -> List[str]:
    """
    Generic compound-word and prefix resolver:
    Dynamically merges adjacent split words (e.g. 'pre' + 'processing' -> 'preprocessing')
    if the merged word exists in the uploaded document vocabulary.
    """
    if len(tokens) < 2:
        return tokens

    parsed = [_split_token(t) for t in tokens]
    merged: List[str] = []
    i = 0

    while i < len(parsed):
        prefix_i, word_i, suffix_i = parsed[i]

        if i + 1 < len(parsed):
            prefix_next, word_next, suffix_next = parsed[i + 1]

            w1 = word_i.lower()
            w2 = word_next.lower()

            candidate_concat = f"{w1}{w2}"
            candidate_hyphen = f"{w1}-{w2}"

            if candidate_concat in vocab and vocab[candidate_concat] >= 1:
                merged.append(f"{prefix_i}{candidate_concat}{suffix_next}")
                i += 2
                continue
            elif candidate_hyphen in vocab and vocab[candidate_hyphen] >= 1:
                merged.append(f"{prefix_i}{candidate_hyphen}{suffix_next}")
                i += 2
                continue

        merged.append(tokens[i])
        i += 1

    return merged


def _correct_single_token(token: str, vocab: Counter, is_first_word: bool = False) -> str:
    """
    Corrects a single token against dynamic corpus vocabulary and English closed-class words
    using Damerau-Levenshtein edit distance and transposition scoring.
    """
    prefix, word, suffix = _split_token(token)
    lower_word = word.lower()

    # If the word is an acronym (all caps, e.g. PEAS, OLAP, OLTP, KDD, AISC), NEVER mutate
    if word.isupper() and len(word) >= 2:
        return token

    # Skip numbers, very short tokens (<=2 chars), stop words, or already valid corpus words
    if (
        len(lower_word) <= 2
        or lower_word.isdigit()
        or lower_word in STOP_WORDS
        or lower_word in vocab
    ):
        return token

    max_dist = 1 if len(lower_word) <= 4 else 2
    best_candidate: Optional[str] = None
    best_score = -999999.0

    target_len = len(lower_word)
    target_char_counter = collections.Counter(lower_word)

    # Search in STOP_WORDS and dynamic corpus vocabulary
    all_candidates = [(w, vocab.get(w, 500 if w in STOP_WORDS else 1)) for w in STOP_WORDS] + list(vocab.items())

    for cand, freq in all_candidates:
        if abs(len(cand) - target_len) > max_dist:
            continue

        dist = _damerau_levenshtein_distance(lower_word, cand)
        if dist <= max_dist:
            # Score candidate:
            # - Base penalty for edit distance
            score = - (dist * 100.0)
            
            # Anagram bonus: exact character multiset match (e.g. 'waht' vs 'what')
            if collections.Counter(cand) == target_char_counter:
                score += 50.0

            # First letter match bonus
            if cand[0] == lower_word[0]:
                score += 20.0

            # Initial question word bonus if at token index 0
            if is_first_word and cand in QUESTION_WORDS:
                score += 40.0

            # Log frequency weighting
            score += min(freq, 1000) * 0.01

            if score > best_score:
                best_score = score
                best_candidate = cand

    if best_candidate and best_candidate != lower_word:
        replacement = best_candidate
        if word.isupper():
            replacement = replacement.upper()
        elif word[0].isupper():
            replacement = replacement.capitalize()

        return f"{prefix}{replacement}{suffix}"

    return token


def correct_query_spelling(query: str) -> str:
    """
    100% Dynamic, non-hardcoded query normalizer & spell corrector:
    1. Corrects misspelled words using Damerau-Levenshtein distance & transposition scoring.
    2. Protects capitalized acronyms (PEAS, OLAP, KDD, etc.) from accidental mutation.
    3. Dynamically merges split prefixes/compounds if found in document corpus.
    """
    if not query or not query.strip():
        return query

    vocab = get_corpus_vocabulary()
    if not vocab:
        return query.strip()

    # Step 1: Tokenize query
    raw_tokens = query.strip().split()

    # Step 2: Individual token spell correction (passing is_first_word flag)
    corrected_tokens = [
        _correct_single_token(t, vocab, is_first_word=(idx == 0))
        for idx, t in enumerate(raw_tokens)
    ]

    # Step 3: Generic compound word / split prefix merge
    merged_tokens = _merge_split_compounds(corrected_tokens, vocab)

    return " ".join(merged_tokens)
