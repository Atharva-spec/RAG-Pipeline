import re
from typing import List, Tuple


def extract_cited_indices(text: str, max_index: int) -> List[int]:
    """
    Extracts citation numbers from text (e.g. [1], [2], [1, 2], [1][2], [Source 1]).
    Returns a sorted list of unique 1-based chunk indices that exist in the retrieved context.
    """
    # Matches patterns like [1], [2], [1, 2], [1,2,3], [Source: 1], [Source 1], 【1†L1-L4】, 【2】
    patterns = [
        r'\[(\d+)\]',
        r'\[(?:Source|Doc|Ref)\s*:?\s*(\d+)\]',
        r'\[(\d+(?:\s*,\s*\d+)+)\]',
        r'【(\d+)(?:[†^][^】]*)?】',
        r'\[(\d+)\s*[†^][^\]]*\]',
    ]

    indices = set()

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            if "," in match:
                for sub in match.split(","):
                    val = sub.strip()
                    if val.isdigit():
                        idx = int(val)
                        if 1 <= idx <= max_index:
                            indices.add(idx)
            elif match.isdigit():
                idx = int(match)
                if 1 <= idx <= max_index:
                    indices.add(idx)

    # If no explicit brackets were generated, fallback to all returned qualified chunks
    if not indices and max_index > 0:
        indices = set(range(1, max_index + 1))

    return sorted(indices)


def resolve_citations(answer_text: str, chunks: List[dict]) -> Tuple[List[dict], str]:
    """
    Resolves citation references from the LLM answer and returns:
    1. A structured list of citation metadata objects.
    2. A formatted Markdown citations footer.
    """
    if not chunks:
        return [], ""

    max_idx = len(chunks)
    cited_indices = extract_cited_indices(answer_text, max_idx)

    citations = []
    for idx in cited_indices:
        chunk = chunks[idx - 1]
        source = chunk.get("source", "Unknown Document")
        section = chunk.get("section_header", "General")
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        doc_hash = chunk.get("doc_hash", "")
        conf_score = chunk.get("confidence_score")
        
        # Calculate display confidence percentage
        conf_pct = f"{int(conf_score * 100)}%" if conf_score is not None else "N/A"

        # Build clean snippet excerpt (first 160 chars)
        text_body = chunk.get("text", "")
        excerpt = " ".join(text_body.split())
        if len(excerpt) > 160:
            excerpt = excerpt[:157] + "..."

        citation_entry = {
            "index": idx,
            "source": source,
            "section_header": section,
            "chunk_id": chunk_id,
            "doc_hash": doc_hash[:8] if doc_hash else "",
            "confidence_score": conf_score,
            "confidence_display": conf_pct,
            "excerpt": excerpt,
            "exact_quote": excerpt,
            "quote": excerpt,
            "text": text_body,
            "parent_context": chunk.get("parent_text", text_body),
            "strategy": chunk.get("strategy", "hierarchical"),
        }
        citations.append(citation_entry)

    footer = format_citations_footer(citations)
    return citations, footer


def format_citations_footer(citations: List[dict]) -> str:
    """
    Renders a clean, verified Markdown citations footer.
    """
    if not citations:
        return ""

    lines = ["\n\n📚 Sources & Citations:"]
    for c in citations:
        idx = c["index"]
        source = c["source"]
        section = c.get("section_header", "General")
        conf = c.get("confidence_display", "N/A")
        excerpt = c.get("excerpt", "")

        section_str = f" | Section: {section}" if section and section != "General" else ""
        lines.append(f"  [{idx}] {source}{section_str} (Confidence: {conf})")
        if excerpt:
            lines.append(f'      "{excerpt}"')

    return "\n".join(lines)
