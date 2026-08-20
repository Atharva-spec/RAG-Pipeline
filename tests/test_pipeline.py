import unittest
from ingestion.document_loader import _clean
from ingestion.chunker import chunk_documents
from generation.prompt_builder import build_prompt, SYSTEM_PROMPT
from retrieval.reranker import rerank_chunks
from retrieval.vector_store import hybrid_search


class TestDocumentLoader(unittest.TestCase):
    def test_text_cleaning(self):
        raw_text = "Hello    world!\n\n\n\nThis is   a test.\t\tFine."
        cleaned = _clean(raw_text)
        self.assertEqual(cleaned, "Hello world!\n\nThis is a test. Fine.")
        self.assertNotIn("\n\n\n", cleaned)

    def test_unstructured_parser_with_text_file(self):
        import tempfile
        from pathlib import Path
        from ingestion.document_loader import _parse_with_unstructured

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("# Heading\n\nThis is an unstructured test document with a paragraph.\n- Item 1\n- Item 2")
            temp_path = Path(f.name)

        try:
            parsed = _parse_with_unstructured(temp_path)
            self.assertIn("Heading", parsed)
            self.assertIn("unstructured test document", parsed)
        finally:
            temp_path.unlink(missing_ok=True)



class TestChunker(unittest.TestCase):
    def test_chunking_with_metadata(self):
        docs = [
            {
                "source": "sample.txt",
                "text": "Paragraph one.\n\nParagraph two with some more detailed content that will be chunked.",
            }
        ]
        chunks = chunk_documents(docs)
        self.assertGreater(len(chunks), 0)
        first = chunks[0]
        self.assertIn("id", first)
        self.assertIn("chunk_index", first)
        self.assertIn("char_count", first)
        self.assertIn("text", first)
        self.assertEqual(first["source"], "sample.txt")
        self.assertEqual(first["chunk_index"], 0)
        self.assertEqual(first["char_count"], len(first["text"]))


class TestPromptBuilder(unittest.TestCase):
    def test_build_prompt_structure(self):
        question = "What is RAG?"
        chunks = [
            {"source": "rag_intro.pdf", "text": "RAG stands for Retrieval-Augmented Generation."},
            {"source": "rag_intro.pdf", "text": "It combines retrieval and generation models."},
        ]
        messages = build_prompt(question, chunks)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Context:", messages[1]["content"])
        self.assertIn("[1] Source: rag_intro.pdf", messages[1]["content"])
        self.assertIn("Question: What is RAG?", messages[1]["content"])


class TestReranker(unittest.TestCase):
    def test_empty_chunks(self):
        result = rerank_chunks("test query", [])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
