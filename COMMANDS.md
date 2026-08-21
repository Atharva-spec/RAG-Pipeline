# ⚡ Enterprise RAG Pipeline — CLI Commands & Operations Guide

A complete, copy-pasteable reference cheat sheet for all commands, flags, background services, evaluation suites, and interactive shortcuts in the RAG pipeline.

---

## 🚀 Quick Reference Card

| Goal | Command | Description |
| :--- | :--- | :--- |
| **Start Chat** | `./rag --chat` | Interactive multi-turn conversational session with memory |
| **Hands-Free Watcher** | `./rag --watch` | Auto-indexes new documents the second you drop them in `data/raw/` |
| **Single Question** | `./rag --query "<text>"` | Fast one-off answer with verified citations & sources |
| **Incremental Sync** | `./rag --ingest` | Scans disk & skips unchanged files in **<50ms** |
| **Rebuild Index** | `./rag --rebuild` | Wipes OpenSearch index and re-ingests everything from scratch |
| **RAGAS Evaluation** | `./rag --evaluate` | 100% local benchmarking (Faithfulness, Relevance, Precision, Recall) |
| **Calibrate Threshold** | `./rag --calibrate` | Automated precision-recall threshold sweep for zero hallucination |
| **Run Unit Tests** | `python -m unittest discover -s tests -v` | Executes all 35 automated unit tests |

---

## 1. Background Services Setup

Ensure Docker and Ollama are running before querying:

```bash
# 1. Start OpenSearch Container
docker start native-rag-pipeline

# 2. Start Local Ollama LLM
ollama serve
```

---

## 2. Interactive & Search Commands

### A. Conversational Terminal Chat
Starts a continuous multi-turn chat session with automatic conversation memory (5-turn buffer) and intelligent follow-up query rewriting:
```bash
./rag --chat
```
*(Or `python main.py --chat`)*

#### In-Chat Special Commands:
* `Type your question`: Asks a question with grounded citations.
* `/clear` or `/reset`: Clears conversational memory and starts a fresh session.
* `exit`, `quit`, or `q`: Ends the chat session.

---

### B. Single-Query Mode (One-Shot)
Ask a direct question without entering an interactive loop:
```bash
./rag --query "What are the three essential managerial skills according to Robert Katz?"
```
*(Returns the grounded answer along with the verified `📚 Sources & Citations` footer).*

---

## 3. Document Ingestion & Live Sync

### A. Live Background File Watcher *(Hands-Free)*
Monitors `data/raw/` in real time. Whenever you drag & drop a PDF, Word doc, PowerPoint presentation, or Markdown file into the folder, it is **auto-indexed in the background**:
```bash
./rag --watch
```

---

### B. Smart Incremental Ingestion
Runs a SHA-256 hash differential check. Skips unchanged files in **0.05s**, updates modified files, and purges deleted documents from OpenSearch:
```bash
./rag --ingest
```

---

### C. Clean Rebuild from Scratch
Wipes the existing OpenSearch index, resets mappings to HNSW (`faiss`), and re-parses every document in `data/raw/`:
```bash
./rag --rebuild
```

---

## 4. Quantitative Evaluation & Calibration

### A. Local RAGAS Benchmark Evaluation Suite
Evaluates the entire pipeline against test datasets and outputs a quantitative scorecard:
```bash
./rag --evaluate
```
* **Evaluated Metrics**:
  * **Faithfulness (Grounding)**: Zero hallucination verification against context passages.
  * **Answer Relevance**: Semantic alignment between user query and generated answer.
  * **Context Precision**: Evaluates whether relevant chunks rank at the top vs bottom.
  * **Context Recall**: Checks if all necessary ground-truth facts are captured.
* Automatically exports JSON reports to `data/eval_reports/`.

---

### B. Confidence Threshold Calibration
Sweeps cross-encoder sigmoid confidence thresholds from $0.10 \dots 0.90$ across in-domain and out-of-domain queries to find the optimal zero-hallucination threshold:
```bash
./rag --calibrate
```

---

## 5. Automated Unit Tests

Run the full automated test suite (all 35 unit tests):

```bash
# If inside venv:
python -m unittest discover -s tests -v

# Or directly:
./venv/bin/python -m unittest discover -s tests -v
```

---

## 6. Optional: Global Shortcut (`rag` anywhere)

To make `rag` executable from **any terminal directory** on your Mac, add this line to `~/.zshrc`:

```bash
alias rag="/Users/atharva/RAG-pipeline/rag"
```

Then reload your shell:
```bash
source ~/.zshrc
```

Now you can simply type:
```bash
rag --chat
rag --watch
rag --evaluate
```
