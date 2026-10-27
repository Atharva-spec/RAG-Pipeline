# End-to-end check — 2026-10-28

## Scope and environment

- Ran the production query handler (`api.server.execute_query`) against the configured local PostgreSQL/pgvector and Redis services and configured Groq model. Calls were in-process, not through a separately started HTTP server.
- PostgreSQL and Groq health checks passed. The indexed table contained 460 chunks. The index returned Ayurveda/healthcare documents; it did not return the management or logic notes named by the checked-in benchmark.
- Timings below are single-run wall-clock timings from this machine, including model startup/network effects. They are not a load test or a latency percentile. The first calls include cold-start effects.
- `use_cache=False` was supplied. The live query handler does not currently use that request flag or call the answer-cache helpers.

## Query log

| # | Query | Result | Wall time | Pipeline telemetry | Citation |
|---|---|---|---:|---|---|
| 1 | What are the three essential managerial skills according to Robert Katz? | Refused: “I could not find sufficient information in your uploaded documents to answer that question accurately. Please make sure the relevant document is uploaded or try refining your question.” | 10,204.78 ms | 8 candidates; confidence 0.000; refusal | None |
| 2 | What does First-Order Logic represent and what are its key components? | Same safe refusal as #1. | 107.73 ms | 8 candidates; confidence 0.000; refusal | None |
| 3 | What is the capital of France and what is its current weather? | Same safe refusal as #1. | 101.58 ms | 8 candidates; confidence 0.000; refusal | None |
| 4 | Hi | “Hello! I am your Enterprise Document Assistant. How can I help you with your documents today?” | 1.10 ms | Intent fast path; no retrieval | None |
| 5 | What does cross-referral ensure in a hierarchical health care delivery system? | “Cross‑referral ensures that the hierarchical health‑care delivery system provides **adequate and quality health care to all health seekers, as appropriate**[1].” | 8,503.32 ms | 8 candidates; confidence 0.9999; answered | `Ayurveda_and_Conventional_Medicine.pdf`; quote starts “Cross-referral is an important component of the hierarchical health care delivery system…” |

## Automated tests

Command: `./venv/bin/python -m unittest discover -s tests -v`

- 69 tests ran; 66 passed and 3 errored.
- All three errors were RAG evaluation metric tests that attempted to fetch `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face and failed because network name resolution was unavailable in the test process. The ordinary pipeline, API-mocked, guardrail, cache, and ingestion tests passed.
- Test output also showed deprecation warnings for FastAPI `on_event` and a warning that `libmagic` is unavailable (the text parser test still passed).

## Readout

- The current indexed healthcare corpus can produce at least one grounded, cited answer, and the out-of-domain/unsupported questions are refused.
- Relevance is not validated by this run: benchmark questions for Robert Katz and First-Order Logic failed because their expected documents are not in the live index. Re-ingest a representative customer corpus and run an aligned benchmark before making quality claims.
- Cold grounded answers took 8.5 seconds; the first unsupported query took 10.2 seconds. Warm unsupported queries took about 0.10 seconds and greeting routing took 1.1 ms. These timings do not establish low or predictable latency under concurrent load.
- An unsupported question still retrieved eight candidates before refusing. The refusal behavior is useful, but quality and threshold calibration require representative in-domain and out-of-domain data.
