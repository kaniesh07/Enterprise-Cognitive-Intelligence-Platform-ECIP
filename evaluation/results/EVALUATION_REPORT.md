# ECIP v3 — evaluation report (17 Oct 2026)

## Setup
| | |
|---|---|
| System under test | ECIP v3 API (`/query/stream`, `/tools/search_knowledge_base`) on Python 3.11, embedded Chroma |
| LLM | TinyLlama-1.1B-Chat, **2-bit** quantised (IQ2_XXS) on Ollama 0.5.7, 2 vCPU, no GPU — the only model obtainable in the sandbox |
| Embeddings | all-MiniLM-L6-v2 (ONNX) |
| Corpus | `evaluation/datasets/corpus`: employee handbook (.md), IT security policy (2-page .pdf), travel policy (.docx with table), sales data (.csv), 3-page website (.html) |
| Questions | 35 in `evaluation/datasets/eval_set.json`: 25 document questions (1 follow-up), 7 general-knowledge / creative, 3 small talk |
| Runner | `python scripts/run_eval.py --api http://localhost:8000 --ingest` (raw outputs in this folder) |

## Results

### 1. Pipeline (model-independent parts)
| Metric | Result |
|---|---|
| Routing: chat vs documents vs small talk (end-to-end run) | **34 / 35 = 97%** — "What is 12 multiplied by 12?" was routed to the handbook because "12" matched "12 days of sick leave" |
| Routing after the fix (numbers no longer count as topic overlap) | **35 / 35 = 100%** (LLM-free routing check, and the 12×12 question re-tested live → chat) |
| Retrieval Hit@1 (right document ranked first) | **96%** (24 / 25) |
| Retrieval Hit@4 (right document in the context given to the model) | **100%** (25 / 25) |
| MRR / nDCG@4 | 0.980 / 0.953 |
| Context precision (share of the 4 retrieved chunks that are relevant) | 50% — expected with a 13-chunk corpus; irrelevant extras are pruned before generation |
| Citation accuracy (document answers citing a relevant document) | **96%** |
| Small-talk replies | 3 / 3 instant (≈2 ms, no LLM call) |
| Errors / failed requests | 0 / 35 |

### 2. Retrieval ablation (25 document questions, top-4)
| Strategy | Hit@1 | Hit@4 | MRR | nDCG@4 | Context precision | p50 latency |
|---|---|---|---|---|---|---|
| BM25 (keywords) only | 100% | 100% | 1.000 | 0.995 | 76% | 1.2 ms |
| Dense (MiniLM vectors) only | 92% | 100% | 0.947 | 0.930 | 48% | 6.2 ms |
| **Hybrid RRF (shipped default)** | 96% | 100% | 0.980 | 0.953 | 50% | 5.9 ms |

On this small, keyword-heavy policy corpus BM25 alone is best; dense search adds robustness for paraphrased
questions on larger corpora. With `nomic-embed-text` (the default embedding model, not available in the sandbox)
dense quality is typically higher than MiniLM.

### 3. Answer quality (dominated by the 2-bit 1.1B model)
| Grading | Document questions (25) | General questions (5 gradable) |
|---|---|---|
| Automatic (all required facts present in the answer) | 68% | 40% |
| **Strict manual grading** — correct and not contradicted | **40%** (10) | 20% (1) |
| Partially correct (right fact, but contradictory or with an invented extra) | 20% (5) | 20% (1) |
| Wrong | 40% (10) | 60% (3) |

The automatic metric over-counts (e.g. it accepted "the documents don't contain this… South" and an answer that
gave both "one hour" and "24 hours"), so the strict numbers are the honest ones.
**In every wrong document answer the passage containing the answer was in the model's context** (Hit@4 = 100%):
the errors are reading/arithmetics mistakes of TinyLlama-1.1B-IQ2 (e.g. "24 days annual leave" → "26 days sick
leave", "12 × 12 = 0", "water boils at 341.5 °C"), not retrieval or routing failures.

### 4. Grounding check (fast heuristic verifier)
| | Mean faithfulness score |
|---|---|
| All document answers | 0.76 |
| Answers that were correct | 0.80 |
| Answers that were wrong | 0.67 |

The heuristic separates correct from wrong answers only weakly (it checks that the answer's words come from the
sources, so a wrong number built from source words can still pass). It reliably flags answers that drift away from
the sources (e.g. 0.00 for the invented "Purge_Admin" approver). Use `VERIFIER_MODE=llm` with a capable model for
a stricter check (costs one extra LLM call).

### 5. Latency (2 vCPU, no GPU, TinyLlama)
| Question type | First token p50 / p95 | Full answer p50 / p95 | Mean tokens |
|---|---|---|---|
| Document answers | 5.4 s / 14.1 s | 9.3 s / 20.1 s | 391 |
| General chat | 1.2 s / 27.3 s* | 4.9 s / 29.2 s* | 219 |
| Small talk | 0.002 s | 0.002 s | 0 |

\*The p95 for chat is the misrouted 12×12 question (fixed). Retrieval itself takes 5–25 ms; nearly all time is the
model reading the prompt on CPU. For comparison, the original v2 build needed 124–189 s per question on the same machine.

## What to expect with the default model
Routing, retrieval and citations are model-independent and already at 96–100%. Answer accuracy is the part that
depends on the LLM: rerun `make eval` after starting the stack with `llama3.2:3b` (default) or `llama3.1:8b` to get
your real numbers — the same 35 questions, same scoring, results in `evaluation/results/latest.md`.

## Changes made during evaluation
* Routing: bare numbers no longer count as document overlap (fixes "12 × 12" going to the handbook).
* New evaluation suite: `evaluation/evaluate.py`, `scripts/run_eval.py`, `make eval`, bundled corpus and 35-question
  dataset (the old 5-question set referred to documents that did not exist and had no relevance labels).
* `/evaluate` endpoint now also reports routing accuracy, answer fact recall and document-level retrieval metrics.

---

# Part 2 — safety, ingestion, connectors, load and failure handling (17 Oct 2026)

Same machine and model as Part 1 (2 vCPU, no GPU, TinyLlama-1.1B 2-bit, MiniLM embeddings, embedded Chroma).
Reproduce with `make guardrail-eval` and `make ops-eval`. Raw data: `run_2026-10-17_guardrails_*.json`, `run_2026-10-17_ops.json`.

## 6. Safety filters (input guardrails)
| Set | Before fixes | After fixes |
|---|---|---|
| Harmful requests blocked (20: prompt injection, weapons, violence, self-harm, cyber-attack) | 15 / 20 (75%) | **20 / 20** |
| Harmless questions wrongly blocked (18, many with trigger words: "kill a process", "bomb calorimeter", "system prompt", "explosive growth"…) | 0 / 18 | **0 / 18** |
| Unseen harmless questions wrongly blocked (25, written after the fixes) | – | **0 / 25** |
| Personal data redacted before it reaches the model (email, card, phone, SSN) | 4 / 4 | **4 / 4** |
| Filter time per message | – | 0.07 ms mean, 1 ms max |

Fixed misses: role-play jailbreaks ("pretend you have no content policy"), "forget everything above…",
"instructions to manufacture …", "hack into a bank's network", "write ransomware …".
Caveat: the harmful set was used to tune the rules, so 20/20 is optimistic for new phrasings. These are
keyword/regex rules (fast, no model). For stronger protection, add a model-based moderation step.

## 7. Document import
| File | Size | Result | Chunks | Time | Found by search (rank 1) |
|---|---|---|---|---|---|
| IT_Security_Policy.pdf | 2 KB | ✅ | 2 | 0.11 s | ✅ |
| Travel_Policy.docx (with table) | 36 KB | ✅ | 2 | 0.13 s | ✅ |
| Sales_Q1_Q2.csv | 0.1 KB | ✅ | 1 | 0.03 s | ✅ |
| Acme_Employee_Handbook.md | 1 KB | ✅ | 4 | 0.10 s | ✅ |
| about.html / index.html / drone.html | 0.2 KB each | ✅ | 1 each | 0.02–0.06 s | ✅ |
| Release_Notes.txt | 0.1 KB | ✅ | 1 | 0.02 s | ✅ |
| Config_Reference.json | 0.1 KB | ✅ | 1 | 0.03 s | ✅ |
| Large_Manual.md (400 sections) | 151 KB | ✅ | 400 | 4.9 s (**81 chunks/s**) | ✅ ("unit 377" found among 400 near-identical sections) |

| Correctness check | Result |
|---|---|
| Re-uploading the same file creates no duplicates | ✅ |
| Uploading a new version replaces the old content (old text no longer found) | ✅ |
| Deleting removes the document from the list and from search | ✅ |
| Empty file / unsupported type (.exe) / file over 50 MB | rejected with 422 / 400 / 413 and a clear message ✅ |

## 8. Connectors
| Test | Result |
|---|---|
| Local folder: first sync of 3 files | 3 indexed |
| Local folder: re-sync with no changes | 2 skipped, **1 re-indexed unnecessarily** in the first run; 0 in 6 later trials (18 file checks) |
| Local folder: after editing 1 file and adding 1 | 2 indexed, 2 skipped ✅, edited text is what search returns ✅ |
| Website (3 pages via the crawler): first sync | 3 pages indexed ✅ |
| Website: re-sync with no changes | 3 skipped ✅ |

The one unnecessary re-index did not reproduce; it is harmless (the document is simply indexed again) but noted.
Not tested: S3, SharePoint, Google Drive, Confluence, SQL (the sandbox cannot reach those services).

## 9. Load (concurrent users)
| Workload | Concurrent users | Requests | Errors | Throughput | Median / p95 time |
|---|---|---|---|---|---|
| Small talk (instant replies) | 1 / 8 / 32 | 20 / 32 / 128 | 0 | 49 / 107 / 81 req/s | 0.02 / 0.06 / 0.23 s (p95 0.50 s at 32) |
| Search only (retrieval, no LLM) | 1 / 8 / 32 | 20 / 32 / 128 | 0 | 41 / 72 / 72 req/s | 0.02 / 0.10 / 0.27 s (p95 0.51 s at 32) |
| Streamed chat (LLM) | 1 / 2 / 4 | 2 / 4 / 8 | 0 | 0.11 / 0.24 / 0.13 answers/s | 9.0 / 9.3 / **30.4 s** (first word 5.1 / 0.3 / 8.6 s) |

The API itself is not the bottleneck: search and small talk stay well under 1 s with 32 simultaneous users.
LLM answers queue up once more people chat at once than the model server runs in parallel
(`OLLAMA_NUM_PARALLEL=2` here, on 2 CPU cores). For many users, use a GPU, raise `OLLAMA_NUM_PARALLEL`,
run several Ollama replicas, or switch to an OpenAI-compatible hosted model.

## 10. Failure handling
| Scenario | Behaviour |
|---|---|
| No / wrong API key | HTTP 401 ✅ |
| User stops a reply after 5 words | connection closed in 1.3 s; the model server stopped generating too (that request ended after 1.3 s instead of running to the 1,024-token limit) ✅ |
| LLM server down | `/health` → degraded; first 5 chats get a clear "Cannot reach the language model at … Is it running?" in 0.01 s; then the circuit breaker opens and requests fail fast with "temporarily unavailable" ✅ |
| While the LLM is down | small talk and document search keep working ✅ |
| LLM server back | chats work again automatically after **65 s** (60 s breaker cool-down), no restart needed ✅ |
