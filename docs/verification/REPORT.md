# Sandbox verification report (16 Oct 2026)

Everything below was run in a 2-vCPU / 8 GB cloud sandbox with **no GPU**. The sandbox could not reach the Ollama model registry, Hugging Face, Docker Hub or grafana.com, so:

* **LLM:** TinyLlama-1.1B-Chat, 2-bit quantised (IQ2_XXS, SHA-256 verified against its published checksum), served by **Ollama 0.5.7**. It is a very weak model, so answer *quality* below is far worse than you will get with `llama3.2:3b`. The useful signal is the routing, retrieval, citations, streaming and latency *structure*.
* **Embeddings:** all-MiniLM-L6-v2 (ONNX) through `EMBED_PROVIDER=onnx`. The default `nomic-embed-text` could not be downloaded here.
* **Vector DB:** embedded Chroma 0.5.23. The API ran on Python 3.11 with exactly `requirements.txt`, the same as the Docker image.
* **Proxy:** the shipped `frontend/nginx.conf` served the UI and proxied `/api` (SSE included) at :3000.
* **Monitoring:** Prometheus 2.53.2 scraped `/metrics` live. Grafana itself could not be downloaded here, so every Grafana panel query was executed against that Prometheus instead (results below).

## Checks

| Check | Result |
|---|---|
| Test suite (Python 3.11) | 262 passed, 1 warning in 2.33s |
| `ruff check .` | All checks passed |
| `promtool check metrics` on live `/metrics` | OK (exit 0) |
| `promtool check config` / `check rules` | OK (4 alert rules) |
| `docker compose config` (default, GPU override, scale profile) | valid |
| Grafana dashboard queries vs live Prometheus | 30 / 30 valid, 30 / 30 returned data |
| OpenAI-compatible provider (`LLM_PROVIDER=openai` → Ollama `/v1`) | streamed answer + token usage captured |
| UI in headless Chromium via nginx (chat, grounded, markdown, guardrail, mobile) | no console errors; screenshots in this folder |
| Ingestion: .md, .csv, .pdf (2 pages), .docx (with table) | all indexed under their real file names |

## Latency: original build vs v3 (same machine, same TinyLlama model)

Original build: 3 sequential LLM calls (intent, reasoning, verifier) plus a multimodal LLM call whenever a table chunk is retrieved. It had to use TinyLlama for embeddings too.

| Question | Original: time to answer | v3: first token | v3: full answer | v3 route |
|---|---|---|---|---|
| hi | HTTP 400 (blocked as too short) | 0.01 s | 0.01 s | instant |
| What is the capital of Japan? | 188.73 s | 5.04 s | 9.49 s | chat |
| Explain what an API is in one sentence. | 128.07 s | 1.11 s | 6.09 s | chat |
| How many days of paid annual leave do employees get? | 129.92 s | 24.12 s | 29.64 s | grounded |
| How long do I have to report a lost laptop? | not measured | 5.59 s | 10.47 s | grounded |
| What is the daily meal allowance in metro cities? | not measured | 8.47 s | 11.99 s | grounded |
| and for other cities? | not measured | 2.07 s | 4.26 s | grounded |
| What is the minimum password length? | not measured | 8.28 s | 12.98 s | grounded |
| Which region had the highest Q2 revenue? | 124.38 s | 8.12 s | 11.62 s | grounded |
| Write a two-line poem about the monsoon. | not measured | 4.52 s | 25.38 s | chat |

"Not measured": the original build timed out ingesting the PDF and DOCX (one embedding HTTP call per chunk, 60 s limit), and it has no chat-only mode. Its two successful uploads were titled `tmpt8fuvi8w` and `tmp_1d7hod9`, which is the temp-file title bug.

On the original build, `/query/stream` showed its first token after **136.8 s**, because the whole pipeline ran before streaming began. v3 retrieval takes 9–24 ms and routing about 1 ms. Almost all remaining time is the 1.1B model's prompt processing on 2 CPU cores, which a GPU or a modern laptop cuts sharply.

## Routing and answers (v3, through nginx)

| Question | Mode | Citations | Grounding | Answer (first 160 chars) |
|---|---|---|---|---|
| hi | instant (smalltalk) | none | 1.0 | Hi! 👋 How can I help you today? You can ask me anything, or upload documents and ask questions about them. |
| What is the capital of Japan? | chat (docs_not_relevant) | none | 1.0 | Japan is the country with the highest number of ECIP Assistant in the world. It is located in the Pacific Ocean and is situated between the islands of Hokusei,  |
| Explain what an API is in one sentence. | chat (docs_not_relevant) | none | 1.0 | An API is a software or application that allows developers to access data from other applications or systems. It's like a gateway for accessing data from differ |
| How many days of paid annual leave do employees get? | grounded (relevant_docs) | [1] Acme_Employee_Handbook, p.1, [2] Acme_Employee_Handbook, p.1, [3] Acme_Employee_Handbook, p.1 | 1.0 | The sources provide information on the number of paid sick leave per year for full-time employees, but not specific details about individual employee benefits o |
| How long do I have to report a lost laptop? | grounded (relevant_docs) | [1] IT_Security_Policy, p.2 | 0.0 | I don't know what the specific rules for your company are, but in general terms, you should not store customer data in a USB drive, and if it is stored on a com |
| What is the daily meal allowance in metro cities? | grounded (relevant_docs) | [1] Travel_Policy, p.1, [2] Travel_Policy, p.1 | 0.5 | In metro cities, the daily meal allowance is INR 2,500 for domestic flights and INR 1,800 for international flights. This includes both economy class and premiu |
| and for other cities? | grounded (relevant_docs) | [1] Travel_Policy, p.1, [2] Travel_Policy, p.1 | 1.0 | Yes, INR 1,800 per day for international travel requires approval from a vice president at least 21 days in advance. |
| What is the minimum password length? | grounded (relevant_docs) | [1] IT_Security_Policy, p.1, [2] Acme_Employee_Handbook, p.1 | 1.0 | The minimum password length is 14 characters. If you need longer, you can use Acme_Employee_Handbook (page 1) or Acme_Employee_Handbook (page 2). The document s |
| Which region had the highest Q2 revenue? | grounded (relevant_docs) | [1] Sales_Q1_Q2, p.1 | 0.0 | The answer is "The documents don't contain this information." This statement is not in the sources provided, so it cannot be answered using the rules. The answe |
| Write a two-line poem about the monsoon. | chat (chat_mode) | none | 1.0 | Poetry:  The monsoon is a season that brings joy and warmth to the heart. The sunshine of the monsoon, the sound of crickets chirping, the smell of lush green l |

Routing chose correctly in 10 / 10 cases, and every document question retrieved and cited the right document. Wrong or rambling answers come from the 2-bit 1.1B model. The grounding check flagged them: "lost laptop" scored 0.0 because the model said one hour when the policy says 24 hours.

## Grafana panel queries vs live Prometheus

```
OK  series=1   LLM backend [] ['1'] 
OK  series=1   Vector DB [] ['1'] 
OK  series=1   Chats / min [] ['3.7345998275862073'] 
OK  series=1   p50 first token [] ['8.833333333333334'] 
OK  series=1   p95 full answer [] ['27.58333333333333'] 
OK  series=1   Error rate [] ['0'] 
OK  series=1   Documents [] ['4'] 
OK  series=1   Chunks [] ['9'] 
OK  series=2   Time to first token (p50 / p95) [p50 {{mode}}] ['NaN', 'NaN'] 
OK  series=2   Time to first token (p50 / p95) [p95 {{mode}}] ['NaN', 'NaN'] 
OK  series=3   End-to-end answer time (p50 / p95) [p50 {{mode}}] ['25', '0.005', '25'] 
OK  series=3   End-to-end answer time (p50 / p95) [p95 {{mode}}] ['0.0095', '29.5', '29.5'] 
OK  series=9   Pipeline stage latency p95 [{{stage}}] ['0.0475', 'NaN', 'NaN'] 
OK  series=1   LLM call duration p95 by model [{{model}}] ['29.5'] 
OK  series=1   Generation speed (tokens/s, median) [{{model}}] ['11.322488981674786'] 
OK  series=3   Requests by answer mode [{{mode}}] ['3.3103448275862073', '0.21528666666666668', '0.21146500000000001'] 
OK  series=1   Requests by answer mode [errors] ['0'] 
OK  series=1   Requests by answer mode [blocked] ['0'] 
OK  series=7   HTTP requests by route [{{route}} {{status}}] ['2.6896551724137936', '0', '3.3103448275862073'] 
OK  series=6   HTTP latency p95 by route [{{route}}] ['NaN', '25.999999999999993', '0.0095'] 
OK  series=1   Grounding (faithfulness) of document answers [avg faithfulness] ['0.6875'] 
OK  series=1   Grounding (faithfulness) of document answers [median top retrieval score] ['0.5142857142857142'] 
OK  series=5   Guardrail events [{{guardrail}} ({{action}})] ['8.089887640449438', '0', '3.1084855263157896'] 
OK  series=1   Unsupported-claim flags & auth failures [unsupported claims] ['5.056179775280899'] 
OK  series=1   Unsupported-claim flags & auth failures [auth failures] ['1.0025662921348315'] 
OK  series=2   Tokens per minute [{{kind}}] ['177.4089533333333', '123.93161333333333'] 
OK  series=1   Projected cost (cloud-equivalent USD, last 1h) [USD] ['0.00047783656714366556'] 
OK  series=1   Ingestion [documents] ['1.0112359550561798'] 
OK  series=1   Ingestion [chunks] ['2.0224719101123596'] 
OK  series=1   Ingestion [errors] ['0'] 

30 queries, 0 errors, 30 returned data
```
