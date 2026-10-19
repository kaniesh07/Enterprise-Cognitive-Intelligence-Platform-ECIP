# ECIP evaluation results

## Retrieval ablation (25 document questions, embeddings `onnx-package`, top-4)

| Strategy | Hit@1 | Hit@4 | MRR | nDCG@4 | Context precision | p50 latency |
|---|---|---|---|---|---|---|
| bm25_only | 100% | 100% | 1.000 | 0.995 | 76% | 1.2 ms |
| dense_only | 92% | 100% | 0.947 | 0.930 | 48% | 6.2 ms |
| hybrid_rrf | 96% | 100% | 0.980 | 0.953 | 50% | 5.9 ms |
Routing decision on all questions (no LLM involved): **100%** correct


