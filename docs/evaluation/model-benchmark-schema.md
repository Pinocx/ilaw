# Reviewed LLM Benchmark Dataset

`llm-evaluate` consumes JSONL. The first row declares the label provenance;
every subsequent row is one human-reviewed model response.

```json
{"record_type":"metadata","schema_version":"1","label_provenance":"Independent reviewer batch 2026-11"}
{"query_id":"q-001","run_id":"research-abc","provider":"ollama","model":"qwen2.5:14b","expected_source_ids":["doc-a::chunk-4"],"generated_source_ids":["doc-a::chunk-4"],"citations_verified":true,"instruction_compliance":true,"legal_quality_rating":4,"latency_ms":2100,"input_tokens":900,"output_tokens":240,"estimated_cost":null,"human_reviewed":true}
```

The evaluator reports citation precision/recall/F1, repository citation
verification, instruction compliance, reviewer-rated legal quality, latency,
token use, and configured estimated cost by provider/model. It rejects records
not explicitly marked as human reviewed. It never creates labels or claims legal
correctness. Do not include query text, confidential case facts, prompts, or
source text in the benchmark file unless the evaluation owner has approved that
data handling.
