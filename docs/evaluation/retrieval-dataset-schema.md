# Retrieval evaluation dataset

The RAG evaluator accepts UTF-8 JSONL with one reviewed query per line:

```json
{
  "query_id": "query-0001",
  "query": "What does Article 1 provide?",
  "relevant_sources": {
    "document-id:chunk-000001": 3,
    "document-id:chunk-000015": 1
  },
  "label_provenance": "reviewer name or review record"
}
```

`relevant_sources` maps candidate chunk IDs to positive integer relevance grades.
Use a higher grade for a more directly relevant source. Query IDs must be unique;
each query must have at least one labeled relevant source. The evaluator does not
invent labels and refuses to report metrics for an empty dataset.

Metrics include Precision@K, Recall@K, MRR, nDCG@K, query retrieval coverage, and
citation traceability. These measure retrieval behavior against supplied labels,
not legal correctness, completeness of law, or validity of a legal conclusion.
