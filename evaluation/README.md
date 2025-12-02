# ArXplorer Evaluation Framework

Automated evaluation of the ArXplorer retrieval system using two complementary metrics:
- **Precision@10**: Relevance quality of retrieved results (LLM-judged)
- **Recall@10 + MRR**: Ability to retrieve canonical papers with query variants

## Overview

### Metrics

1. **Precision@10** (queries_ml_100.csv)
   - 100 machine learning queries
   - Top-10 results judged for relevance by LLM (qwen2.5-1.5b-instruct)
   - Measures: Average proportion of relevant papers in results

2. **Recall@10 and MRR** (canon_papers_60.csv)
   - 57 canonical papers (3 missing from 300k dataset)
   - ~4 query variants per paper (template-based + typos + exact title)
   - Measures: 
     - **Recall@10**: Percentage of papers found with any variant
     - **MRR**: Mean Reciprocal Rank (ranking quality)
     - **Query-level recall**: Success rate across all queries

### Architecture

```
evaluation/
├── data/
│   ├── queries_ml_100.csv          # 100 ML queries for precision
│   ├── canon_papers_60.csv         # 60 canonical papers
│   └── canonical_queries.json      # Generated query variants (228 queries)
│
├── scripts/
│   ├── utils.py                    # API utilities, arXiv ID handling
│   ├── query_generator.py          # Generate canonical query variants
│   ├── llm_judge.py                # LLM relevance judgments
│   ├── precision_evaluator.py      # Precision@10 evaluation
│   ├── recall_evaluator.py         # Recall@10 + MRR evaluation
│   └── run_evaluation.py           # Main orchestrator
│
├── results/                         # Evaluation outputs (created on run)
│   ├── precision_results.json
│   ├── recall_results.json
│   └── summary_YYYYMMDD_HHMMSS.json
│
└── requirements.txt                 # torch, transformers, requests
```

## Setup

### 1. Install Dependencies

```powershell
# Install evaluation requirements
conda activate arxplorer-env
```

**Note**: Requires ~4GB disk space for qwen2.5-1.5b-instruct model (first run only).

### 2. Generate Canonical Queries

```powershell
cd evaluation/scripts
python query_generator.py
```

This creates `evaluation/data/canonical_queries.json` with ~240 query variants.

### 3. Configure API Endpoint

**Option A: Auto-detect from terraform** (if running from repo with terraform outputs)
```powershell
# No action needed, automatically reads terraform output
```

**Option B: Manual endpoint**
```powershell
$env:ARXPLORER_API_ENDPOINT = "http://your-api-endpoint.com"
```

## Running Evaluations

### Full Evaluation (both metrics)

```powershell
cd evaluation/scripts
python run_evaluation.py
```

**Output:**
- `evaluation/results/precision_results.json` - Detailed P@10 results
- `evaluation/results/recall_results.json` - Detailed R@10/MRR results  
- `evaluation/results/summary_YYYYMMDD_HHMMSS.json` - Combined metrics

**Expected runtime**: 
- Precision@10: ~15-25 minutes (100 queries × 10 results × LLM judgment)
- Recall@10: ~10-15 minutes (228 queries, no LLM)
- **Total**: ~30-40 minutes

### Individual Metrics

**Precision@10 only:**
```powershell
python run_evaluation.py --skip-recall
```

**Recall@10 + MRR only:**
```powershell
python run_evaluation.py --skip-precision
```

### Options

```powershell
python run_evaluation.py --help
```

- `--api-endpoint URL` - Override API endpoint
- `--skip-precision` - Skip Precision@10 evaluation
- `--skip-recall` - Skip Recall@10/MRR evaluation
- `--top-k N` - Number of results to retrieve (default: 10)

## Output Format

### Precision Results

```json
{
  "avg_precision_at_10": 0.73,
  "num_queries": 100,
  "queries_with_zero_relevant": 5,
  "queries_with_all_relevant": 12,
  "detailed_results": [
    {
      "query_id": "Q001",
      "query": "neural networks for image classification",
      "precision": 0.8,
      "relevant_count": 8,
      "total_results": 10,
      "papers": [
        {
          "rank": 1,
          "doc_id": "1512.03385",
          "title": "Deep Residual Learning...",
          "relevant": true,
          "reasoning": "ResNet is a foundational..."
        }
      ]
    }
  ]
}
```

### Recall Results

```json
{
  "recall_at_10": 0.89,
  "mrr": 0.45,
  "num_canonical_papers": 57,
  "papers_found_any_variant": 51,
  "query_level_recall": 0.67,
  "detailed_results": [
    {
      "entry": "1",
      "target_id": "1301.3781",
      "target_title": "Distributed Representations of Words...",
      "found_in_any_variant": true,
      "best_rank": 1,
      "avg_reciprocal_rank": 0.75,
      "query_results": [
        {
          "query": "word2vec paper",
          "variant_type": "template_{keyword} paper",
          "found": true,
          "rank": 1,
          "reciprocal_rank": 1.0
        }
      ]
    }
  ]
}
```

### Combined Summary

```json
{
  "timestamp": "2025-01-28T14:30:00",
  "api_endpoint": "http://3.95.123.45",
  "precision": {
    "avg_precision_at_10": 0.73,
    "num_queries": 100
  },
  "recall": {
    "recall_at_10": 0.89,
    "mrr": 0.45,
    "num_canonical_papers": 57,
    "papers_found": 51
  }
}
```

## Implementation Details

### LLM Judge (qwen2.5-1.5b-instruct)

**Why qwen2.5?**
- Small (1.5B params, ~4GB), fast inference (~500ms/judgment)
- Instruction-tuned for structured output
- Free, no API costs

**Prompt structure:**
```
Is paper X relevant to query Y?
Title: [title]
Abstract: [abstract]
Query: [query]

Respond: Yes/No + reasoning
```

**Parsing:** Extracts binary decision from free-form LLM response.

### Query Variants

**Template-based:**
- `"original {keyword} paper"`
- `"seminal {keyword} work"`
- `"{keyword} {year}"`
- `"{keyword} research paper"`

**Typo simulation:**
- Random character swap/delete/duplicate in title
- Example: "Attention is All You Need" → "Atention is All You Neeed"

**Exact title:** Use full paper title as query.

**Total:** ~4 variants × 57 papers = 228 queries

### API Integration

- Queries cloud API endpoint (auto-detected from terraform)
- 0.3-0.5s delay between queries to avoid overload
- Graceful error handling with detailed logging
- Returns top-10 results per query

### Missing Papers

3 papers from canon_papers_60.csv not found in arxiv_300k.jsonl:
- Entry 28: 2402.03300 (Orca-Math)
- Entry 46: 2312.00752 (Mamba)
- Entry 48: 2305.14314 (QLoRA)

These are newer papers (2023-2024) beyond the dataset cutoff.

## Interpreting Results

### Good Performance Targets

- **Precision@10**: ≥ 0.70 (70% relevant papers)
- **Recall@10**: ≥ 0.85 (85% canonical papers found)
- **MRR**: ≥ 0.40 (average rank ~2-3 for found papers)

### Failure Analysis

**Low Precision:**
- Too many irrelevant papers in top-10
- Check reranking weights, intent boosting
- Review LLM query analysis accuracy

**Low Recall:**
- Canonical papers not retrieved
- Check embedding quality (SPECTER2 adapters)
- Verify sparse encoding (SPLADE)
- Test metadata template effectiveness

**Low MRR (but high Recall):**
- Papers found but ranked poorly
- Adjust reranking weights
- Tune intent booster multipliers
- Check citation count boosting

## Extending Evaluation

### Add New Query Set

1. Create CSV: `query_id,query`
2. Adapt `precision_evaluator.py` or `recall_evaluator.py`
3. Run with `--skip-X` to isolate new metric

### Add New Metrics

**NDCG (Normalized Discounted Cumulative Gain):**
- Requires graded relevance (0-3 scale instead of binary)
- Modify LLM judge prompt for graded output

**F1@K:**
- Combine precision and recall at same K
- `F1 = 2 * (P * R) / (P + R)`

**Citation-weighted Precision:**
- Weight relevance by paper citation count
- Higher weight for foundational papers

### Custom LLM Judge

Replace qwen2.5 with different model:

```python
# In llm_judge.py
self.model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Example
```

**Considerations:**
- Model size vs speed tradeoff
- Instruction-following capability
- Consistency across judgments

## Troubleshooting

### LLM Judge Fails to Load

**Error:** `transformers` not installed
```powershell
pip install torch transformers
```

**Error:** Out of VRAM
- Use CPU: Set `device="cpu"` in `llm_judge.py`
- Use smaller model: qwen2.5-0.5b (2GB)

### API Connection Errors

**Error:** `Connection refused`
- Check API is running: `curl http://your-endpoint/health`
- Verify security group allows your IP
- Test with browser: `http://your-endpoint/docs`

**Error:** `API endpoint not found`
- Set manually: `$env:ARXPLORER_API_ENDPOINT = "http://..."`
- Or use `--api-endpoint` flag

### Canonical Queries Not Found

**Error:** `canonical_queries.json` missing
```powershell
cd evaluation/scripts
python query_generator.py
```

### Slow Evaluation

**Precision@10 takes >30 minutes:**
- Normal for 100 queries × 10 results × LLM judgment
- Reduce queries: Edit `queries_ml_100.csv`
- Use GPU for LLM judge (3-5x speedup)

**Recall@10 takes >20 minutes:**
- Normal for 228 queries
- Reduce canonical papers: Edit `canon_papers_60.csv`
- Increase API rate limit if throttling

## Citation

If using this evaluation framework, please cite ArXplorer:

```bibtex
@software{arxplorer2025,
  title = {ArXplorer: Multi-Stage Hybrid Search for Academic Papers},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/arxplorer}
}
```
