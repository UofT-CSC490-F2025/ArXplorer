# A4 – ArXiv paper dataset + baselines + GRPO finetuning

This repo prepares arXiv data, builds a weak-label dataset for query–abstract relevance, runs two baselines (local LLM and embedding classifier), and fine-tunes a strict yes/no judge with GRPO.

Key folders:
- `scripts/` — all pipelines (data prep, weak labels, baselines, GRPO)
- `data/` — inputs and derived artifacts
- `experiments/` — baseline outputs (preds, metrics)
- `runs/` — GRPO training runs (checkpoints, metrics, TensorBoard)

## Conda setup

Use the provided environment for consistent versions (WSL2/Linux or Windows):

```bash
conda env create -f environment.yml
conda activate a4-precision

# If you update the file later
conda env update -f environment.yml --prune
```

Notes:
- GPU PyTorch + CUDA 12.1 are installed via Conda channels.
- Pip section includes transformers/TRL/PEFT/bitsandbytes, sentence-transformers, rank-bm25, tqdm, etc.
- For Ollama-based steps, install and run Ollama separately on your host.

## Dataset creation

End goal: a CSV of weakly labeled pairs with columns `query_id,query,abstract_id,abstract,label` (1 relevant, 0 not) and a cleaned version for splitting.

1) Extract compact arXiv corpus (id,title,abstract) from Kaggle dumps

```bash
python ./scripts/kaggle_arxiv_to_csv.py \
  --src ./data/kaggle_arxiv \
  --out ./data/arxiv_100k_ml.csv \
  --limit 100000 \
  --preset_cs_ml_ai
```

2) Generate diverse ML queries (seed list via local LLM)

```bash
python -m scripts.weak_labels.generate_queries \
  --n 150 \
  --out ./data/weak_labels/queries_ml.csv \
  --ollama http://localhost:11434 \
  --model llama3:8b
```

Note: It is recommended to do a quality check on the generated queries and remove bad results.

3) Build candidates with BM25 + embedding retrieval

```bash
python -m scripts.weak_labels.build_candidates \
  --queries ./data/weak_labels/queries_ml.csv \
  --corpus ./data/arxiv_100k_ml.csv \
  --out ./data/weak_labels/candidates.csv \
  --k-bm25 50 --k-emb 50 --random-neg 10 \
  --embed-model all-MiniLM-L6-v2
```

4) Score signals: keyword overlap, LLM votes, optional cross-encoder

```bash
python -m scripts.weak_labels.score_signals \
  --candidates ./data/weak_labels/candidates.csv \
  --out ./data/weak_labels/scores.csv \
  --ollama http://localhost:11434 \
  --llm-model llama3:8b \
  --llm-votes 3 \
  --xenc --xenc-model cross-encoder/ms-marco-MiniLM-L-6-v2 --xenc-top 1000
```

5) Combine signals into labels (precision-oriented weights; thresholds configurable)

```bash
python -m scripts.weak_labels.combine_and_label \
  --scores ./data/weak_labels/scores.csv \
  --out ./data/weak_labels/query_abstract_pairs_weak.csv \
  --pos-th 0.8 --neg-th 0.2 --balance
```

6) Make a clean three-column dataset (adds/stabilizes query_id if missing)

```bash
python -m scripts.weak_labels.make_clean_dataset \
  --in ./data/weak_labels/query_abstract_pairs_weak.csv \
  --out ./data/weak_labels/query_abstract_pairs_clean.csv
```

Artifacts:
- `data/arxiv_*.csv` — compact corpus (id,title,abstract)
- `data/weak_labels/*.csv` — queries, candidates, scored signals, weak labels, clean pairs

## Baseline models

First split the cleaned dataset by query_id into train/val/test, then run the two baselines.

1) Split pairs by query_id

```bash
python -m scripts.pairs_verify_and_split \
  --pairs ./data/weak_labels/query_abstract_pairs_clean.csv \
  --out ./data/splits \
  --ratios 0.6 0.2 0.2
```

2) Embedding classifier baseline (cosine threshold)

```bash
python -m scripts.baseline_embed_cls \
  --splits ./data/splits \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --outdir ./experiments/embed_cls
```

Outputs: `experiments/embed_cls/preds_*.json`, `metrics.json` with precision, recall, f1, accuracy, threshold, counts.

3) Local LLM baseline (Ollama yes/no classifier)

```bash
python -m scripts.baseline_llm_local \
  --splits ./data/splits \
  --model llama3:8b \
  --outdir ./experiments/llm_local \
  --ollama-base-url http://localhost:11434
```

Outputs: `experiments/llm_local/preds_*.json`, `metrics.json` with precision, recall, f1, accuracy, counts.

## GRPO finetuning (yes/no judge)

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with 4-bit QLoRA using TRL’s GRPO.

Recommended run (tested on on 12 GB VRAM):

```bash
python -m scripts.train_judge_grpo \
  --splits ./data/splits \
  --out ./runs/grpo_qwen15b \
  --steps 3000 \
  --batch-size 16 \
  --micro-batch 2 \
  --eval-interval 250 \
  --max-abstract-chars 1200 \
  --max-prompt-tokens 1800 \
  --max-new-tokens 1 \
  --num-generations 2 \
  --dataloader-workers 0 \
  --log-steps 100
```

Notes and tips:
- The script disables cache and enables gradient checkpointing to reduce VRAM.
- If VRAM climbs mid-run, lower `--max-prompt-tokens` and/or `--micro-batch`, keep `--max-new-tokens 1`.
- Metrics logged to TensorBoard under `runs/.../tb` and to `metrics.csv` (periodic val eval).

## Troubleshooting

- Ollama from WSL2: ensure the base URL reaches the host (e.g., `http://localhost:11434`); see `docs/ollama-wsl2.md` if present.
- Memory spikes during GRPO: cap `--max-prompt-tokens` (e.g., 512–768), reduce `--micro-batch`, keep `--max-new-tokens 1` and `--num-generations 2`.
