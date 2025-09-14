# REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking

# REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking

Code for the EMNLP 2025 paper **"REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking."**

This repository now includes **two algorithms**:
1. **REALM** - The original recursive relevance modeling approach
2. **QI-ADR** - Quantum-Inspired Adaptive Document Reranking (new implementation)

---

## Repository Layout

~~~text
.
├── src/
│   ├── models.py                 # Flan-T5 / GPT wrappers (tokenization, logits, truncation)
│   ├── algorithm.py              # REALM core: realm(), rating updates, recursive sort, utils
│   ├── quantum_inspired_adr.py   # QI-ADR: quantum-inspired adaptive reranking
│   ├── main.py                   # CLI: load data, run algorithms, evaluate NDCG@10
│   └── run.sh                    # convenience launcher (calls main.py)
├── data/
│   ├── retrieve_results_dl19.json
│   ├── qrels_dl19.json
│   ├── retrieve_results_dl20.json
│   ├── qrels_dl20.json
│   └── ....
└── requirements.txt
~~~

---

## Installation

**Python:** 3.9–3.11 recommended.

1) Create a fresh environment.

2) Install the dependencies:

~~~bash
pip install -r requirements.txt
~~~

---

## Algorithms

### REALM (Original)
REALM operates within classical probability theory, modeling document relevance as Gaussian distributions and using Bayesian inference to recursively update parameters based on new evidence.

### QI-ADR (Quantum-Inspired Adaptive Document Reranking)
QI-ADR implements a quantum-inspired approach with four stages:

1. **Probabilistic State Representation**: Documents represented as vectors in high-dimensional space (quantum "superposition" analogue)
2. **Uncertainty Quantification**: Identifies most ambiguous documents for evaluation priority
3. **Iterative State Collapse**: Uses LLM to "measure" uncertain documents and propagates knowledge to remaining candidates
4. **Final Ranking Aggregation**: Combines LLM scores with inferred scores for final ranking

**Key Parameters for QI-ADR:**
- `--llm-budget`: Maximum LLM evaluations (default: 20)
- `--batch-size`: Documents evaluated per iteration (default: 5)
- `--learning-rate`: Knowledge propagation rate (default: 0.1)
- `--embedding-model`: Embedding model for vector representation (default: "all-MiniLM-L6-v2")

---

## Models

- **Open-source (local):** `google/flan-t5-large|xl|xxl` (auto-downloaded by 🤗 Transformers).
- **API option:** `gpt-5` (via OpenAI). Set your key:

~~~bash
export OPENAI_API_KEY=sk-...
~~~

**Offline hint:** Pre-download models to a local directory and pass that path to `--model`.

---

## Data Format

- **Retrieve results** (`retrieve_results_*.json`): list of queries and candidate hits

~~~json
[
  {
    "query": "what is llm-based re-ranking?",
    "hits": [
      {"qid": 123, "docid": "D1", "content": "passage text ..."},
      {"qid": 123, "docid": "D2", "content": "another passage ..."}
    ]
  }
]
~~~

- **Qrels** (`qrels_*.json`): relevance labels per query id

~~~json
{
  "123": { "D1": 3, "D9": 2, "D2": 0 }
}
~~~

`main.py` attaches labels when present and evaluates **NDCG@10** using `pytrec_eval`.

---

## Quickstart

### A) Using `run.sh` (recommended)

From the repo root:

~~~bash
cd src
bash run.sh
~~~

### B) Direct CLI

#### Running REALM (Original Algorithm)

~~~bash
cd src
python main.py \
  --algorithm realm \
  --dataset dl19 \
  --model google/flan-t5-large \
  --order bm25
~~~

#### Running QI-ADR (Quantum-Inspired Algorithm)

~~~bash
cd src
python main.py \
  --algorithm qi-adr \
  --dataset dl19 \
  --model google/flan-t5-large \
  --order bm25 \
  --llm-budget 20 \
  --batch-size 5 \
  --learning-rate 0.1
~~~

**Arguments**

**Common:**
- `--dataset, -d`: dataset name (e.g., `dl19`, `dl20`)
- `--model, -m`: `google/flan-t5-large|xl|xxl` or `gpt-5`
- `--order, -o`: initial order `bm25`, `random`, or `inverse`
- `--algorithm, -a`: `realm` or `qi-adr`

**QI-ADR Specific:**
- `--llm-budget`: Maximum LLM evaluations (default: 20)
- `--batch-size`: Documents per iteration (default: 5)
- `--top-k`: Number of top documents to return (default: 10)
- `--learning-rate`: Knowledge propagation rate (default: 0.1)
- `--embedding-model`: Vector embedding model (default: "all-MiniLM-L6-v2")

**Output Example (REALM)**

~~~text
REALM NDCG@10: 0.7050

Inference Count: 80.4
Tokens in Prompt: 25823.0
Latency (s): 4.0
Depth: 2.3
~~~

**Output Example (QI-ADR)**

~~~text
Running Quantum-Inspired ADR with google/flan-t5-large
LLM Budget: 20, Batch Size: 5

QI-ADR Statistics:
Average LLM calls per query: 18.50
Average tokens per query: 12400.0
Average latency per query: 2.150s

QI-ADR NDCG@10: 0.6890
~~~

---

## Algorithm Comparison

| Feature | REALM | QI-ADR |
|---------|-------|--------|
| **Approach** | Recursive probabilistic ranking | Quantum-inspired adaptive learning |
| **LLM Usage** | Variable (depends on recursion) | Fixed budget per query |
| **Efficiency** | Adaptive stopping | Controlled by budget |
| **Knowledge Propagation** | TrueSkill rating updates | Vector space updates |
| **Uncertainty Modeling** | Gaussian distributions | Vector similarity-based |