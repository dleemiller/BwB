# BwB (Beam-search with BM25)

**BwB** is a Python library designed to combine **BM25**-based retrieval with **beam-search expansions** and a **reward model** for ranking. It allows you to:

- **Index** textual corpora using [BM25S](https://github.com/xhluca/bm25s) for lightning-fast retrieval.  
- **Expand** search queries in multiple “beam-search” steps using a language model.  
- **Score** and **rank** documents via a reward model like [sentence-transformers CrossEncoder](https://www.sbert.net/docs/pretrained_cross-encoders.html) or any custom model.  
- **Summarize** results, or otherwise generate final textual output about the top-ranked documents.  

---

## Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
3. [Quickstart](#quickstart)  
4. [Configuration](#configuration)  
5. [Core Concepts](#core-concepts)  
   - [1. BM25Search](#1-bm25search)  
   - [2. Scorer](#2-scorer)  
   - [3. BeamSearchRanker](#3-beamsearchranker)  
   - [4. Pipeline](#4-pipeline)  
6. [Examples](#examples)  
   - [1. Simple Query-Retrieve-Rank](#1-simple-query-retrieve-rank)  
   - [2. Using a Pre-Indexed BM25 Cache](#2-using-a-pre-indexed-bm25-cache)  
   - [3. Customizing BM25 Parameters](#3-customizing-bm25-parameters)  
7. [Advanced Topics](#advanced-topics)  
   - [Indexing Larger Datasets](#indexing-larger-datasets)  
   - [Memory-Mapping to Reduce RAM Footprint](#memory-mapping-to-reduce-ram-footprint)  
   - [Extending to Parquet or Text Files](#extending-to-parquet-or-text-files)  
8. [Developer Guide](#developer-guide)  
9. [Citation and Acknowledgments](#citation-and-acknowledgments)  

---

## Features

- **BM25S Integration** – Leverage [BM25S](https://github.com/xhluca/bm25s) for high-speed lexical retrieval.  
- **Configurable** – YAML/TOML/ENV-based settings for your base model, teacher model, reward model, dataset name, BM25 parameters, etc.  
- **Beam-Search Expansions** – Improve queries iteratively with a language model, broadening or focusing retrieval.  
- **Scoring** – Compare results using a reward model (e.g. cross-encoder).  
- **Summaries** – Summarize final top-k results using a chain-of-thought approach.  
- **Easy Save/Load** – Persist indexes for re-use, reducing overhead on repeated runs.

---

## Installation

### 1. Install from Source (recommended during early development)

```bash
git clone https://github.com/<your-org>/BwB.git
cd BwB
# Install dependencies using Poetry, pip, or your favorite tool:
pip install -r requirements.txt
pip install .
```

Or, if you have a **pyproject.toml**, simply do:

```bash
pip install build  # if you haven't installed build
python -m build
pip install dist/*.whl
```

### 2. (Future) Install from PyPI

Once the library is published to PyPI (planned for the future):

```bash
pip install bwb
```

---

## Quickstart

Below is a minimal script that uses **BwB** to:

1. Load a config from `my_config.toml`  
2. Index a small dataset from Hugging Face’s **BeIR/scidocs** corpus  
3. Perform a beam-search expansion and rank with a reward model  
4. Print the final top results  

```python
from bwb.pipeline import search_and_rank
from bwb.config import Config

# 1) Load config
config = Config.from_toml("my_config.toml")

# 2) Provide a query
query = "What are the health consequences of long-term obesity?"

# 3) Let’s index the dataset (if needed) and rank
prediction = search_and_rank(query, config=config, use_preindexed=False)

# 4) Inspect results
print("---- Terms ----")
print(prediction.terms)
print("\n---- Top Results ----")
for idx, doc in enumerate(prediction.results):
    print(f"Rank {idx+1}:", doc)

print("\n---- Score ----")
print(prediction.score)

print("\n---- Summary ----")
print(prediction.summary)
```

**First run** will build the index and store it. Subsequent runs can load the existing index (see [Using a Pre-Indexed BM25 Cache](#2-using-a-pre-indexed-bm25-cache)).

---

## Configuration

BwB uses a **`Config`** object that can be loaded from:
1. A **TOML** file, e.g. `my_config.toml`
2. **Environment variables** with fallback defaults
3. Another approach (like custom Python code)

Below is an **example** TOML file. Save it as `my_config.toml`:

```toml
[models]
# Base model for expansions
base_model = "ollama_chat/gemma2:2b-instruct-q8_0"
temperature = 1.0

# Teacher model for additional generation or reference
teacher_model = "openrouter/qwen/qwen-2.5-72b-instruct"
teacher_temperature = 0.8

# Reward model for scoring retrieval results
reward_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

[dataset]
# The huggingface dataset name
name = "BeIR/scidocs"

[bm25s]
# BM25S index/scoring parameters
method = "lucene"
k1 = 1.2
b = 0.75
delta = 1.0

# Where to save or load the BM25 index
save_dir = "bm25_index"
use_mmap = false
```

**Usage**:

```python
from bwb.config import Config

config = Config.from_toml("my_config.toml")
print(config.dataset)
print(config.bm25s_config.method)
```

---

## Core Concepts

### 1. **BM25Search**

A class that encapsulates:
- **Indexing** from one or more data sources (Hugging Face dataset, local text, etc.).
- **Saving/Loading** of the index.
- **Querying** to obtain top-k results.

Typical usage:

```python
from bwb.search import BM25Search
from bwb.config import BM25SConfig

bm25_cfg = BM25SConfig(method="lucene", k1=1.5, b=0.75, delta=1.0)
bm25_search = BM25Search(bm25_cfg)

# Index from huggingface
bm25_search.index_hf_dataset(
    dataset_name="BeIR/scidocs",
    subset="corpus",
    split="train",
    column="text"
)

# Query
results = bm25_search.query("What is the best cat food?", k=3)
print(results)
# ['Doc with cat mention', 'Another doc with cat mention', '...']

# Save the index
bm25_search.save_index(load_corpus=True)
```

### 2. **Scorer**

Manages:
- **Scoring** documents with a given reward model
- **Caching** scores to avoid re-computation
- **Maintaining a top-k** min-heap for best results

You normally won’t instantiate `Scorer` directly unless you want advanced usage. The **BeamSearchRanker** orchestrates it for you.

### 3. **BeamSearchRanker**

Coordinates:
1. **Initial retrieval** from BM25Search  
2. **Expansion** of search terms in multiple “beam” steps  
3. **Scoring** expansions with a reward model  
4. **Final** top-k selection  

Sample usage:

```python
from bwb.ranker import BeamSearchRanker
from bwb.scorer import Scorer
from bwb.search import BM25Search
from sentence_transformers import CrossEncoder

reward_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
bm25_search = BM25Search(bm25_cfg)
bm25_search.index_hf_dataset("BeIR/scidocs", subset="corpus", split="train")

ranker = BeamSearchRanker(
    reward_model=reward_model,
    depth=3,
    expansions=3,
    k=5,
    bm25_search=bm25_search
)

prediction = ranker.rank("What are potential negative effects of chronic obesity?")
print("Top results:", prediction.results)
print("Score:", prediction.score)
```

### 4. **Pipeline**

A simpler, one-stop function that handles:
1. **Config** loading  
2. **Building** or **loading** the BM25 index  
3. **Creating** a `BeamSearchRanker`  
4. **Returning** final predictions  

**`search_and_rank(query, config, use_preindexed=False)`** is the recommended entrypoint:

```python
from bwb.pipeline import search_and_rank

prediction = search_and_rank("Does obesity increase risk of diabetes?", use_preindexed=False)
print(prediction.results)
```

---

## Examples

### 1. **Simple Query-Retrieve-Rank**

```python
from bwb.pipeline import search_and_rank

query = "How does obesity affect cardiovascular health?"
result = search_and_rank(query, use_preindexed=False)

print("Terms used:", result.terms)
print("\nTop results:\n", result.results)
print("\nFinal score:", result.score)
print("\nSummary:\n", result.summary)
```

On first run, it will index the dataset (according to the config file) and store the index. On subsequent runs, pass `use_preindexed=True` to skip re-indexing.

---

### 2. **Using a Pre-Indexed BM25 Cache**

If you already built the BM25 index and saved it:

```python
from bwb.pipeline import search_and_rank
from bwb.config import Config

# Load config
config = Config.from_toml("my_config.toml")

# We specify use_preindexed=True to load from the on-disk index
result = search_and_rank("High blood pressure and obesity correlation", config=config, use_preindexed=True)
```

No re-indexing overhead—just **load** and **retrieve**.

---

### 3. **Customizing BM25 Parameters**

Use the `[bm25s]` section in your config or environment variables. For instance:

```toml
[bm25s]
method = "bm25+"
k1 = 1.8
b = 0.6
delta = 1.2
save_dir = "custom_bm25_index"
use_mmap = true
```

This instructs your pipeline to create a **BM25+** model with `k1=1.8, b=0.6, delta=1.2`, saving/loading at `custom_bm25_index`, and memory-mapping for low RAM usage.

---

## Advanced Topics

### Indexing Larger Datasets

For extremely large Hugging Face datasets, consider:
- **Streaming** or chunk-based approach.  
- `bm25_search.index_hf_dataset(...)` might need a custom approach to handle partial indexing.  

*(Future enhancements to BwB may include a streaming-based index builder.)*

### Memory-Mapping to Reduce RAM Footprint

If `use_mmap=true` in `[bm25s]`, the library will memory-map the BM25 arrays. This helps handle very large corpora (millions of documents) without exhausting your system’s RAM.

### Extending to Parquet or Text Files

We provide an example stub in `BM25Search` for indexing from Hugging Face. You can similarly implement:

```python
def index_parquet(self, parquet_path: str, column: str = "text"):
    # Load parquet, read the column, index it
    ...
```

Or:

```python
def index_local_text(self, dir_path: str):
    # Walk directory, read text files, index
    ...
```

---

## Developer Guide

To contribute or modify, note the **modular** structure:

```
bwb/
├─ config.py        # config logic
├─ search.py        # BM25Search + indexing sources
├─ scorer.py        # Scorer logic
├─ ranker.py        # BeamSearchRanker class + expansions
├─ pipeline.py      # search_and_rank - one-stop function
└─ ...
```

**Testing**: Add your unit tests under a `tests/` directory. For example:

```
tests/
├─ test_search.py
├─ test_ranker.py
└─ ...
```

You can run tests with `pytest`.

---

## Citation and Acknowledgments

- **BM25S** code from [xhluca/bm25s](https://github.com/xhluca/bm25s).  
- **Chain-of-thought** expansions with [dspy](https://github.com/your-org/dspy) or a custom approach.  
- **Reward Model** referencing [SentenceTransformers CrossEncoder](https://www.sbert.net/docs/pretrained_cross-encoders.html).  

