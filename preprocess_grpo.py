#!/usr/bin/env python
"""
Preprocess Script for GRPO Training Data

Steps:
  1) Load the BEIR MSMARCO "dev" split and extract the corpus.
  2) Build a BM25s index from the corpus texts and save the index and vocabulary.
  3) For each original query, use the BM25s retriever to get top-k results,
     extract representative text snippets using WordLlama, and build a prompt.
  4) Compute NDGC@1000 for the original query and store retrieval details required
     for later NDGC computation after query augmentation.
  5) Save the prompts (and original queries and NDGC info) to a Parquet file.
"""

import os
import logging
import pandas as pd
import tqdm
import numpy as np

from sklearn.metrics import ndcg_score

# BEIR imports
from beir import util
from beir.datasets.data_loader import GenericDataLoader

# BM25s and dependencies
import bm25s
from bm25s.tokenization import Tokenizer
import Stemmer
from wordllama import WordLlama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_and_save_bm25s_index(corpus_texts, stemmer, index_dir):
    """
    Create a BM25s index from the corpus texts and save the index and vocabulary.
    """
    # Create tokenizer with a splitter and a stemmer.
    bm25s_tokenizer = Tokenizer(splitter=lambda x: x.split(), stemmer=stemmer)
    # Tokenize the corpus texts into a tuple of tokens (as required by bm25s)
    corpus_tokens = bm25s_tokenizer.tokenize(corpus_texts, return_as="tuple")
    # Build BM25 index using the raw texts.
    retriever = bm25s.BM25(corpus=corpus_texts)
    retriever.index(corpus_tokens)
    # Save the BM25 index and tokenizer vocabulary.
    retriever.save(index_dir)
    bm25s_tokenizer.save_vocab(save_dir=index_dir)
    return retriever, bm25s_tokenizer


def load_bm25s_index(stemmer, index_dir):
    """
    Load the BM25s index and tokenizer vocabulary.
    """
    retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    bm25s_tokenizer = Tokenizer(splitter=lambda x: x.split(), stemmer=stemmer)
    bm25s_tokenizer.load_vocab(index_dir)
    return retriever, bm25s_tokenizer


def get_result_snippets(query, bm25s_retriever, bm25s_tokenizer, wl, stemmer, corpus_ids, corpus_map, k_select=[1, 2, 10]):
    """
    Retrieve top-k documents using BM25s and extract representative text snippets.

    For each rank in k_select:
      - Tokenize the query using bm25s_tokenizer.
      - Retrieve the document IDs.
      - Look up the document text from corpus_map.
      - Split the text into segments using WordLlama.
      - Choose the best segment (based on matching the query) and wrap with ellipses as needed.

    Returns:
      A markdown-formatted string with a snippet for each selected rank.
    """
    # Tokenize the query without updating the vocabulary.
    query_tokens = bm25s_tokenizer.tokenize([query], update_vocab=False, stemmer=stemmer)
    results, scores = bm25s_retriever.retrieve(query_tokens, corpus=corpus_ids, k=max(k_select), n_threads=4)
    docs = results[0]  # List of document IDs.
    markdown_sections = []
    for rank in k_select:
        if rank <= len(docs):
            doc_id = docs[rank - 1]
            text = corpus_map.get(doc_id, "")
            # Split the text into segments.
            split_text = wl.split(text)
            # Select the best segment based on the query.
            best_segment = sorted(split_text, key=wl.key(query))[0]
            best_index = split_text.index(best_segment)
            # Wrap with ellipses if needed.
            if best_index == 0 and len(split_text) > 1:
                snippet = f"{best_segment} ..."
            elif best_index == len(split_text) - 1 and len(split_text) > 1:
                snippet = f"... {best_segment}"
            elif len(split_text) > 1:
                snippet = f"... {best_segment} ..."
            else:
                snippet = best_segment
            markdown_sections.append(f"### Text snippet from document at k={rank}\n{snippet}\n")
    return "\n".join(markdown_sections)


def main():
    # ---------------------------
    # Load BEIR MSMARCO Dataset
    # ---------------------------
    data_dir = "./datasets/msmarco"
    if not os.path.exists(data_dir):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
        logger.info(f"Downloading MSMARCO dataset from {url}")
        util.download_and_unzip(url, dest_dir=data_dir)
    corpus, queries, qrels = GenericDataLoader(data_dir).load(split="dev")
    logger.info(f"Loaded MSMARCO dataset with {len(corpus)} documents and {len(queries)} queries.")

    # ---------------------------
    # Prepare Corpus Texts for BM25s Indexing
    # ---------------------------
    # Build two lists: one for document IDs and one for document texts.
    corpus_ids = []
    corpus_texts = []
    for doc_id, doc in corpus.items():
        text = doc.get("text", "")
        if doc.get("title"):
            text = doc["title"] + " " + text
        corpus_ids.append(doc_id)
        corpus_texts.append(text)
    logger.info(f"Prepared {len(corpus_texts)} documents for BM25s indexing.")

    # Create a mapping from doc_id to text for snippet extraction.
    corpus_map = dict(zip(corpus_ids, corpus_texts))

    # ---------------------------
    # Create and/or Load BM25s Index
    # ---------------------------
    index_dir = "bm25s_index"
    stemmer = Stemmer.Stemmer("english")
    if os.path.exists(index_dir):
        bm25s_retriever, bm25s_tokenizer = load_bm25s_index(stemmer, index_dir)
        logger.info("Loaded BM25s index")
    else:
        os.makedirs(index_dir, exist_ok=True)
        bm25s_retriever, bm25s_tokenizer = create_and_save_bm25s_index(corpus_texts, stemmer, index_dir)
        logger.info(f"BM25s index saved to directory: {index_dir}")

    # Activate numba scorer for faster retrieval.
    bm25s_retriever.activate_numba_scorer()

    # Load WordLlama for snippet extraction.
    wl = WordLlama.load()

    # ---------------------------
    # Generate Prompts with Retrieval Snippets and Compute NDGC@1000
    # ---------------------------
    prompt_data = []
    for qid, query in tqdm.tqdm(queries.items()):
        # Retrieve snippets to build the prompt.
        snippets = get_result_snippets(query, bm25s_retriever, bm25s_tokenizer, wl, stemmer, corpus_ids, corpus_map, k_select=[1, 2, 10])
        prompt = (
            f"Original Search Query: {query}\n\n"
            f"Initial Results Snippets:\n{snippets}\n\n"
            "Please analyze these initial results and brainstorm an augmented query to improve retrieval."
        )

        # Retrieve top 1000 documents for NDGC computation.
        query_tokens_full = bm25s_tokenizer.tokenize([query], update_vocab=False, stemmer=stemmer)
        results_full, scores_full = bm25s_retriever.retrieve(query_tokens_full, corpus=corpus_ids, k=1000, n_threads=4)
        docs_full = results_full[0]  # List of document IDs.
        scores_full = scores_full[0]  # Corresponding BM25 scores.

        # Build the ground truth relevance scores using qrels.
        y_true = [qrels.get(qid, {}).get(doc_id, 0) for doc_id in docs_full]
        k_val = min(1000, len(y_true))
        if k_val > 0:
            original_ndcg = ndcg_score(np.array([y_true]), np.array([scores_full]), k=k_val)
        else:
            original_ndcg = 0.0

        # Save prompt and retrieval details.
        prompt_data.append({
            "query_id": qid,
            "original_query": query,
            "prompt": prompt,
            "original_ndcg": original_ndcg,
            "original_retrieved_doc_ids": docs_full,
            "original_retrieved_scores": scores_full.tolist(),
            "original_relevance_scores": y_true,
        })

    # ---------------------------
    # Save Prompts and NDGC Information to Parquet File
    # ---------------------------
    df = pd.DataFrame(prompt_data)
    output_parquet = "grpo_prompts.parquet"
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved prompt data with NDGC information to Parquet file: {output_parquet}")


if __name__ == "__main__":
    main()

