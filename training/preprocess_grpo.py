#!/usr/bin/env python
"""
Preprocess Script for GRPO Training Data

Steps:
  1) Load the BEIR MSMARCO "dev" split using the evaluator class.
  2) For each original query, use the evaluator’s BM25 index to get top-k results,
     extract representative text snippets using WordLlama, and build a prompt.
  3) Save the prompts (and original queries) to a Parquet file.
"""

import os
import logging
import pandas as pd
import tqdm
from transformers import AutoTokenizer

# Import the evaluator class (which handles BM25 index loading/creation)
from grpo.bm25_reward import BM25Reward
from wordllama import WordLlama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_result_snippets(evaluator, query, wl, k_select=[1, 2, 10]):
    """
    Retrieve top-k documents using the evaluator and extract representative text snippets.

    For each rank specified in k_select:
      - Retrieve the document text via evaluator.get_doc_text.
      - Use WordLlama to split the document text into segments.
      - Choose the best segment based on its match to the query.
      - Wrap the snippet with ellipses as needed.

    Returns:
      A markdown-formatted string with a snippet for each selected rank.
    """
    # Use evaluator.retrieve() to get top documents (returns doc_ids and scores)
    doc_ids, scores = evaluator.retrieve(query, k=max(k_select))

    markdown_sections = []
    for rank in k_select:
        if rank <= len(doc_ids):
            doc_id = doc_ids[rank - 1]
            text = evaluator.get_doc_text(doc_id)
            # Split text into segments with WordLlama
            segments = wl.split(text)
            # Select best segment based on matching to the query
            best_segment = sorted(segments, key=wl.key(query))[0]
            best_index = segments.index(best_segment)
            # Wrap snippet with ellipses if not at start or end
            if best_index == 0 and len(segments) > 1:
                snippet = f"{best_segment} ..."
            elif best_index == len(segments) - 1 and len(segments) > 1:
                snippet = f"... {best_segment}"
            elif len(segments) > 1:
                snippet = f"... {best_segment} ..."
            else:
                snippet = best_segment
            markdown_sections.append(
                f"### Text snippet from document at k={rank}\n{snippet}\n"
            )
    return "\n".join(markdown_sections)


def main():
    # Load tokenizer to retrieve Gemma-specific special tokens.
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token

    # Create a single evaluator instance for the BEIR MSMARCO dev split.
    evaluator = BM25Reward(dataset_name="msmarco", split="dev")
    logger.info(
        f"Loaded MSMARCO dev split with {len(evaluator.corpus)} documents and {len(evaluator.queries)} queries."
    )

    # Load WordLlama for text segmentation.
    wl = WordLlama.load()

    prompt_data = []
    # evaluator.queries is a dict mapping query_id to query text.
    for qid, query in tqdm.tqdm(evaluator.queries.items()):
        snippets = get_result_snippets(evaluator, query, wl, k_select=[1, 2, 10])
        # Build the instruction portion.
        instruction = (
            f"Original Search Query: {query}\n\n"
            f"Initial Results Snippets:\n{snippets}\n\n"
            "Please analyze these initial results and brainstorm an augmented query to improve retrieval."
        )
        # Construct the full prompt using Gemma2 formatting.
        prompt = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
            f"{bos_token}<start_of_turn>user\n"
            f"{instruction}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        # Optionally, you can append the eos_token if desired:
        # prompt += eos_token

        prompt_data.append({"query_id": qid, "original_query": query, "prompt": prompt})

    # Save the prompts to a Parquet file.
    df = pd.DataFrame(prompt_data)
    output_parquet = "grpo_prompts.parquet"
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved prompt data to Parquet file: {output_parquet}")


if __name__ == "__main__":
    main()
