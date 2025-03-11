import os
import pathlib
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass

import Stemmer
import bm25s

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval


@dataclass
class RetrievalScores:
    """Stores retrieval results and evaluation metrics for a single query."""

    doc_ids: List[str]
    scores: List[float]
    ndcg: Dict[str, float]
    map: Dict[str, float]
    recall: Dict[str, float]
    precision: Dict[str, float]


class BM25Reward:
    """
    A simplified BM25-based retrieval evaluator that independently assesses an original query
    and an augmented query. It computes the NDCG score for each and returns the delta.
    """

    def __init__(
        self,
        dataset_name: str = "msmarco",
        split: Literal["train", "dev", "test"] = "dev",
        out_dir: Optional[str] = None,
        index_root_dir: str = "indexes",
        use_mmap: bool = False,
    ) -> None:
        self.dataset_name = dataset_name

        if out_dir is None:
            out_dir = os.path.join(pathlib.Path(".").parent.absolute(), "datasets")
        self.out_dir = out_dir

        # Create a dedicated folder for the dataset's index
        self.index_dir = os.path.join(index_root_dir, dataset_name)
        os.makedirs(self.index_dir, exist_ok=True)

        # Download dataset if not already present
        self._download_dataset_if_needed()

        # Load dataset from disk
        self.corpus, self.queries, self.qrels = GenericDataLoader(
            data_folder=self.data_path
        ).load(split=split)

        # Initialize English stemmer and evaluator for BEIR metrics
        self.stemmer = Stemmer.Stemmer("english")
        self.evaluator = EvaluateRetrieval()

        # Build or load BM25 index
        self.retriever: bm25s.BM25 = self._build_or_load_index(use_mmap=use_mmap)

    def _download_dataset_if_needed(self) -> None:
        """Downloads and unzips the dataset if not already present."""
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        self.data_path = util.download_and_unzip(url, self.out_dir)

    def _build_or_load_index(self, use_mmap: bool) -> bm25s.BM25:
        """Builds a new BM25 index or loads an existing one."""
        required_files = ["data.csc.index.npy", "indices.csc.index.npy"]
        index_files_exist = all(
            os.path.exists(os.path.join(self.index_dir, f)) for f in required_files
        )

        if index_files_exist:
            # Load existing index
            retriever = bm25s.BM25.load(
                self.index_dir, mmap=use_mmap, load_corpus=False
            )
            retriever.activate_numba_scorer()
            return retriever

        # Otherwise, build a new index from the corpus
        corpus_keys = list(self.corpus.keys())
        corpus_texts = [
            (
                f"{self.corpus[doc_id]['title']}: {self.corpus[doc_id]['text']}"
                if "title" in doc and doc["title"]
                else doc["text"]
            )
            for doc_id in corpus_keys
        ]
        corpus_tokens = bm25s.tokenize(
            corpus_texts, stopwords="en", stemmer=self.stemmer
        )
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.activate_numba_scorer()
        retriever.save(self.index_dir)
        return retriever

    def get_query_text(self, query_id: str) -> str:
        """Returns the raw query text for a given query ID."""
        return self.queries[query_id]

    def get_doc_text(self, doc_id: str) -> str:
        """Gets the combined title and text for a given document ID."""
        doc = self.corpus[doc_id]
        if "title" in doc and doc["title"]:
            return f"{doc['title']}: {doc['text']}"
        else:
            return doc["text"]

    def retrieve(
        self, query_text: str, k: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """Retrieves documents for the given query text using BM25."""
        if k is None:
            k = len(self.corpus)
        query_tokens = bm25s.tokenize(query_text, stopwords="en", stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(
            query_tokens, k=k, backend_selection="numba"
        )

        idx_list = results.squeeze().tolist()
        score_list = scores.squeeze().tolist()

        if not isinstance(idx_list, list):
            idx_list = [idx_list]
            score_list = [score_list]

        doc_keys = list(self.corpus.keys())
        doc_ids = [doc_keys[i] for i in idx_list]
        return doc_ids, score_list

    def score_query(
        self, query_id: str, k_values: Optional[List[int]] = None
    ) -> RetrievalScores:
        """
        Evaluates the original query using BM25 and returns retrieval scores.
        """
        if k_values is None:
            k_values = [10, len(self.corpus)]
        query_text = self.get_query_text(query_id)
        doc_ids, scores = self.retrieve(query_text, k=k_values[-1])
        this_qrels = {query_id: self.qrels[query_id]}
        this_results = {query_id: dict(zip(doc_ids, scores))}

        ndcg, map_, recall, precision = self.evaluator.evaluate(
            this_qrels, this_results, k_values
        )
        return RetrievalScores(
            doc_ids=doc_ids,
            scores=scores,
            ndcg=ndcg,
            map=map_,
            recall=recall,
            precision=precision,
        )

    def evaluate_augmented_query(
        self, query_id: str, augmented_query: str, k_value: int = 1000
    ) -> Dict[str, float]:
        """
        Independently evaluates an augmented query and returns the raw NDCG delta
        compared to the original query.

        Args:
            query_id: The ID of the original query.
            augmented_query: The new (augmented) query text.
            k_value: The cutoff value for evaluation (e.g., 10 for NDCG@10).

        Returns:
            A dictionary with the original NDCG, augmented NDCG, and their delta.
        """
        # Evaluate the original query
        original_scores = self.score_query(query_id, k_values=[k_value])
        original_ndcg = original_scores.ndcg.get(f"NDCG@{k_value}", 0.0)

        # Evaluate the augmented query independently
        doc_ids, scores = self.retrieve(augmented_query, k=k_value)
        this_qrels = {query_id: self.qrels[query_id]}
        this_results = {query_id: dict(zip(doc_ids, scores))}
        ndcg, _, _, _ = self.evaluator.evaluate(this_qrels, this_results, [k_value])
        augmented_ndcg = ndcg.get(f"NDCG@{k_value}", 0.0)

        delta = augmented_ndcg - original_ndcg

        return {
            "original_NDCG": original_ndcg,
            "augmented_NDCG": augmented_ndcg,
            "delta": delta,
        }

    def compute_reward(
        self,
        query_id: str,
        augmented_query: str,
        k_value: int = 1000,
        binary_bonus: float = 1.0,
        delta_weight: float = 2.0,
    ) -> float:
        """
        Computes a reward signal based on the improvement of an augmented query over the original query.
        The reward is the raw delta (augmented NDCG minus original NDCG) plus:
          - A bonus if delta > 0.
          - A continuous component: delta_weight times a normalized delta (delta divided by original NDCG,
            capped between 0 and 1).
        This design allows negative deltas to yield negative rewards (penalizing poorer augmentations).

        Args:
            query_id: The ID of the original query.
            augmented_query: The new augmented query text.
            k_value: The cutoff value for NDCG evaluation (e.g., 10 for NDCG@10).
            binary_bonus: Fixed bonus reward if the augmented query improves over the original.
            delta_weight: Weight for the normalized delta component.

        Returns:
            The computed reward as a float.
        """
        # Evaluate both the original and augmented queries.
        eval_results = self.evaluate_augmented_query(query_id, augmented_query, k_value)
        original_ndcg = eval_results["original_NDCG"]
        augmented_ndcg = eval_results["augmented_NDCG"]
        delta = augmented_ndcg - original_ndcg

        # Bonus is added only when there's improvement.
        bonus = binary_bonus if delta > 0 else 0.0

        # Compute normalized delta.
        if original_ndcg > 0:
            norm_delta = delta / original_ndcg
        else:
            norm_delta = 1.0 if delta > 0 else 0.0

        # Threshold normalized delta to lie between 0 and 1.
        norm_delta = max(0.0, min(norm_delta, 1.0))

        # Final reward: raw delta plus bonus plus weighted normalized delta.
        reward = delta + bonus + delta_weight * norm_delta
        return reward
