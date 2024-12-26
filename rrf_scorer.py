import os
import pathlib
from typing import Iterator, List, Dict, Optional, Tuple
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


class SimpleBM25BeirScorer:
    """A wrapper to download/load a BEIR dataset, build a BM25 index using bm25s,
    and retrieve + evaluate queries.
    """

    def __init__(
        self,
        dataset_name: str = "scidocs",
        out_dir: Optional[str] = None,
        index_root_dir: str = "indexes",
        use_mmap: bool = False,
    ) -> None:
        """Initializes the scorer. Downloads the dataset if needed,
        then loads or builds the BM25 index.

        Args:
            dataset_name: Name of the BEIR dataset.
            out_dir: Where to save the raw dataset files. Defaults to ../datasets.
            index_root_dir: Directory where BM25 indices are saved.
            use_mmap: If True, memory-map the BM25 index for reduced memory usage.
        """
        self.dataset_name = dataset_name

        if out_dir is None:
            out_dir = os.path.join(pathlib.Path(".").parent.absolute(), "datasets")
        self.out_dir = out_dir

        # Create a dedicated folder for this dataset's index
        self.index_dir = os.path.join(index_root_dir, dataset_name)
        os.makedirs(self.index_dir, exist_ok=True)

        # Download dataset if not present
        self._download_dataset_if_needed()

        # Load dataset from disk
        self.corpus, self.queries, self.qrels = GenericDataLoader(
            data_folder=self.data_path
        ).load(split="test")

        # Initialize optional English stemmer
        self.stemmer = Stemmer.Stemmer("english")

        # Prepare evaluator for BEIR metrics
        self.evaluator = EvaluateRetrieval()

        # Build or load BM25 index
        self.retriever: bm25s.BM25 = self._build_or_load_index(use_mmap=use_mmap)

    def _download_dataset_if_needed(self) -> None:
        """Downloads and unzips the dataset if it is not already present."""
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        self.data_path = util.download_and_unzip(url, self.out_dir)

    def _build_or_load_index(self, use_mmap: bool) -> bm25s.BM25:
        """Builds a new BM25 index if none is found, or loads from disk otherwise.

        Args:
            use_mmap: Whether to load the index via memory-mapping.

        Returns:
            A BM25 retriever instance, either newly built or loaded from disk.
        """
        required_files = ["data.csc.index.npy", "indices.csc.index.npy"]
        index_files_exist = all(
            os.path.exists(os.path.join(self.index_dir, f)) for f in required_files
        )

        if index_files_exist:
            # Load existing index
            retriever = bm25s.BM25.load(
                self.index_dir, mmap=use_mmap, load_corpus=False
            )
            return retriever

        # Otherwise, build a new index
        corpus_keys = list(self.corpus.keys())
        corpus_texts = [
            f"{self.corpus[doc_id]['title']}: {self.corpus[doc_id]['text']}"
            for doc_id in corpus_keys
        ]
        corpus_tokens = bm25s.tokenize(
            corpus_texts, stopwords="en", stemmer=self.stemmer
        )

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(self.index_dir)  # saves model in self.index_dir
        return retriever

    def get_query_ids(self) -> Iterator[str]:
        """Yields all query IDs in the dataset."""
        yield from self.queries.keys()

    def get_query_text(self, query_id: str) -> str:
        """Gets the raw query text for a given query ID."""
        return self.queries[query_id]

    def get_doc_text(self, doc_id: str) -> str:
        """Gets the combined title and text for a given document ID."""
        doc = self.corpus[doc_id]
        return f"{doc['title']}: {doc['text']}"

    def retrieve_scores(
        self, query_id: str, k: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """Retrieves and ranks documents for a single query based on query_id.

        Args:
            query_id: The query ID to retrieve documents for.
            k: Number of top documents to retrieve. If None, retrieves all.

        Returns:
            A tuple of (doc_ids, scores):
            - doc_ids: A list of document IDs, ranked by relevance descending.
            - scores: A list of BM25 scores, in descending order.
        """
        if k is None:
            k = len(self.corpus)
        query_text = self.get_query_text(query_id)
        return self._retrieve(query_text, k)

    def tokenize(self, text:str):
        return bm25s.tokenize(text, stopwords="en", stemmer=self.stemmer)
    
    def _retrieve(self, query_text: str, k: int) -> Tuple[List[str], List[float]]:
        """Internal retrieve method for raw query text."""
        query_tokens = bm25s.tokenize(query_text, stopwords="en", stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        idx_list = results.squeeze().tolist()
        score_list = scores.squeeze().tolist()

        if not isinstance(idx_list, list):
            idx_list = [idx_list]
            score_list = [score_list]

        doc_keys = list(self.corpus.keys())
        doc_ids = [doc_keys[i] for i in idx_list]
        return doc_ids, score_list

    def retrieve_augmented_query_scores(
        self, augmented_query: str, k: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """Retrieves documents for an alternate (augmented) query text,
        returning doc_ids and scores (no evaluation).

        Args:
            augmented_query: The text of the augmented query.
            k: Number of top documents to retrieve. If None, retrieves all.

        Returns:
            A tuple of (doc_ids, scores), each a list of length k.
        """
        if k is None:
            k = len(self.corpus)
        return self._retrieve(augmented_query, k)

    def score_query(
        self, query_id: str, k_values: Optional[List[int]] = None
    ) -> RetrievalScores:
        """Retrieves documents for one query and computes NDCG, MAP, Recall, and Precision.

        Args:
            query_id: The query ID to evaluate.
            k_values: List of cutoff values (e.g., [10, 100]) to evaluate at.
                      If None, defaults to [10, len(corpus)].

        Returns:
            A RetrievalScores object that includes doc_ids, scores, and the metrics.
        """
        if k_values is None:
            k_values = [10, len(self.corpus)]
        doc_ids, scores = self.retrieve_scores(query_id, k=k_values[-1])
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

    class BM25Run:
        """A container for a single query's per-document BM25 scores, plus the maximum score."""

        def __init__(self, doc_scores: Dict[str, float]) -> None:
            self.doc_scores = doc_scores
            self.max_score = max(doc_scores.values()) if doc_scores else 0.0

        @classmethod
        def from_query(cls, scorer: "SimpleBM25BeirScorer", query_id: str) -> "BM25Run":
            """Creates a BM25Run by scoring all documents for a given query."""
            doc_ids, scores = scorer.retrieve_scores(query_id)
            doc_scores = {did: sc for did, sc in zip(doc_ids, scores)}
            return cls(doc_scores)

        @classmethod
        def from_augmented_query(
            cls, scorer: "SimpleBM25BeirScorer", augmented_query: str
        ) -> "BM25Run":
            """Creates a BM25Run by scoring all docs for an augmented query text."""
            doc_ids, scores = scorer.retrieve_augmented_query_scores(augmented_query)
            doc_scores = {did: sc for did, sc in zip(doc_ids, scores)}
            return cls(doc_scores)

    class FusionRun:
        """A container to hold one or more BM25Run objects and fuse them via
        a flexible score-based RRF formula.

        You can evaluate the fused ranking against a specific query's qrels.
        """

        def __init__(
            self,
            query_id: str,
            scorer: "SimpleBM25BeirScorer",
            shift: float,
            k_values: List[int],
        ) -> None:
            """Initializes with knowledge of the original query ID, the scorer, shift, and k_values."""
            self.query_id = query_id
            self.scorer = scorer
            self.shift = shift
            self.k_values = k_values
            self.runs: List["SimpleBM25BeirScorer.BM25Run"] = []

        @classmethod
        def init_run(
            cls,
            scorer: "SimpleBM25BeirScorer",
            query_id: str,
            shift: float = 0.5,
            k_values: List[int] = [10, 100, 1000, 10000],
        ) -> "FusionRun":
            """Classmethod that initializes a FusionRun with an initial BM25Run
            for the original query.

            Args:
                scorer: The BM25 scorer instance.
                query_id: The query ID for which to create the FusionRun.
                shift: The RRF shift (defaults to 0.5).
                k_values: List of cutoff values for evaluation (defaults to [10, 1000]).

            Returns:
                A FusionRun instance.
            """
            fusion = cls(query_id, scorer, shift, k_values)
            base_run = scorer.BM25Run.from_query(scorer, query_id)
            fusion.runs.append(base_run)
            return fusion

        def update_with_run(self, run: "SimpleBM25BeirScorer.BM25Run") -> None:
            """Adds a BM25Run to the list of runs (permanent update)."""
            self.runs.append(run)

        def create_augmented_run(
            self, augmented_query: str
        ) -> "SimpleBM25BeirScorer.BM25Run":
            """
            Creates (but does not store) a BM25Run from an augmented query.

            You can then call `simulate_fusion` on this run to see if it improves
            overall performance before deciding whether to add it via `update_with_run`.
            """
            return self.scorer.BM25Run.from_augmented_query(
                self.scorer, augmented_query
            )

        def simulate_fusion(
            self,
            candidate_run: "SimpleBM25BeirScorer.BM25Run",
            exponent: float = 1.0,
        ) -> RetrievalScores:
            """
            Temporarily fuses the existing runs + a candidate run, returning
            retrieval metrics *without* permanently adding the candidate run.

            Args:
                candidate_run: The new run to test.
                exponent: The exponent to apply to (max_score - doc_score).

            Returns:
                The fused retrieval scores (NDCG, MAP, etc.).
            """
            # Combine existing runs with the candidate (in memory only)
            all_runs = self.runs + [candidate_run]

            fused_dict = self._fuse_scores(
                all_runs, shift=self.shift, exponent=exponent
            )
            return self._evaluate_fused(fused_dict)

        def evaluate_fused(self, exponent: float = 1.0) -> RetrievalScores:
            """
            Fuses *all currently stored runs* using RRF and returns retrieval metrics.

            Args:
                exponent: The exponent for heavier weighting near max_score.

            Returns:
                Fused retrieval metrics (NDCG, MAP, etc.).
            """
            fused_dict = self._fuse_scores(
                self.runs, shift=self.shift, exponent=exponent
            )
            return self._evaluate_fused(fused_dict)

        def _fuse_scores(
            self,
            runs: List["SimpleBM25BeirScorer.BM25Run"],
            shift: float,
            exponent: float,
        ) -> Dict[str, float]:
            """
            Computes the fused scores for a set of runs using a modified RRF:
                rrf_component = 1 / (shift + (max_score - doc_score)^exponent).

            Returns:
                A dict of {doc_id: fused_score}.
            """
            # Collect all doc_ids across runs
            all_doc_ids = set()
            for run in runs:
                all_doc_ids.update(run.doc_scores.keys())

            fused_scores = {}
            for doc_id in all_doc_ids:
                total_rrf = 0.0
                for run in runs:
                    doc_score = run.doc_scores.get(doc_id, 0.0)
                    # Heavily favor documents with doc_score near run.max_score
                    denominator = shift + (run.max_score - doc_score) ** exponent
                    total_rrf += 1.0 / denominator
                fused_scores[doc_id] = total_rrf
            return fused_scores

        def _evaluate_fused(self, fused_dict: Dict[str, float]) -> RetrievalScores:
            """
            Given a fused_doc_id -> fused_score dict, produce a sorted ranking and evaluate.

            Returns:
                A RetrievalScores object with doc_ids, scores, and standard metrics.
            """
            # Sort doc_ids by fused_score descending
            sorted_docs = sorted(fused_dict.items(), key=lambda x: x[1], reverse=True)
            doc_ids = [doc_id for (doc_id, _) in sorted_docs]
            scores = [score for (_, score) in sorted_docs]

            # Prepare for BEIR EvaluateRetrieval
            qid = self.query_id
            this_qrels = {qid: self.scorer.qrels[qid]}
            this_results = {qid: dict(zip(doc_ids, scores))}

            ndcg, map_, recall, precision = self.scorer.evaluator.evaluate(
                this_qrels, this_results, self.k_values
            )
            return RetrievalScores(
                doc_ids=doc_ids,
                scores=scores,
                ndcg=ndcg,
                map=map_,
                recall=recall,
                precision=precision,
            )

        def evaluate_augmented_queries(
            self, augmented_queries: List[str], k_value: int
        ) -> List[Dict[str, float]]:
            """
            Evaluate a set of augmented queries and compare their NDCG@k scores to the current fused scores.

            Args:
                augmented_queries: List of augmented query texts.
                k_value: The cutoff value for evaluation.

            Returns:
                List of dictionaries indicating score changes for each augmented query.
            """
            baseline_scores = self.evaluate_fused().ndcg.get(f"NDCG@{k_value}", 0.0)
            score_deltas = []

            for aug_query in augmented_queries:
                candidate_run = self.create_augmented_run(aug_query)
                sim_result = self.simulate_fusion(candidate_run)
                augmented_ndcg = sim_result.ndcg.get(f"NDCG@{k_value}", 0.0)
                delta = augmented_ndcg - baseline_scores

                score_deltas.append(
                    {
                        "query": aug_query,
                        "NDCG@k": augmented_ndcg,
                        "candidate_run": candidate_run,
                        "delta": delta,
                    }
                )

            return score_deltas
