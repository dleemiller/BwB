# src/bwb/search.py

import bm25s
import Stemmer
from datasets import load_dataset
import logging
import os
from .config import BM25SConfig


class BM25Search:
    """
    Encapsulates the logic for building, indexing, and querying a BM25S model.
    """

    def __init__(self, config: BM25SConfig):
        """
        :param config: BM25S configuration (method, k1, b, delta, etc)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.stemmer = Stemmer.Stemmer("english")  # Could also be made config-based
        self.corpus = None
        self.retriever = None

    def index_hf_dataset(
        self,
        dataset_name: str,
        subset: str | None = None,
        split: str = "train",
        column: str = "text",
    ):
        """
        Index data from a Hugging Face dataset.
        :param dataset_name: e.g. "BeIR/scidocs"
        :param subset: e.g. "corpus" for scidocs
        :param split: e.g. "train", "test", "validation"
        :param column: the text field to index
        """
        self.logger.info(
            f"Loading dataset from HF: {dataset_name}, subset={subset}, split={split}, column={column}"
        )

        if subset:
            ds = load_dataset(dataset_name, subset, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        # Convert to plain Python list of strings
        # NOTE: if the dataset is huge, you might want streaming or batch indexing
        self.corpus = ds[column]

        # Tokenize
        self.logger.info(f"Tokenizing {len(self.corpus)} documents...")
        corpus_tokens = bm25s.tokenize(
            self.corpus, stopwords="en", stemmer=self.stemmer
        )

        # Create BM25 model with user config
        self.retriever = bm25s.BM25(
            method=self.config.method,
            k1=self.config.k1,
            b=self.config.b,
            delta=self.config.delta,
        )
        self.logger.info("Indexing BM25 model (this may take some time)...")
        self.retriever.index(corpus_tokens)
        self.logger.info("Indexing complete.")

    # Future extension:
    # def index_parquet(self, parquet_path: str, column: str = "text"): ...
    # def index_local_text(self, dir_path: str): ...

    def query(self, query_text: str, k: int = 5) -> list[str]:
        """
        Return top-k documents for the query.
        """
        if not self.retriever:
            raise RuntimeError(
                "BM25Search has no retriever indexed. Call index_hf_dataset or load_index first."
            )

        query_tokens = bm25s.tokenize(
            query_text, stemmer=self.stemmer, show_progress=False
        )
        # retrieve returns (doc_ids, scores)
        results, _scores = self.retriever.retrieve(
            query_tokens, k=k, show_progress=False
        )
        # convert doc_ids to actual text from self.corpus
        return [self.corpus[i] for i in results[0]]

    def save_index(self, load_corpus: bool = True):
        """
        Save the BM25 index to disk. The directory is taken from self.config.save_dir.
        :param load_corpus: if True, we also save the corpus (so we can do retrieval later).
        """
        if not self.retriever:
            raise RuntimeError("No retriever available to save.")
        self.logger.info(
            f"Saving BM25 index to {self.config.save_dir} (load_corpus={load_corpus})..."
        )
        os.makedirs(self.config.save_dir, exist_ok=True)
        self.retriever.save(
            self.config.save_dir, corpus=self.corpus if load_corpus else None
        )
        self.logger.info("BM25 index saved successfully.")

    def load_index(self, save_dir: str | None = None, load_corpus: bool = True):
        """
        Load a BM25 index from disk into self.retriever. By default, uses self.config.save_dir.
        :param save_dir: directory containing the index files
        :param load_corpus: if True, also load the corpus from disk
        """
        save_dir = save_dir or self.config.save_dir
        self.logger.info(
            f"Loading BM25 index from {save_dir} (load_corpus={load_corpus})..."
        )
        self.retriever = bm25s.BM25.load(
            save_dir, load_corpus=load_corpus, mmap=self.config.use_mmap
        )
        self.corpus = self.retriever.corpus if load_corpus else None
        self.logger.info("BM25 index loaded successfully.")
