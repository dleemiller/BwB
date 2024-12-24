# src/bwb/search.py

import logging
import os
from typing import List, Optional

import bm25s
import Stemmer
import pandas as pd
from datasets import load_dataset

from .config import BM25SConfig


class BM25Search:
    """Encapsulates the logic for building, indexing, and querying a BM25 model."""

    def __init__(self, config: BM25SConfig):
        """Initializes a BM25Search instance.

        Args:
            config: BM25S configuration object (method, k1, b, delta, etc.).
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.stemmer = Stemmer.Stemmer("english")
        self.corpus = None
        self.retriever = None

    def index_hf_dataset(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "train",
        column: str = "text",
    ) -> None:
        """Indexes data from a Hugging Face dataset.

        Args:
            dataset_name: Name of the HF dataset (e.g., 'BeIR/scidocs').
            subset: Subset name of the dataset (e.g., 'corpus' for scidocs). If None, no subset is used.
            split: Split of the dataset to load (e.g., 'train', 'test', 'validation').
            column: Name of the text field to index.
        """
        self.logger.info(
            "Loading dataset from HF: %s, subset=%s, split=%s, column=%s",
            dataset_name,
            subset,
            split,
            column,
        )
        if subset:
            ds = load_dataset(dataset_name, subset, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        self.corpus = ds[column]
        self.logger.info("Tokenizing %d documents...", len(self.corpus))
        corpus_tokens = bm25s.tokenize(
            self.corpus, stopwords="en", stemmer=self.stemmer
        )

        self._build_index(corpus_tokens)

    def index_parquet(self, parquet_path: str, column: str = "text") -> None:
        """Indexes data from a local Parquet file.

        Args:
            parquet_path: Path to the local Parquet file containing text data.
            column: Name of the column in the Parquet file that contains text.

        Raises:
            FileNotFoundError: If the specified Parquet file does not exist.
            ValueError: If the specified column does not exist in the Parquet file.
        """
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the Parquet file.")

        self.corpus = df[column].tolist()
        self.logger.info(
            "Tokenizing %d documents from %s...", len(self.corpus), parquet_path
        )
        corpus_tokens = bm25s.tokenize(
            self.corpus, stopwords="en", stemmer=self.stemmer
        )

        self._build_index(corpus_tokens)

    def index_local_text(self, dir_path: str, extension: str = ".txt") -> None:
        """Indexes text from all files in a local directory.

        Args:
            dir_path: Path to the local directory containing text files.
            extension: File extension to filter by (defaults to '.txt').

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        texts = []
        for root, _, files in os.walk(dir_path):
            for file_name in files:
                if file_name.endswith(extension):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            texts.append(text)

        self.logger.info(
            "Found %d %s files in directory %s.", len(texts), extension, dir_path
        )
        self.corpus = texts
        self.logger.info("Tokenizing %d documents...", len(self.corpus))
        corpus_tokens = bm25s.tokenize(
            self.corpus, stopwords="en", stemmer=self.stemmer
        )

        self._build_index(corpus_tokens)

    def query(self, query_text: str, k: int = 5) -> List[str]:
        """Returns the top-k documents for a given query.

        Args:
            query_text: Query string.
            k: Number of top documents to retrieve.

        Returns:
            A list of top-k documents relevant to the query.

        Raises:
            RuntimeError: If no index has been built prior to querying.
        """
        if not self.retriever:
            raise RuntimeError(
                "BM25Search has no retriever indexed. Call one of the indexing methods or load_index first."
            )

        query_tokens = bm25s.tokenize(
            query_text, stemmer=self.stemmer, show_progress=False
        )
        results, _scores = self.retriever.retrieve(
            query_tokens, k=k, show_progress=False
        )
        return [self.corpus[i] for i in results[0]]

    def save_index(self, load_corpus: bool = True) -> None:
        """Saves the BM25 index to disk.

        Args:
            load_corpus: If True, includes the corpus in the saved index.

        Raises:
            RuntimeError: If there is no retriever available to save.
        """
        if not self.retriever:
            raise RuntimeError("No retriever available to save.")
        self.logger.info(
            "Saving BM25 index to %s (load_corpus=%s)...",
            self.config.save_dir,
            load_corpus,
        )
        os.makedirs(self.config.save_dir, exist_ok=True)
        self.retriever.save(
            self.config.save_dir, corpus=self.corpus if load_corpus else None
        )
        self.logger.info("BM25 index saved successfully.")

    def load_index(
        self, save_dir: Optional[str] = None, load_corpus: bool = True
    ) -> None:
        """Loads a BM25 index from disk.

        Args:
            save_dir: Directory containing the index files. If None, uses self.config.save_dir.
            load_corpus: If True, loads the corpus along with the index.

        Raises:
            FileNotFoundError: If the specified directory does not exist or is invalid.
        """
        save_dir = save_dir or self.config.save_dir
        self.logger.info(
            "Loading BM25 index from %s (load_corpus=%s)...", save_dir, load_corpus
        )
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Index directory not found: {save_dir}")

        self.retriever = bm25s.BM25.load(
            save_dir, load_corpus=load_corpus, mmap=self.config.use_mmap
        )
        self.corpus = self.retriever.corpus if load_corpus else None
        self.logger.info("BM25 index loaded successfully.")

    def _build_index(self, corpus_tokens: List[List[str]]) -> None:
        """Builds the BM25 index using the provided corpus tokens.

        Args:
            corpus_tokens: A list of tokenized documents to index.
        """
        self.retriever = bm25s.BM25(
            method=self.config.method,
            k1=self.config.k1,
            b=self.config.b,
            delta=self.config.delta,
        )
        self.logger.info(
            "Indexing BM25 model with %d documents (this may take some time)...",
            len(corpus_tokens),
        )
        self.retriever.index(corpus_tokens)
        self.logger.info("Indexing complete.")
