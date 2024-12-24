# src/bwb/pipeline.py

import logging
from typing import Optional, Dict, Any

import dspy
from dspy import LM
from sentence_transformers import CrossEncoder

from .config import Config, load_default_config
from .search import BM25Search
from .ranker import BeamSearchRanker


def build_reward_model(model_name: str):
    return CrossEncoder(model_name)


def _configure_dspy_lm(config: Config):
    """
    Create and configure a dspy LM if needed.
    This ensures chain-of-thought can run without "No LM is loaded" error.
    """
    lm = LM(
        model=config.base_model,
        # Here you might make these flexible (e.g. from config):
        api_base="http://localhost:11434",
        api_key="",  # or from config if needed
        temperature=config.temperature,
        cache=False,
    )
    dspy.configure(lm=lm)


def search_and_rank(
    query: str,
    config: Optional[Config] = None,
    index_mode: str = "auto",
    data_source: str = "huggingface",
    data_config: Optional[Dict[str, Any]] = None,
):
    logger = logging.getLogger(__name__)
    if not config:
        config = load_default_config()

    # Step 1: Configure dspy with an LM
    # ----------------------------------
    _configure_dspy_lm(config)

    # Step 2: Build or load your reward model
    reward_model = build_reward_model(config.reward_model)

    # Step 3: Build or load the BM25 index
    bm25_search = BM25Search(config.bm25s_config)

    if index_mode == "use_preindexed":
        bm25_search.load_index(load_corpus=True)
    elif index_mode == "build":
        _do_indexing(bm25_search, data_source, data_config)
        bm25_search.save_index(load_corpus=True)
    elif index_mode == "auto":
        try:
            bm25_search.load_index(load_corpus=True)
        except Exception as e:
            logger.warning(f"Could not load index: {e}. Building from scratch...")
            _do_indexing(bm25_search, data_source, data_config)
            bm25_search.save_index(load_corpus=True)
    else:
        raise ValueError(f"Unknown index_mode: {index_mode}")

    # Step 4: Create ranker and run
    ranker = BeamSearchRanker(
        reward_model=reward_model,
        depth=3,  # or from config
        expansions=3,  # or from config
        k=5,  # or from config
        bm25_search=bm25_search,
    )

    result = ranker.rank(query)
    return result


def _do_indexing(
    bm25_search: BM25Search, data_source: str, data_config: Dict[str, Any]
):
    if not data_config:
        data_config = {}

    if data_source == "huggingface":
        dataset_name = data_config.get("dataset_name", "BeIR/scidocs")
        subset = data_config.get("subset", None)
        split = data_config.get("split", None)  # or empty
        column = data_config.get("column", "text")
        bm25_search.index_hf_dataset(
            dataset_name, subset=subset, split=split, column=column
        )
    else:
        raise NotImplementedError("Only huggingface indexing is shown in this example.")
