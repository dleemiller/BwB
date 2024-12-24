# src/bwb/config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv
import tomllib  # for Python 3.11+; or 'import toml' for older versions

load_dotenv()  # Loads .env if present


@dataclass
class BM25SConfig:
    """
    BM25S indexing/scoring configuration.
    method: one of ["lucene", "robertson", "bm25+", "bm25l", "atire"]
    k1, b, delta: typical BM25 parameters
    save_dir: directory or path prefix for saving the index
    use_mmap: whether to load index from memory-mapped files
    """

    method: str = "lucene"
    k1: float = 1.2
    b: float = 0.75
    delta: float = 1.0
    save_dir: str = "bm25_index"
    use_mmap: bool = False


@dataclass
class Config:
    # model configuration
    base_model: str
    temperature: float
    teacher_model: str
    teacher_temperature: float
    reward_model: str
    dataset: str  # e.g. "HuggingFaceH4/MATH-500"
    api_key: str | None

    # Our new BM25S config
    bm25s_config: BM25SConfig

    @classmethod
    def from_toml(cls, toml_path: str) -> "Config":
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        # Example structure in your TOML might be:
        # [models]
        # base_model = "..."
        # temperature = 1.0
        # teacher_model = "..."
        # teacher_temperature = 0.8
        # reward_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        #
        # [dataset]
        # name = "BeIR/scidocs"
        #
        # [bm25s]
        # method = "lucene"
        # k1 = 1.2
        # b = 0.75
        # delta = 1.0
        # save_dir = "my_bm25_index"
        # use_mmap = false

        models = data["models"]
        dset = data["dataset"]
        bm25s_data = data["bm25s"]

        return cls(
            base_model=models["base_model"],
            temperature=models["temperature"],
            teacher_model=models["teacher_model"],
            teacher_temperature=models["teacher_temperature"],
            reward_model=models["reward_model"],
            dataset=dset["name"],
            api_key=os.environ.get("OPENROUTER_APIKEY", None),
            bm25s_config=BM25SConfig(
                method=bm25s_data.get("method", "lucene"),
                k1=float(bm25s_data.get("k1", 1.2)),
                b=float(bm25s_data.get("b", 0.75)),
                delta=float(bm25s_data.get("delta", 1.0)),
                save_dir=bm25s_data.get("save_dir", "bm25_index"),
                use_mmap=bm25s_data.get("use_mmap", False),
            ),
        )


def load_default_config() -> Config:
    """
    Provide defaults from environment variables or fallback values.
    """
    return Config(
        base_model=os.environ.get("BASE_MODEL", "ollama_chat/gemma2:2b-instruct-q8_0"),
        temperature=float(os.environ.get("TEMPERATURE", "1.0")),
        teacher_model=os.environ.get(
            "TEACHER_MODEL", "openrouter/qwen/qwen-2.5-72b-instruct"
        ),
        teacher_temperature=float(os.environ.get("TEACHER_TEMPERATURE", "0.8")),
        reward_model=os.environ.get(
            "REWARD_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        dataset=os.environ.get("DATASET", "BeIR/scidocs"),
        api_key=os.environ.get("OPENROUTER_APIKEY", None),
        bm25s_config=BM25SConfig(
            method=os.environ.get("BM25_METHOD", "lucene"),
            k1=float(os.environ.get("BM25_K1", "1.2")),
            b=float(os.environ.get("BM25_B", "0.75")),
            delta=float(os.environ.get("BM25_DELTA", "1.0")),
            save_dir=os.environ.get("BM25_SAVE_DIR", "bm25_index"),
            use_mmap=(os.environ.get("BM25_USE_MMAP", "False").lower() == "true"),
        ),
    )
