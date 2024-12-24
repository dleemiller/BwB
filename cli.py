#!/usr/bin/env python
# cli.py

"""
Command-line interface for BwB (Beam-search with BM25).
"""

import argparse
import os
import sys
import logging
from functools import partial

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich import box

from bwb.config import Config
from bwb.pipeline import search_and_rank


logger = logging.getLogger("bm25s")
logger.setLevel(logging.ERROR)

# Disable TQDM in sub-modules if it interferes with Rich
os.environ["DISABLE_TQDM"] = "1"


def main():
    """Main entry point for the BwB CLI tool.

    Parses command-line arguments, prompts the user for a query if none is provided,
    loads the configuration, and executes the BM25-based search and ranking pipeline.
    Finally, displays the results in a formatted console layout.
    """
    console = Console()
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Obtain or prompt for the query
    query = args.query
    if not query:
        console.print("[bold]No query provided. Please enter one now.[/bold]")
        query = console.input("[bold]Query[/bold]> ")

    data_config = {
        "dataset_name": args.data_dataset_name,
        "subset": args.data_subset,
        "split": args.data_split,
        "column": args.data_column,
    }

    try:
        config = Config.from_toml(args.config)
    except Exception as e:
        console.print(
            f"[bold red]Error loading config from {args.config}:[/bold red] {e}"
        )
        sys.exit(1)

    with Progress(transient=True) as progress:
        task_id = progress.add_task("[cyan]Processing...", total=1)
        progress_update = partial(progress.update, task_id)

        # Execute the search/ranking. Example usage of progress messages can be done inside `search_and_rank`.
        result = search_and_rank(
            query=query,
            config=config,
            index_mode=args.index_mode,
            data_source=args.data_source,
            data_config=data_config,
            progress_update=progress_update,
        )

    _display_results(console, result)


def _build_arg_parser():
    """Builds the argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: Configured argument parser for BwB CLI.
    """
    parser = argparse.ArgumentParser(
        prog="bwb",
        description="Beam-search + BM25 retrieval. Query your text corpora with expansions!",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="my_config.toml",
        help="Path to the TOML config file (defaults to my_config.toml)",
    )
    parser.add_argument(
        "--index-mode",
        type=str,
        choices=["auto", "build", "use_preindexed"],
        default="auto",
        help="How to handle BM25 indexing: 'auto' (try load, else build), 'build' (force), 'use_preindexed' (load only)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["huggingface", "parquet", "local_text"],
        default="huggingface",
        help="Data ingestion method for building a new index (ignored if use_preindexed).",
    )
    parser.add_argument(
        "--data-dataset-name",
        type=str,
        default="BeIR/scidocs",
        help="Hugging Face dataset name or path. (Used if data_source = huggingface)",
    )
    parser.add_argument(
        "--data-subset",
        type=str,
        default=None,
        help="Subset name if applicable (e.g. 'corpus' for scidocs). (huggingface only)",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="train",
        help="Split name (huggingface only).",
    )
    parser.add_argument(
        "--data-column",
        type=str,
        default="text",
        help="Column name containing text to index (huggingface or parquet).",
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="The query to search for. If omitted, it will prompt for input.",
    )
    return parser


def _display_results(console, result):
    """Display final search results and summary in the console.

    Args:
        console (Console): Rich Console object for printing.
        result (object): The result object returned by `search_and_rank`.
            Expected to contain attributes like `terms`, `results`, `score`,
            and possibly a `summary` with `reasoning` and `answer`.
    """
    console.rule("[bold green]BwB - Results[/bold green]")
    terms_str = ", ".join(result.terms) if result.terms else "No terms"
    console.print(Panel.fit(f"[bold]Terms used:[/bold]\n{terms_str}", box=box.DOUBLE))

    if result.results:
        doc_lines = []
        for i, doc in enumerate(result.results):
            doc_lines.append(f"[bold]Rank {i+1}[/bold]: {doc}")
        docs_text = "\n\n".join(doc_lines)
    else:
        docs_text = "No documents retrieved."

    console.print(Panel.fit(docs_text, title="Top Results", box=box.DOUBLE))

    # Partial terms display
    if result.terms:
        console.print(
            f"[bold]Terms used (partial):[/bold] {', '.join(result.terms[:10])}"
        )
    else:
        console.print("[bold]No terms[/bold]")

    # Top doc snippet
    if result.results:
        top_doc = result.results[0]
        snippet = top_doc[:400].replace("\n", " ")
        console.print(
            Panel(f"[bold]Top Document:[/bold]\n{snippet}...", box=box.SQUARE)
        )
    else:
        console.print("[red]No documents found.[/red]")

    # Score
    console.print(f"[bold]Score:[/bold] {result.score:.3f}")

    # Reasoning & Answer
    reasoning = ""
    answer = ""
    if getattr(result, "summary", None):
        reasoning = getattr(result.summary, "reasoning", "")
        answer = getattr(result.summary, "answer", str(result.summary))

    console.print(
        Panel(
            f"[bold magenta]Reasoning:[/bold magenta]\n{reasoning}\n\n"
            f"[bold green]Answer:[/bold green]\n{answer}",
            title="Final Summary",
            box=box.ROUNDED,
        )
    )


if __name__ == "__main__":
    main()
