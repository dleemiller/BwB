import json
import re
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords

# Import your scorer class; adjust the import as needed for your project.
from nano_bm25_scorer import NanoBM25BeirScorer


class JsonToParquetConverter:
    def __init__(
        self,
        root_dir="./",
        excluded_dirs=None,
        output_parquet="lm25_nanobeir.parquet",
        clean_queries=True,
        query_field="augmented_query",
        output_column="cleaned_augmented_query",
        retain_duplicates=False,
    ):
        self.root_dir = Path(root_dir)
        self.excluded_dirs = (
            set(excluded_dirs)
            if excluded_dirs is not None
            else {"thinking", "thinking_examples", ".ipynb_checkpoints", "nano_indexes"}
        )
        self.output_parquet = output_parquet
        self.clean_queries = clean_queries
        self.query_field = query_field
        self.output_column = output_column
        self.retain_duplicates = retain_duplicates

        # Ensure the NLTK stopwords are downloaded
        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))

    def _find_json_files(self):
        # Recursively find JSON files, excluding specified directories
        json_files = [
            file
            for file in self.root_dir.rglob("*.json")
            if not any(ex_dir in file.parts for ex_dir in self.excluded_dirs)
        ]
        print(
            f"Found {len(json_files)} JSON files after excluding directories {self.excluded_dirs}."
        )
        return json_files

    def _load_json_files(self, json_files):
        # Load JSON data from the list of files, ensuring only dict objects are added
        data_list = []
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            data_list.append(item)
                        else:
                            print(f"Warning: Skipping non-dict item in {file}: {item}")
                elif isinstance(data, dict):
                    data_list.append(data)
                else:
                    print(
                        f"Warning: Skipping file {file} because JSON is not a dict or list of dicts: {data}"
                    )
        return data_list

    def _clean_query(self, query, retain_duplicates_override=None):
        # If query is not a string, return as-is
        if not isinstance(query, str):
            return query

        # Convert to lowercase and remove punctuation
        query = query.lower()
        query = re.sub(r"[^\w\s]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        tokens = query.split()
        # Remove stopwords and Boolean operators
        tokens = [
            token
            for token in tokens
            if token not in self.stop_words and token not in ["and", "or"]
        ]

        # Decide whether to retain duplicates
        retain_duplicates = (
            self.retain_duplicates
            if retain_duplicates_override is None
            else retain_duplicates_override
        )
        if not retain_duplicates:
            seen = set()
            unique_tokens = []
            for token in tokens:
                if token not in seen:
                    unique_tokens.append(token)
                    seen.add(token)
            tokens = unique_tokens

        return " ".join(tokens)

    def convert(self):
        # Discover JSON files and load data
        json_files = self._find_json_files()
        data_list = self._load_json_files(json_files)
        df = pd.DataFrame(data_list)

        # Apply cleaning if the query field exists
        if self.clean_queries and self.query_field in df.columns:
            print(
                f"Cleaning field '{self.query_field}' and writing results to '{self.output_column}'."
            )
            df[self.output_column] = df[self.query_field].apply(
                lambda q: self._clean_query(q)
            )

        # If present, keep only the record with max delta per query_id/dataset group
        required_cols = {"query_id", "dataset", "delta"}
        if required_cols.issubset(df.columns):
            df = df.loc[
                df.groupby(["query_id", "dataset"])["delta"].idxmax()
            ].reset_index(drop=True)

        # Subset the columns to the ones we care about
        # cols = ["query_id", "query", "augmented_query", self.output_column, "delta", "dataset", "instruction"]
        # df = df[cols]
        df.to_parquet(self.output_parquet, engine="pyarrow", index=False)
        print(f"Saved {len(df)} records to {self.output_parquet}.")
        return df

    def evaluate_and_repair_performance(
        self, k_value=10, shift=0.5, output_csv="bm25_delta_differences.csv"
    ):
        """
        For each query, evaluate the BM25 scores for the cleaned query versus the original.
        If the cleaned query degrades performance (delta_diff < 0), first try cleaning with duplicates permitted.
        If that still degrades performance, fall back to the original uncleaned query.
        The DataFrame is updated with the best cleaned query and written back to the Parquet file.
        A new column 'final_delta' is added that indicates the delta after the cleaned query.
        Evaluation results are saved to CSV.
        """
        df = self.convert()

        results = []
        for dataset, group in df.groupby("dataset"):
            print(f"Processing dataset: {dataset}")
            scorer = NanoBM25BeirScorer(dataset_name=dataset)
            for idx, row in group.iterrows():
                query_id = row["query_id"]
                original_query = row["augmented_query"]
                # Start with the current cleaned query (without duplicates)
                cleaned_query = row[self.output_column]

                fusion = scorer.FusionRun.init_run(scorer, query_id, shift=shift)
                eval_results = fusion.evaluate_augmented_queries(
                    [cleaned_query, original_query], k_value
                )
                if len(eval_results) != 2:
                    print(
                        f"Warning: Unexpected evaluation results for query_id {query_id}."
                    )
                    continue
                delta_cleaned = eval_results[0]["delta"]
                delta_original = eval_results[1]["delta"]
                delta_diff = delta_cleaned - delta_original

                # If performance is degraded, try cleaning with duplicates permitted
                if delta_diff < 0:
                    new_cleaned = self._clean_query(
                        original_query, retain_duplicates_override=True
                    )
                    new_fusion = scorer.FusionRun.init_run(
                        scorer, query_id, shift=shift
                    )
                    new_eval_results = new_fusion.evaluate_augmented_queries(
                        [new_cleaned, original_query], k_value
                    )
                    if len(new_eval_results) != 2:
                        print(
                            f"Warning: Unexpected evaluation results (new) for query_id {query_id}."
                        )
                        continue
                    new_delta_cleaned = new_eval_results[0]["delta"]
                    new_delta_original = new_eval_results[1]["delta"]
                    new_delta_diff = new_delta_cleaned - new_delta_original
                    if new_delta_diff > delta_diff:
                        print(
                            f"Query {query_id} improved from delta_diff {delta_diff} to {new_delta_diff} by permitting duplicates."
                        )
                        cleaned_query = new_cleaned
                        delta_cleaned = new_delta_cleaned
                        delta_diff = new_delta_diff
                        df.loc[df["query_id"] == query_id, self.output_column] = (
                            new_cleaned
                        )

                    # If performance is still degraded, fall back to the original query.
                    if delta_diff < 0:
                        print(
                            f"Query {query_id} still degraded after permitting duplicates; falling back to original query."
                        )
                        cleaned_query = original_query
                        delta_cleaned = delta_original
                        delta_diff = 0
                        df.loc[df["query_id"] == query_id, self.output_column] = (
                            original_query
                        )

                df.loc[df["query_id"] == query_id, "final_delta"] = delta_cleaned
                if delta_cleaned < delta_original:
                    print(
                        f"WARNING {query_id} {dataset} {data_cleaned} < {delta_original}"
                    )
                results.append(
                    {
                        "dataset": dataset,
                        "query_id": query_id,
                        "delta_cleaned": delta_cleaned,
                        "delta_original": delta_original,
                        "delta_diff": delta_diff,
                        "final_delta": delta_cleaned,  # New column: final delta after the best cleaning
                        "final_cleaned_query": cleaned_query,
                    }
                )

        results_df = pd.DataFrame(results)
        print("Sample evaluation results with repairs and fallback:")
        print(results_df.head())

        results_df.to_csv(output_csv, index=False)
        print(f"Saved evaluation results with repairs to {output_csv}.")

        # Update the Parquet file with the best cleaned queries (and final deltas)
        df.to_parquet(self.output_parquet, engine="pyarrow", index=False)
        print(f"Updated best cleaned queries saved to {self.output_parquet}.")

        return results_df


if __name__ == "__main__":
    converter = JsonToParquetConverter(
        root_dir="./",
        output_parquet="lm25_nanobeir.parquet",
        clean_queries=True,
        query_field="augmented_query",
        output_column="cleaned_augmented_query",
        retain_duplicates=False,  # Initially, remove duplicates
    )
    # First, convert JSON files to a Parquet file with cleaned queries.
    # df = converter.convert()

    # Then, run evaluation and repair:
    # If performance degrades, try permitting duplicates; if still degraded, fall back to the original query.
    evaluation_results = converter.evaluate_and_repair_performance(
        k_value=10000, shift=0.5, output_csv="bm25_delta_differences.csv"
    )
