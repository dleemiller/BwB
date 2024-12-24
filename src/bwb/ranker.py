import logging
import textwrap
import dspy
from typing import Tuple, Set
from .search import BM25Search
from .scorer import Scorer
from dspy import Prediction

GENERATE_TERMS = """
You are creatively brainstorming novel terms that generate new, different results based on a user’s query.

The goal is to propose broader search terms, different from any in the "previous search terms" list.

To accomplish this, use these steps:

- Determine the objective of the query
- Using previous search terms and results, determine what terms NOT to use (avoid replicating them)
- Write new search terms

Important Requirements:

- DO NOT repeat or rehash any of the search terms in "previous search terms"
- Generate new words, phrases, synonyms, or angles that might help retrieve new/different results relevant to the query.
"""


class GenerateNewSearchTerms(dspy.Signature):
    __doc__ = GENERATE_TERMS

    query: str = dspy.InputField(
        desc="The query you are trying to obtain the best results for."
    )
    previous_search_terms: list[str] = dspy.InputField(desc="Previous search terms")
    previous_results: list[str] = dspy.InputField(desc="Previously retrieved results")
    # steps: str = dspy.OutputField(
    #    desc="Steps following the instructions to create new terms"
    # )
    new_search_terms: list[str] = dspy.OutputField(
        desc="New unique terms (tokens) to search."
    )


class BeamSearchRanker(dspy.Module):
    def __init__(
        self,
        reward_model,
        depth: int = 3,
        expansions: int = 3,
        k: int = 5,
        bm25_search: BM25Search | None = None,
    ):
        """
        :param reward_model: A model used to score the results
        :param depth: # of expansion steps
        :param expansions: # of expansions per depth step
        :param k: # top results to keep
        :param bm25_search: an instance of BM25Search (or we create a new one if None)
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.reward_model = reward_model
        self.depth = depth
        self.expansions = expansions
        self.k = k
        self.predictors = [
            dspy.ChainOfThought(GenerateNewSearchTerms) for _ in range(depth)
        ]
        self.answer_question = dspy.ChainOfThought(
            "query, context: list[str] -> answer"
        )
        self.bm25_search = bm25_search or BM25Search()

    def forward(self, query: str, progress_update=None) -> Prediction:
        """
        Perform a beam-search ranking using expansions from dspy.
        """
        self.logger.info("=== Starting BeamSearchRanker ===")
        initial_results = self.bm25_search.query(query, k=self.k)
        terms = set(query.split())
        results_set = set(initial_results)

        scorer = Scorer.new(k=self.k)
        # Score the initial results
        scorer.score_results(query, initial_results, self.reward_model)

        for depth_step in range(self.depth):
            self.logger.info(f"Depth Step {depth_step+1}/{self.depth}")
            top_score_for_this_depth = -float("inf")
            best_terms_set = set()
            best_results_set = set()

            for expansion in range(self.expansions):
                prediction = self.predictors[depth_step](
                    query=query,
                    previous_search_terms=list(terms),
                    previous_results=list(results_set),
                )
                # self.logger.info(
                #     f"Expansion {expansion+1}/{self.expansions}: {prediction.steps}"
                # )
                new_search_terms = prediction.new_search_terms
                dspy.Suggest(
                    len(new_search_terms) > 0, "Search terms must be provided."
                )

                overlap = set(" ".join(new_search_terms).split()) & terms
                dspy.Suggest(len(overlap) < 5, f"Too much overlap in terms: {overlap}")

                # Retrieve new results from BM25
                new_query = " ".join(new_search_terms)
                expanded_results = self.bm25_search.query(new_query, k=self.k)
                # Score them
                scorer.score_results(query, expanded_results, self.reward_model)
                # Check average
                avg_score = scorer.calculate_average_score(expanded_results)
                if avg_score > top_score_for_this_depth:
                    top_score_for_this_depth = avg_score
                    best_terms_set = set(new_search_terms)
                    best_results_set = set(expanded_results)

                if progress_update:
                    wrapped_text = textwrap.fill(
                        ",".join(prediction.new_search_terms), width=100
                    )
                    progress_update(
                        # description=wrapped_text,
                        description=f"Searching... {wrapped_text}",
                        advance=1.0 / (self.expansions * self.depth),
                    )

            # Update global terms/results
            terms.update(" ".join(best_terms_set).split())
            results_set.update(best_results_set)

        # Once expansions are done, gather top results
        all_terms_results = self.bm25_search.query(" ".join(terms), k=self.k)
        scorer.score_results(query, all_terms_results, self.reward_model)
        final_results, final_score = scorer.get_topk()
        # Summarize
        summary = self.answer_question(query=query, context=final_results)

        return Prediction(
            terms=list(terms), results=final_results, score=final_score, summary=summary
        )
