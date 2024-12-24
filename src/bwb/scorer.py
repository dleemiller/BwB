import hashlib
import heapq
from typing import List, Tuple, Dict
from collections import defaultdict
import logging


def hash_result(result: str) -> str:
    """Generates a SHA-256 hash for a given result string.

    Args:
        result (str): The string to be hashed.

    Returns:
        str: The SHA-256 hash of the input string.
    """
    return hashlib.sha256(result.encode("utf-8")).hexdigest()


class Scorer:
    """Scores results using a reward model and maintains a top-k list of high-scoring results.

    This class caches the scores of previously scored results (by their hash),
    manages a min-heap to store only the top-k highest scores, and provides
    utility methods for calculating average scores and retrieving the top-k
    results.
    """

    def __init__(self, k: int):
        """Initializes the Scorer.

        Args:
            k (int): The maximum size of the top-k heap.
        """
        self.k = k
        self.score_cache: Dict[str, float] = defaultdict(float)
        self.heap: List[Tuple[float, str]] = []
        self.in_heap = set()  # Track which result hashes are currently in the heap
        self.logger = logging.getLogger(__name__)

    @classmethod
    def new(cls, k: int = 5):
        """Creates a new Scorer instance.

        Args:
            k (int, optional): The maximum size of the top-k heap. Defaults to 5.

        Returns:
            Scorer: A new instance of the Scorer class.
        """
        return cls(k)

    def score_results(
        self, query: str, results: List[str], reward_model
    ) -> List[float]:
        """Scores the given results using the provided reward model.

        Caches the scores of any new results, and updates the top-k heap.

        Args:
            query (str): The query string or context to be used by the reward model.
            results (List[str]): A list of result strings to be scored.
            reward_model: A model with a `predict` method that takes a list of
                (query, result) tuples and returns a list of float scores.

        Returns:
            List[float]: The scores for the provided results in the same order.
        """
        results_to_score = [
            r for r in results if hash_result(r) not in self.score_cache
        ]

        # Pre-fill scores array with cached scores if available
        scores = []
        for r in results:
            r_hash = hash_result(r)
            if r_hash in self.score_cache:
                s = self.score_cache[r_hash]
                self._push_to_heap(s, r, r_hash)
                scores.append(s)
            else:
                scores.append(None)

        # Batch score new results using the reward model
        if results_to_score:
            try:
                query_result_pairs = [(query, r) for r in results_to_score]
                new_scores = reward_model.predict(query_result_pairs)
            except Exception as e:
                self.logger.error(f"Error during batch scoring: {e}")
                new_scores = [0.0] * len(results_to_score)

            for r, s in zip(results_to_score, new_scores):
                r_hash = hash_result(r)
                self.score_cache[r_hash] = s
                self._push_to_heap(s, r, r_hash)

            # Replace `None` placeholders with newly computed scores
            score_iter = iter(new_scores)
            for i, sc in enumerate(scores):
                if sc is None:
                    scores[i] = next(score_iter)

        return scores

    def _push_to_heap(self, score: float, result: str, result_hash: str) -> None:
        """Pushes a scored result to the top-k min-heap, ensuring no duplicates.

        Args:
            score (float): The score of the result.
            result (str): The result string.
            result_hash (str): The SHA-256 hash of the result string.
        """
        # If result is already in the heap, skip
        if result_hash in self.in_heap:
            return

        # If heap is not full, add the new result
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (score, result))
            self.in_heap.add(result_hash)
        else:
            # If new score is higher than the min in the heap, pop the lowest and add the new one
            if score > self.heap[0][0]:
                popped_score, popped_result = heapq.heappushpop(
                    self.heap, (score, result)
                )
                popped_hash = hash_result(popped_result)

                # Remove the popped hash from tracking set
                if popped_hash in self.in_heap:
                    self.in_heap.remove(popped_hash)

                self.in_heap.add(result_hash)

    def calculate_average_score(self, results: List[str]) -> float:
        """Calculates the average score of the given results.

        If a result has not been cached (never scored), it is not included.

        Args:
            results (List[str]): A list of result strings.

        Returns:
            float: The average score of the cached results or 0.0 if no scores are available.
        """
        cached_scores = [
            self.score_cache[hash_result(r)]
            for r in results
            if hash_result(r) in self.score_cache
        ]
        if not cached_scores:
            return 0.0
        return sum(cached_scores) / len(cached_scores)

    def get_topk(self) -> Tuple[List[str], float]:
        """Retrieves the top-k highest-scoring results and their average score.

        Returns:
            Tuple[List[str], float]:
                A tuple containing:
                - A list of the top-k results in descending order of score.
                - The average score of the results in the top-k heap.
        """
        sorted_heap = sorted(self.heap, key=lambda x: x[0], reverse=True)
        top_results = [r for _, r in sorted_heap]
        if not sorted_heap:
            return [], 0.0

        avg_score = sum(s for s, _ in sorted_heap) / len(sorted_heap)
        return top_results, avg_score
