import hashlib
import heapq
from typing import List, Tuple, Dict
from collections import defaultdict
import logging


def hash_result(result: str) -> str:
    """Generate a SHA-256 hash for a given result string."""
    return hashlib.sha256(result.encode("utf-8")).hexdigest()


class Scorer:
    """
    Handles scoring of results, caching scores, and maintaining a top-k heap.
    """

    def __init__(self, k: int):
        self.k = k
        self.score_cache: Dict[str, float] = defaultdict(float)
        self.heap: List[Tuple[float, str]] = []
        self.logger = logging.getLogger(__name__)

    @classmethod
    def new(cls, k: int = 5):
        return cls(k)

    def score_results(
        self, query: str, results: List[str], reward_model
    ) -> List[float]:
        """
        Scores the given results using the reward model.
        """
        # Identify results not yet cached
        results_to_score = [
            r for r in results if hash_result(r) not in self.score_cache
        ]

        scores = []
        for r in results:
            r_hash = hash_result(r)
            if r_hash in self.score_cache:
                s = self.score_cache[r_hash]
                self._push_to_heap(s, r, r_hash)
                scores.append(s)
            else:
                scores.append(None)

        # Batch score new results
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

            # fill in the `None` placeholders
            score_iter = iter(new_scores)
            for i, sc in enumerate(scores):
                if sc is None:
                    scores[i] = next(score_iter)

        return scores

    def _push_to_heap(self, score: float, result: str, result_hash: str):
        """
        Push a result with its score to a min-heap, preserving only top-k.
        """
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (score, result))
        else:
            if score > self.heap[0][0]:
                heapq.heappushpop(self.heap, (score, result))

    def calculate_average_score(self, results: List[str]) -> float:
        """
        Average score of the given results, based on cached values.
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
        """
        Retrieve top-k results from the heap and their average score.
        """
        sorted_heap = sorted(self.heap, key=lambda x: x[0], reverse=True)
        top_results = [r for _, r in sorted_heap]
        if not sorted_heap:
            return [], 0.0
        avg_score = sum(s for s, _ in sorted_heap) / len(sorted_heap)
        return top_results, avg_score
