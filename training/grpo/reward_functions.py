"""
Module containing reward functions for GRPO training.

Each reward function accepts a list of completions (strings) plus extra keyword arguments
and returns a list of floats (one per completion).
"""

import re
from typing import List, Optional
from transformers import PreTrainedTokenizerBase


def reward_fn_thinking_presence(completions: List[str], **kwargs) -> List[float]:
    """
    Reward: 1.0 if a <thinking> block exists, 0.0 otherwise.
    """
    pattern = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
    return [1.0 if pattern.search(comp) else 0.0 for comp in completions]


def reward_fn_thinking_conciseness(
    completions: List[str],
    tokenizer: PreTrainedTokenizerBase,
    token_threshold: int = 320,
    min_token_threshold: int = 64,
    **kwargs
) -> List[float]:
    """
    Computes a reward for the <thinking> block based on its length.

    - If the token count is below `min_token_threshold` (e.g., 64 tokens), the reward scales linearly:
          reward = token_count / min_token_threshold.
    - If the token count is between `min_token_threshold` and `token_threshold`, a full reward (1.0) is given.
    - If the token count exceeds `token_threshold`, the reward is scaled down proportionally:
          reward = token_threshold / token_count.

    Returns 0.0 if no <thinking> block is found.
    """
    pattern = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
    rewards = []
    for comp in completions:
        match = pattern.search(comp)
        if match:
            thinking_text = match.group(1).strip()
            token_count = len(tokenizer.tokenize(thinking_text))
            if token_count < min_token_threshold:
                reward = token_count / min_token_threshold
            elif token_count <= token_threshold:
                reward = 1.0
            else:
                reward = token_threshold / token_count
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards


def reward_fn_augmented_query(completions: List[str], **kwargs) -> List[float]:
    """
    Reward: 1.0 if a non-empty <augmented_query> block exists in the completion, 0.0 otherwise.
    """
    pattern = re.compile(r"<augmented_query>(.*?)</augmented_query>", re.DOTALL)
    rewards = []
    for comp in completions:
        match = pattern.search(comp)
        reward = 1.0 if match and match.group(1).strip() else 0.0
        rewards.append(reward)
    return rewards


def reward_fn_retrieval(
    completions: List[str],
    original_query: List[str],
    query_id: List[Optional[str]],
    evaluator=None,
    **kwargs
) -> List[float]:
    """
    Reward: Uses an external evaluator to compute a retrieval reward.
    For each completion, if evaluator is provided and query_id is not None, it computes the reward
    using the augmented query if available (otherwise falls back to the original query).
    """
    pattern = re.compile(r"<augmented_query>(.*?)</augmented_query>", re.DOTALL)
    rewards = []
    for comp, orig, qid in zip(completions, original_query, query_id):
        if evaluator is not None and qid is not None:
            match = pattern.search(comp)
            augmented_query = match.group(1).strip() if match else orig
            reward = evaluator.compute_reward(
                query_id=qid,
                augmented_query=augmented_query,
                k_value=10000,
                binary_bonus=1.0,
                delta_weight=2.0,
            )
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards
