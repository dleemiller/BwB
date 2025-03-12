# Training LM25

## Step 1: SFT

Following the DeepSeek methodology, seed a model with <thinking>...

We use our dataset created from successful query augmentations, teaching a model
strategies for how to think about query expansions. This step only takes about 30mins.

There are 2 datasets of 649 examples derived from NanoBeir datasets. I have been using
both for training, but one is a "concise" thinking version.

After seeding, I found the adapters need to be merged before GRPO training. Try using the
`merge.py` script.


## Step 2: GRPO

In this step, we use BM25 NDCG@10k reward functions to help guide the model into learning
to augment queries. Still in progress, but I'm using the MSMARCO dataset with 6,980 queries
to use. I am experimenting with a thinking conciseness reward, to encourage reducing the length
of thinking.
