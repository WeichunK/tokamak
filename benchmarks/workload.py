"""The shared synthetic workload for cross-engine comparisons.

Both the tokamak and vLLM benchmark scripts import this module, so every engine
serves byte-identical requests: same seeded prompt token ids, same per-request
generation lengths. Prompts are random token ids (compute cost does not depend on
token content), lengths are chat-like — log-normal prompts, exponential-ish
generation with a cap — and EOS is ignored by both engines so the workload shape
is model-independent.
"""

import random

QWEN3_VOCAB_SIZE = 151_936
# Qwen3's embedding table is padded past the tokenizer: ids >= 151,643 are
# special tokens or unused rows. tokamak feeds any embedding row happily, but
# vLLM rejects prompt ids outside the tokenizer vocab, so drawn ids are folded
# below the first special token. The fold (rather than a smaller randrange
# bound) keeps the RNG stream — and thus every request's length — identical to
# the workload recorded in benchmarks/README.md since M3.
FIRST_SPECIAL_TOKEN_ID = 151_643


def build_workload(
    num_requests: int,
    max_new_cap: int,
    seed: int,
    vocab_size: int = QWEN3_VOCAB_SIZE,
) -> tuple[list[list[int]], list[int]]:
    """Return (prompt_token_ids, new_token_counts), deterministically from seed."""
    rng = random.Random(seed)
    prompts = []
    new_tokens = []
    # RNG call order is load-bearing: it reproduces the workload recorded in
    # benchmarks/README.md since M3 (4,915 prompt / 2,901 output tokens at the
    # default arguments), keeping every table comparable.
    for _ in range(num_requests):
        prompt_len = int(min(max(rng.lognormvariate(5.0, 0.7), 16), 512))
        new_tokens.append(min(int(rng.expovariate(1 / 120)) + 8, max_new_cap))
        prompts.append(
            [rng.randrange(vocab_size) % FIRST_SPECIAL_TOKEN_ID for _ in range(prompt_len)]
        )
    return prompts, new_tokens
