# Guess Four Tokens, Verify Once: Speculative Decoding Without Changing a Single Output

> **Status: draft.** Part 5 of a series building [tokamak](https://github.com/WeichunK/tokamak),
> a minimal LLM inference engine in PyTorch. Parts 2–4 built paged KV memory,
> continuous batching, and a Triton paged-attention kernel.

There's a magic trick at the heart of modern inference engines: a small model
guesses the next few tokens, a big model checks all the guesses in one pass, and
— here's the trick — the output is *mathematically guaranteed* to be distributed
exactly as if the big model had written every token itself. Not approximately.
Exactly, for any draft model, even an adversarially bad one.

This post implements that trick from scratch — the rejection-sampling
verification, the chunked forwards it needs, and the cache bookkeeping nobody
warns you about — and then measures it honestly on hardware where the textbook
speedup assumptions get uncomfortable.

## Why guessing is free money (in theory)

A 4B-parameter model generating one token streams ~8 GB of weights through HBM
to produce one row of logits. Generating the *next* token streams the same 8 GB
again. But scoring five tokens at once? Nearly the same cost as scoring one —
the weights stream once, the tokens ride along. Prefill has always exploited
this; speculative decoding extends it to generation:

1. A cheap **draft** model proposes `k` tokens, one at a time (cheap × k).
2. The **target** model scores all `k` plus one bonus position in a single
   chunked forward (expensive × 1).
3. A verification rule accepts some prefix of the guesses.

If the draft guesses well, you get several tokens per expensive forward. If it
guesses badly, you fall back to one — never worse than that, correctness-wise.

## The theorem that makes it trustworthy

The verification rule, for a draft token `d` sampled from the draft distribution
`q`, with target distribution `p`:

- **Accept** with probability `min(1, p(d) / q(d))`.
- **On rejection**, sample the replacement from `norm(max(0, p − q))` — the
  "residual": probability mass the target has but the draft under-proposed.

Sum the two cases and the emitted token's law is exactly `p` (Leviathan et al.,
2023, Theorem 1). The draft can only affect *how often* you take the fast path,
never *what* the output distribution is.

I don't like trusting theorems I haven't tested, so the unit suite draws 40,000
verification rounds against fixed synthetic distributions — a good draft, a
reversed one, an overconfident one — and checks the emitted tokens land within
total-variation 0.01 of the target distribution every time, and that the
measured acceptance rate matches the analytic `Σ min(p, q)`. One subtlety worth
stealing: `p` and `q` must be the **post-filter** distributions (after
temperature/top-k/top-p), or your guarantee quietly becomes "matches a
distribution nobody sampled from".

## What the engine needed to learn

**Chunked forwards.** Verifying `k+1` tokens means a multi-token forward that
starts *mid-cache* — query row `i` at absolute position `start + i`, attending
over everything before it. Funny story: this is the exact case my M1 attention
code rejected with `NotImplementedError`, because SDPA's `is_causal` flag
silently computes the wrong mask for rectangular attention. Eighteen commits
later the case finally has a legitimate owner: `PrefillContext(start_pos=...)`
with an explicit boolean mask, tested chunk-by-chunk against a full forward.

**Cache amnesia.** Verification writes K/V for *rejected* tokens into both
models' caches — garbage the next iteration must not read. The fix costs
nothing: each model tracks a `cached_len` frontier, and after each iteration the
frontier rolls back to `committed − 1`. Position-addressed caches overwrite the
stale entries on the next chunk before anything reads them. One invariant keeps
the whole loop honest — *a model's next input is always
`all_token_ids[cached_len:]`* — and it transparently covers the weird cases:
the draft catching up over a bonus token it never fed, EOS firing in the middle
of an accepted run.

**Trajectory tests.** Under greedy decoding, speculation must be *invisible*:
token-identical output to plain decoding, whether the draft equals the target
(acceptance must be 100%) or is a completely unrelated pile of random weights
(acceptance craters, output must not change). Both run on tiny random models in
the unit suite; the same-model version runs on real Qwen3-0.6B weights in
integration.

## The numbers, and the caveat that matters more

Setup: Qwen3-4B target, Qwen3-0.6B draft (same tokenizer — a hard requirement),
greedy, single sequence, natural-language continuation prompts, RTX 3080 Laptop.

| Config | Tok/s | Speedup | Acceptance |
|---|---|---|---|
| baseline (target only) | 14.6 | 1.00× | — |
| speculative k=2 | 10.8 | 0.74× | 57.0% |
| speculative k=4 | 7.9 | 0.54× | 37.2% |
| speculative k=6 | 6.7 | 0.46× | 30.9% |

Yes — **slower**, at every k. And I'm publishing the table anyway, because the
arithmetic behind it is the actual lesson.

The textbook says speculation pays when the draft is much cheaper than the
target *per step* — the analyses run on a cost ratio `c = draft/target`, and the
trick multiplies when `c ≲ 0.2`. The textbook assumes steps cost what their
FLOPs cost. On a laptop GPU running an eager-mode Python engine, they don't:
every decode step — 0.6B or 4B — pays a similar ~50 ms kernel-launch and Python
overhead floor (the very first number this series ever measured). My parameter
ratio is 7×; my *step-cost* ratio is `c ≈ 0.7`. Run the k=2 arithmetic: an
iteration costs ~2×48 ms of drafting plus ~68 ms of verification for ~1.9
accepted tokens — 86 ms per token against the baseline's 68. Predicted 0.79×;
measured 0.74×. The model of the failure is quantitatively right, which is how
you know it's the overhead floor and not a bug.

This is the most portfolio-honest lesson in the series so far: **speculative
decoding is a bet on your serving stack's overhead structure, not just on your
draft's accuracy.** Production engines buy back the floor with CUDA graphs and
fused runners first, and *then* speculation multiplies. Measuring exactly where
that line sits on my stack was the point of the milestone.

## What's next

One milestone left on the original roadmap: benchmarking the whole engine
against vLLM itself — same model, same traces, same GPU, in WSL2 — and writing
an honest decomposition of the gap: how much is kernels, how much is scheduling,
how much is the Python overhead this post just measured.

Code, tests, and reproduction commands:
[github.com/WeichunK/tokamak](https://github.com/WeichunK/tokamak).

## References

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023.
