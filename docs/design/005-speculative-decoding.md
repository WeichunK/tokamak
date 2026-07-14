# 005 — Speculative Decoding (M5): more tokens per target forward

**Status:** implemented
**Scope:** chunked forwards, post-filter distribution extraction, rejection-sampling
verification, a draft-and-verify runner, engine integration behind `draft_model=`,
and an acceptance/speedup benchmark.

## The idea

Autoregressive decode pays one full target forward per token, and (M4
notwithstanding) that forward is memory-bound: streaming 4B parameters through HBM
to produce one row of logits. But scoring *several* tokens in one forward costs
barely more than scoring one — prefill taught us that. Speculative decoding
(Leviathan et al., 2023; Chen et al., 2023) exploits the asymmetry:

1. A cheap **draft** model proposes `k` tokens autoregressively.
2. The **target** scores all `k` (+1 bonus position) in a single chunked forward.
3. **Rejection sampling** accepts a prefix of the proposals and corrects the first
   failure — such that the emitted tokens are distributed *exactly* as if the
   target had produced them alone.

The correctness core, for draft token `d ~ q`: accept with probability
`min(1, p(d)/q(d))`; on rejection, resample from `norm(max(0, p − q))`. The two
cases sum back to `p` identically — no approximation, for any draft. Both `p` and
`q` are the **post-filter** distributions (temperature/top-k/top-p applied), so
the guarantee is "identical to target-only sampling with the same parameters";
greedy is the delta-distribution special case, handled as explicit argmax
comparison.

## What had to exist first

**Chunked forwards.** Verification feeds `k+1` tokens starting mid-cache — the
exact case M1 rejected with `NotImplementedError` because SDPA's `is_causal` is
wrong for rectangular attention. `PrefillContext` now takes `start_pos`: chunk row
`i` attends to every cached position ≤ `start_pos + i` via an explicit boolean
mask. The equivalence test (full forward vs. prefill + two mid-cache chunks) makes
retiring that old guard safe.

**Post-filter distributions.** The sampler's filtering pipeline is now exposed as
`sampling_probs(logits, params)` — verification needs whole distributions from
both models, not draws. `sample()` is unchanged (same filters, then multinomial).

## The runner and its one invariant

Each model tracks `cached_len` — how many positions hold valid K/V. The invariant
that keeps every forward a single contiguous chunk:

> a model's next input is always `all_token_ids[cached_len:]`.

Verification writes K/V for rejected positions too. Rather than erasing them, the
frontier rolls back to `committed − 1` after each iteration and the next chunk
overwrites the stale positions before anything reads them — position-addressed
caches make "forgetting" free. The same mechanism transparently handles the other
awkward cases: the draft catching up over the bonus token it never fed, and EOS
firing mid-acceptance (commit checks run per token, in order).

Budget handling: an iteration commits between 1 and `k+1` tokens, so `k` shrinks
to `min(k, remaining − 1)` near the cap and degenerates to a plain decode step at
the boundary. Caches are sized `max_total + k` because verification may write
up to `k` rejected positions past the budget.

## Scope decision: not composed with continuous batching

`LLM(draft_model=...)` switches generation to a sequential draft-and-verify loop
over per-request **contiguous** caches (no paged pool is even allocated — for a
4B target the unused default pool would waste ~6 GiB). Speculation and batching
solve the same under-utilization problem from different ends; composing them —
per-sequence draft states inside an iteration-level scheduler, ragged verify
batches, kernel support for multi-query decode — is a serious engineering project
(vLLM took several releases to stabilize it) and deliberately out of scope. The
milestone studies the algorithm; the benchmark quantifies exactly what the
non-composition costs.

## Correctness strategy

1. **The distribution theorem, empirically** (no models): 40k single-draft rounds
   against fixed synthetic `p`/`q` — close, reversed, and overconfident drafts —
   must emit tokens within total-variation 0.01 of `p`; measured acceptance must
   match the analytic `Σ min(p, q)`; `p == q` must accept everything; everything
   after a first rejection must be discarded.
2. **Trajectory identity on tiny models** (random weights, CPU): greedy speculative
   output must equal plain greedy decoding token-for-token — with the draft equal
   to the target (acceptance must be 100%) *and* with a completely unrelated
   random draft (acceptance must be < 100%, output still identical). Seeded
   sampled runs must be reproducible.
3. **Real weights**: self-drafting Qwen3-0.6B reproduces plain greedy output with
   > 90% acceptance (chunked-vs-stepwise float noise on near-ties accounts for
   the slack).

## Measured results

See `benchmarks/README.md` (benchmark_speculative.py): Qwen3-4B target with a
Qwen3-0.6B draft, greedy, single-sequence, natural-language continuations. The
numbers carry an honest caveat this design note should spell out: **on this
hardware the draft is not proportionally cheap.** Every decode step — 4B or 0.6B —
pays a similar Python/launch overhead floor, so the draft:target step-cost ratio
is far worse than the parameter ratio suggests, and that ratio is the whole
economics of speculation. Production engines lower the floor with CUDA graphs and
fused runners before speculation pays off broadly; measuring that gap here is the
point of the milestone, not a failure of it.

## Known limitations

- Single-sequence; not composed with continuous batching, the paged pool, or the
  M4 kernel (verification is a chunked SDPA forward).
- Draft and target must share a vocabulary (checked at load).
- No tree/multi-branch speculation (Medusa/EAGLE-style) — linear chains only.
- Acceptance statistics are per-request counters, not a rolling controller; `k`
  is static rather than adaptive.

## References

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*,
  ICML 2023.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative
  Sampling*, 2023. arXiv:2302.01318.
