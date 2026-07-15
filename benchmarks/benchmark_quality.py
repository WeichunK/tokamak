"""Quality cost of attention policies: teacher-forced perplexity on long text.

The M7 measurement: windowed policies are inference-time *approximations* of a
dense-attention model, and this script prices them. Each policy scores the same
long-document token segments teacher-forced; a query at position ``p`` sees
exactly what the policy allows (banded-causal mask, sink columns re-enabled),
which is the visibility a windowed decode would have produced at that position.

The expected shape (StreamingLLM, Xiao et al. 2023): a plain window degrades
sharply once positions outgrow it — softmax attention parks surplus probability
mass on the earliest tokens, and the window evicts them — while re-enabling a
handful of sink positions recovers most of the loss at negligible memory cost.
Reproducing that curve is the correctness bar for the whole policy stack.

Usage (defaults reproduce the table in benchmarks/README.md):
    python benchmarks/benchmark_quality.py
    python benchmarks/benchmark_quality.py --policies full,window:512,streaming:512+4
"""

import argparse
import math
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

from tokamak import LLM
from tokamak.model.attention_policy import AttentionPolicy
from tokamak.model.kv_cache import ContiguousKVCache
from tokamak.model.step_context import PrefillContext

# Public-domain long text (Tolstoy, "War and Peace", Project Gutenberg #2600).
# PPL here compares policies on identical tokens; the corpus identity matters
# less than it being long, natural, and reproducible.
TEXT_URL = "https://www.gutenberg.org/files/2600/2600-0.txt"
CACHE = Path(__file__).parent / ".cache"

DEFAULT_POLICIES = (
    "full",
    "window:1024",
    "window:512",
    "window:256",
    "streaming:1024+4",
    "streaming:512+4",
    "streaming:256+4",
)


def load_text(url: str) -> str:
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / url.rsplit("/", 1)[-1]
    if not cached.exists():
        print(f"downloading {url} -> {cached}")
        with urllib.request.urlopen(url) as response:
            cached.write_bytes(response.read())
    text = cached.read_text(encoding="utf-8")
    # Trim Gutenberg boilerplate: keep the middle of the book.
    return text[len(text) // 4 : 3 * len(text) // 4]


@torch.inference_mode()
def score_policy(
    llm: LLM, policy: AttentionPolicy, segments: list[list[int]], logit_chunk: int = 512
) -> float:
    """Sum NLL of every segment's tokens under the policy; return perplexity."""
    total_nll = 0.0
    total_tokens = 0
    for segment in segments:
        input_ids = torch.tensor([segment[:-1]], dtype=torch.long, device=llm.device)
        targets = torch.tensor(segment[1:], dtype=torch.long, device=llm.device)
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=llm.device)[None]
        cache = ContiguousKVCache(
            llm.model_config, max_seq_len=seq_len, device=llm.device, dtype=llm.dtype
        )
        ctx = PrefillContext(cache, policy=policy)
        hidden = llm.model(input_ids, positions, ctx)
        # Logits for 4k positions x 151k vocab don't fit comfortably; chunk.
        for lo in range(0, seq_len, logit_chunk):
            logits = llm.model.compute_logits(hidden[:, lo : lo + logit_chunk]).float()
            total_nll += F.cross_entropy(
                logits[0], targets[lo : lo + logit_chunk], reduction="sum"
            ).item()
        total_tokens += seq_len
        del cache
        if llm.device.type == "cuda":
            torch.cuda.empty_cache()
    return math.exp(total_nll / total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-segments", type=int, default=4)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--text-url", default=TEXT_URL)
    args = parser.parse_args()

    llm = LLM(args.model, max_seq_len=args.seq_len)
    ids = llm.tokenizer(load_text(args.text_url))["input_ids"]
    span = args.seq_len + 1
    available = len(ids) // span
    if available < args.num_segments:
        raise SystemExit(f"text yields {available} segments of {span} tokens, need more")
    segments = [ids[i * span : (i + 1) * span] for i in range(args.num_segments)]
    scored = args.num_segments * args.seq_len
    print(f"scoring {scored} tokens ({args.num_segments} x {args.seq_len}) per policy\n")

    print(f"{'policy':<20} {'kv budget':>10} {'ppl':>8}")
    print("-" * 40)
    baseline = None
    for spec in args.policies.split(","):
        policy = AttentionPolicy.parse(spec.strip())
        budget = "all" if policy.is_full else str(policy.sinks + (policy.window or 0))
        ppl = score_policy(llm, policy, segments)
        baseline = baseline if baseline is not None else ppl
        print(f"{spec.strip():<20} {budget:>10} {ppl:>8.2f}   ({ppl / baseline - 1:+.1%} vs full)")


if __name__ == "__main__":
    main()
