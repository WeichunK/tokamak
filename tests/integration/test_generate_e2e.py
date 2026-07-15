"""End-to-end generation smoke tests on real weights."""

import pytest
import torch

from tokamak import LLM, SamplingParams, kernels
from tokamak.engine.sequence import FinishReason

MODEL_ID = "Qwen/Qwen3-0.6B"

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def llm() -> LLM:
    if torch.cuda.is_available():
        return LLM(MODEL_ID)  # auto: cuda + bf16
    return LLM(MODEL_ID, device="cpu", dtype=torch.float32, max_seq_len=512)


def test_generate_batch_returns_ordered_outputs(llm: LLM) -> None:
    prompts = ["The capital of France is", "1 + 1 ="]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_new_tokens=16),
        use_tqdm=False,
    )

    assert [o.request_id for o in outputs] == [0, 1]
    assert [o.prompt for o in outputs] == prompts
    for output in outputs:
        assert output.output_token_ids
        assert output.output_text.strip()
        assert output.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)


def test_max_new_tokens_is_respected(llm: LLM) -> None:
    outputs = llm.generate(
        "Count from 1 to 1000:",
        SamplingParams(temperature=0.0, max_new_tokens=8, ignore_eos=True),
        use_tqdm=False,
    )
    assert len(outputs[0].output_token_ids) == 8
    assert outputs[0].finish_reason is FinishReason.LENGTH


def test_seeded_sampling_is_reproducible(llm: LLM) -> None:
    params = SamplingParams(temperature=0.8, top_p=0.9, max_new_tokens=16, seed=1234)
    first = llm.generate("Once upon a time", params, use_tqdm=False)
    second = llm.generate("Once upon a time", params, use_tqdm=False)
    assert first[0].output_token_ids == second[0].output_token_ids


def test_oversized_prompt_raises(llm: LLM) -> None:
    # Every " word" repetition is at least one token, so this cannot fit.
    long_prompt = "word " * (llm.max_seq_len + 1)
    with pytest.raises(ValueError, match="does not fit"):
        llm.generate(long_prompt, SamplingParams(max_new_tokens=1), use_tqdm=False)


def test_outputs_carry_timing_metrics(llm: LLM) -> None:
    outputs = llm.generate(
        "Hello", SamplingParams(temperature=0.0, max_new_tokens=4), use_tqdm=False
    )
    assert outputs[0].ttft_s is not None and outputs[0].ttft_s > 0
    assert outputs[0].latency_s is not None
    assert outputs[0].latency_s >= outputs[0].ttft_s


@pytest.mark.gpu
def test_triton_backend_matches_sdpa_greedy() -> None:
    """The kernel decode path must reproduce the reference path token for token."""
    if not kernels.is_available():
        pytest.skip("CUDA + triton required")
    prompts = ["The capital of France is", "1 + 1 ="]
    params = SamplingParams(temperature=0.0, max_new_tokens=16, ignore_eos=True)

    reference = LLM(MODEL_ID, max_seq_len=512, attention_backend="sdpa")
    expected = [o.output_token_ids for o in reference.generate(prompts, params, use_tqdm=False)]
    del reference
    torch.cuda.empty_cache()

    kernel = LLM(MODEL_ID, max_seq_len=512, attention_backend="triton")
    actual = [o.output_token_ids for o in kernel.generate(prompts, params, use_tqdm=False)]

    assert actual == expected


def test_wide_window_matches_full_greedy() -> None:
    """A window larger than any context must be a semantic no-op end to end.

    Runs on CPU/fp32: the policy mask routes SDPA through its explicit-mask
    kernel, whose bf16 CUDA numerics differ from ``is_causal``'s by enough to
    flip near-tie greedy picks — on fp32 the two paths are deterministic.
    """
    prompts = ["The capital of France is"]
    params = SamplingParams(temperature=0.0, max_new_tokens=16, ignore_eos=True)
    full = LLM(MODEL_ID, device="cpu", dtype=torch.float32, max_seq_len=128)
    expected = [o.output_token_ids for o in full.generate(prompts, params, use_tqdm=False)]
    del full

    windowed = LLM(
        MODEL_ID,
        device="cpu",
        dtype=torch.float32,
        max_seq_len=128,
        attention_policy="streaming:128+4",
    )
    actual = [o.output_token_ids for o in windowed.generate(prompts, params, use_tqdm=False)]
    assert actual == expected


def test_windowed_triton_matches_windowed_sdpa_greedy() -> None:
    """The kernel's two-phase visibility walk must reproduce the masked reference.

    The window (16 + 2 sinks) is far smaller than prompt + 96 new tokens, so
    the policy restricts most decode steps, whole blocks go dead behind the
    band (block reclamation runs for real), and the kernel walks a table whose
    dead entries are stale.
    """
    if not kernels.is_available():
        pytest.skip("CUDA + triton required")
    prompts = ["The capital of France is", "1 + 1 ="]
    params = SamplingParams(temperature=0.0, max_new_tokens=96, ignore_eos=True)
    policy = "streaming:16+2"

    reference = LLM(MODEL_ID, max_seq_len=512, attention_backend="sdpa", attention_policy=policy)
    expected = [o.output_token_ids for o in reference.generate(prompts, params, use_tqdm=False)]
    del reference
    torch.cuda.empty_cache()

    kernel = LLM(MODEL_ID, max_seq_len=512, attention_backend="triton", attention_policy=policy)
    assert kernel.block_manager is not None
    baseline_free = kernel.block_manager.num_free_blocks
    reclaimed = []
    original = kernel.block_manager.release_out_of_window

    def spying_release(seq_id: int, first_live_block: int, sink_blocks: int = 0) -> int:
        count = original(seq_id, first_live_block, sink_blocks)
        if count:
            reclaimed.append(count)
        return count

    kernel.block_manager.release_out_of_window = spying_release  # type: ignore[method-assign]
    actual = [o.output_token_ids for o in kernel.generate(prompts, params, use_tqdm=False)]

    assert actual == expected
    # ~100-token sequences under a 16+2 window must shed several 16-token
    # blocks each, and everything must land back in the pool afterwards.
    assert sum(reclaimed) >= 8
    assert kernel.block_manager.num_free_blocks == baseline_free


def test_speculative_same_model_matches_plain_greedy() -> None:
    """Self-drafting must reproduce plain greedy output with near-total acceptance."""
    prompts = ["The capital of France is"]
    params = SamplingParams(temperature=0.0, max_new_tokens=24, ignore_eos=True)

    plain = LLM(MODEL_ID, device="cpu", dtype=torch.float32, max_seq_len=128)
    expected = [o.output_token_ids for o in plain.generate(prompts, params, use_tqdm=False)]
    del plain

    spec = LLM(
        MODEL_ID,
        device="cpu",
        dtype=torch.float32,
        max_seq_len=128,
        draft_model=MODEL_ID,
        num_speculative_tokens=4,
    )
    outputs = spec.generate(prompts, params, use_tqdm=False)

    assert [o.output_token_ids for o in outputs] == expected
    assert outputs[0].spec_proposed is not None and outputs[0].spec_accepted is not None
    # Identical models: disagreements can only come from chunked-vs-stepwise
    # float noise on near-ties, which must be rare.
    assert outputs[0].spec_accepted / outputs[0].spec_proposed > 0.9


def test_preemption_preserves_greedy_output() -> None:
    """Preemption-by-recomputation must be invisible in the generated tokens.

    Two sequences that each need 2 KV blocks are run against a 3-block pool, so
    whichever grows second is repeatedly evicted and re-prefilled mid-generation;
    the outputs must match a run with a roomy pool.
    """
    prompts = ["The capital of France is", "1 + 1 ="]
    params = SamplingParams(temperature=0.0, max_new_tokens=16, ignore_eos=True)

    roomy = LLM(
        MODEL_ID,
        device="cpu",
        dtype=torch.float32,
        max_seq_len=32,
        kv_pool_tokens=4096,
        max_batch_size=2,
        block_size=16,
    )
    expected = [o.output_token_ids for o in roomy.generate(prompts, params, use_tqdm=False)]
    del roomy

    tight = LLM(
        MODEL_ID,
        device="cpu",
        dtype=torch.float32,
        max_seq_len=32,
        kv_pool_tokens=48,
        max_batch_size=2,
        block_size=16,
    )
    actual = [o.output_token_ids for o in tight.generate(prompts, params, use_tqdm=False)]

    assert actual == expected
