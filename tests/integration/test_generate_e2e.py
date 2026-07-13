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
