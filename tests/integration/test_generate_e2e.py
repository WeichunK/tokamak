"""End-to-end generation smoke tests on real weights."""

import pytest
import torch

from tokamak import LLM, SamplingParams
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
