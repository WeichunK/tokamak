"""Numerical parity against the Hugging Face reference implementation.

These tests download Qwen3-0.6B (~1.4 GB) and run it in float32 on CPU, where both
implementations are deterministic enough to compare tightly. Run with:

    pytest -m model tests/integration
"""

import pytest
import torch

from tokamak import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def tokamak_llm() -> LLM:
    return LLM(MODEL_ID, device="cpu", dtype=torch.float32, max_seq_len=512)


@pytest.fixture(scope="module")
def hf_model():  # type: ignore[no-untyped-def]
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()


@torch.inference_mode()
def test_prefill_logits_match_hf(tokamak_llm: LLM, hf_model) -> None:  # type: ignore[no-untyped-def]
    token_ids = tokamak_llm.tokenizer.encode(PROMPT)
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    from tokamak.model.kv_cache import KVCache

    cache = KVCache(
        tokamak_llm.model_config,
        max_seq_len=len(token_ids),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    ours = tokamak_llm.model.compute_logits(tokamak_llm.model(input_ids, cache, start_pos=0))
    theirs = hf_model(input_ids).logits.float()

    max_abs_diff = (ours - theirs).abs().max().item()
    assert max_abs_diff < 1e-3, f"max |Δlogit| = {max_abs_diff}"
    # The ranking must agree exactly at every position.
    assert torch.equal(ours.argmax(dim=-1), theirs.argmax(dim=-1))


@torch.inference_mode()
def test_greedy_generation_matches_hf(tokamak_llm: LLM, hf_model) -> None:  # type: ignore[no-untyped-def]
    max_new_tokens = 32

    outputs = tokamak_llm.generate(
        PROMPT,
        SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens, ignore_eos=True),
        use_tqdm=False,
    )
    ours = outputs[0].output_token_ids

    input_ids = torch.tensor([tokamak_llm.tokenizer.encode(PROMPT)], dtype=torch.long)
    theirs = hf_model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=None,  # mirror ignore_eos
    )[0, input_ids.shape[1] :].tolist()

    assert ours == theirs
