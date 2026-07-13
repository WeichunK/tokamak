"""Numerical parity against the Hugging Face reference implementation.

These tests download Qwen3-0.6B (~1.4 GB) and run it in float32 on CPU, where both
implementations are deterministic enough to compare tightly. The tokamak fixture is
parametrized over both KV backends, so paged block bookkeeping is validated against
the reference on real weights, not just on tiny random models. Run with:

    pytest -m model tests/integration
"""

import pytest
import torch

from tokamak import LLM, SamplingParams
from tokamak.memory import PagedKVCacheView
from tokamak.model.kv_cache import ContiguousKVCache, KVCacheProtocol
from tokamak.model.step_context import PrefillContext

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"

pytestmark = pytest.mark.model


@pytest.fixture(scope="module", params=["contiguous", "paged"])
def tokamak_llm(request: pytest.FixtureRequest) -> LLM:
    return LLM(
        MODEL_ID,
        device="cpu",
        dtype=torch.float32,
        max_seq_len=512,
        kv_backend=request.param,
        block_size=16,
    )


@pytest.fixture(scope="module")
def hf_model():  # type: ignore[no-untyped-def]
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()


def make_cache(llm: LLM, total_tokens: int) -> KVCacheProtocol:
    """A fresh cache of the fixture's backend, with capacity granted."""
    if llm.kv_backend == "paged":
        assert llm.paged_cache is not None and llm.block_manager is not None
        view = PagedKVCacheView(llm.paged_cache, llm.block_manager, seq_id=999)
        view.ensure_capacity(total_tokens)
        return view
    return ContiguousKVCache(
        llm.model_config, max_seq_len=total_tokens, device=llm.device, dtype=llm.dtype
    )


@torch.inference_mode()
def test_prefill_logits_match_hf(tokamak_llm: LLM, hf_model) -> None:  # type: ignore[no-untyped-def]
    token_ids = tokamak_llm.tokenizer.encode(PROMPT)
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    cache = make_cache(tokamak_llm, len(token_ids))
    positions = torch.arange(len(token_ids))[None]
    try:
        ours = tokamak_llm.model.compute_logits(
            tokamak_llm.model(input_ids, positions, PrefillContext(cache))
        )
    finally:
        cache.release()
    theirs = hf_model(input_ids).logits.float()

    max_abs_diff = (ours - theirs).abs().max().item()
    assert max_abs_diff < 1e-3, f"max |dlogit| = {max_abs_diff}"
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
