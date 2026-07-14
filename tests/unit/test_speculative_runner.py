"""Speculative runner on tiny random-weight models.

The trajectory-level invariant: under greedy decoding, speculative output must be
token-identical to plain target-only decoding — for any draft model. A good draft
only changes *speed*; a garbage draft only lowers the acceptance rate.
"""

import pytest
import torch

from tokamak.config import ModelConfig
from tokamak.engine.sequence import Sequence
from tokamak.engine.speculative import SpeculativeRunner
from tokamak.model.kv_cache import ContiguousKVCache
from tokamak.model.step_context import BatchedDecodeContext, PrefillContext
from tokamak.model.transformer import TransformerForCausalLM
from tokamak.sampling_params import SamplingParams

CPU = torch.device("cpu")
VOCAB = 128


def tiny_config(num_layers: int = 2) -> ModelConfig:
    return ModelConfig(
        architecture="Qwen3ForCausalLM",
        vocab_size=VOCAB,
        hidden_size=32,
        num_layers=num_layers,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attention_bias=False,
        use_qk_norm=True,
        eos_token_ids=(0,),
    )


def make_model(config: ModelConfig, seed: int) -> TransformerForCausalLM:
    torch.manual_seed(seed)
    return TransformerForCausalLM(config).eval()


@torch.inference_mode()
def plain_greedy(
    model: TransformerForCausalLM,
    config: ModelConfig,
    prompt: list[int],
    max_new: int,
) -> list[int]:
    """Reference target-only greedy decoding."""
    cache = ContiguousKVCache(config, max_seq_len=256, device=CPU, dtype=torch.float32)
    tokens = list(prompt)
    positions = torch.arange(len(prompt))[None]
    hidden = model(torch.tensor([prompt]), positions, PrefillContext(cache))
    out: list[int] = []
    token = int(model.compute_logits(hidden[:, -1]).argmax().item())
    while True:
        out.append(token)
        tokens.append(token)
        if len(out) >= max_new:
            return out
        pos = len(tokens) - 1
        ctx = BatchedDecodeContext([cache], [pos + 1], CPU)
        hidden = model(torch.tensor([[token]]), torch.tensor([[pos]]), ctx)
        token = int(model.compute_logits(hidden[:, -1]).argmax().item())


def run_speculative(
    target: TransformerForCausalLM,
    draft: TransformerForCausalLM,
    config: ModelConfig,
    prompt: list[int],
    max_new: int,
    k: int,
    params: SamplingParams | None = None,
) -> Sequence:
    runner = SpeculativeRunner(
        target=target,
        target_config=config,
        draft=draft,
        draft_config=config,
        num_speculative_tokens=k,
        max_seq_len=256,
        device=CPU,
        dtype=torch.float32,
    )
    seq = Sequence(
        0,
        prompt,
        params or SamplingParams(temperature=0.0, max_new_tokens=max_new, ignore_eos=True),
    )
    runner.run(seq)
    return seq


@pytest.mark.parametrize("k", [1, 3, 5])
def test_self_draft_greedy_matches_plain_and_accepts_everything(k: int) -> None:
    """Draft == target: every proposal must be accepted, output unchanged."""
    config = tiny_config()
    model = make_model(config, seed=0)
    prompt = [5, 17, 42, 9]

    expected = plain_greedy(model, config, prompt, max_new=24)
    seq = run_speculative(model, model, config, prompt, max_new=24, k=k)

    assert seq.output_token_ids == expected
    assert seq.spec_proposed > 0
    assert seq.spec_accepted == seq.spec_proposed  # identical models never disagree


@pytest.mark.parametrize("k", [1, 2, 4])
def test_foreign_draft_greedy_matches_plain(k: int) -> None:
    """Any draft model: greedy output must still be exactly the target's."""
    config = tiny_config()
    target = make_model(config, seed=0)
    draft = make_model(config, seed=99)  # different random weights entirely
    prompt = [5, 17, 42, 9]

    expected = plain_greedy(target, config, prompt, max_new=24)
    seq = run_speculative(target, draft, config, prompt, max_new=24, k=k)

    assert seq.output_token_ids == expected
    # A random draft agreeing with a random target every time would be a bug too.
    assert seq.spec_accepted < seq.spec_proposed


def test_sampled_mode_respects_budget_and_records_stats() -> None:
    config = tiny_config()
    target = make_model(config, seed=0)
    draft = make_model(config, seed=99)
    params = SamplingParams(temperature=0.9, top_p=0.95, max_new_tokens=16, seed=7, ignore_eos=True)

    seq = run_speculative(target, draft, config, [5, 17, 42], max_new=16, k=3, params=params)

    assert seq.is_finished
    assert len(seq.output_token_ids) == 16
    assert seq.spec_proposed >= seq.spec_accepted >= 0


def test_seeded_sampled_run_is_reproducible() -> None:
    config = tiny_config()
    target = make_model(config, seed=0)
    draft = make_model(config, seed=99)
    params = SamplingParams(
        temperature=0.8, top_k=20, max_new_tokens=20, seed=1234, ignore_eos=True
    )

    first = run_speculative(target, draft, config, [7, 3], max_new=20, k=3, params=params)
    second = run_speculative(target, draft, config, [7, 3], max_new=20, k=3, params=params)

    assert first.output_token_ids == second.output_token_ids


def test_invalid_k_raises() -> None:
    config = tiny_config()
    model = make_model(config, seed=0)
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        SpeculativeRunner(
            target=model,
            target_config=config,
            draft=model,
            draft_config=config,
            num_speculative_tokens=0,
            max_seq_len=256,
            device=CPU,
            dtype=torch.float32,
        )
