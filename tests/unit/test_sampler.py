import pytest
import torch

from tokamak import SamplingParams
from tokamak.sampling import sample


def make_generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_greedy_returns_argmax() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]])
    tokens = sample(logits, SamplingParams(temperature=0.0))
    assert tokens.tolist() == [1, 0]


def test_top_k_one_matches_greedy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 100)
    greedy = sample(logits, SamplingParams(temperature=0.0))
    top_k_one = sample(logits, SamplingParams(temperature=2.0, top_k=1), make_generator())
    assert torch.equal(greedy, top_k_one)


def test_top_k_restricts_support() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    params = SamplingParams(temperature=1.0, top_k=2)
    draws = {int(sample(logits, params, make_generator(seed)).item()) for seed in range(200)}
    assert draws <= {3, 4}
    assert len(draws) == 2  # both top-2 tokens are reachable


def test_top_p_keeps_nucleus() -> None:
    # probs = [0.5, 0.3, 0.2] -> nucleus at top_p=0.7 is exactly {token0, token1}
    probs = torch.tensor([[0.5, 0.3, 0.2]])
    logits = probs.log()
    params = SamplingParams(temperature=1.0, top_p=0.7)
    draws = {int(sample(logits, params, make_generator(seed)).item()) for seed in range(200)}
    assert draws == {0, 1}


def test_top_p_always_keeps_best_token() -> None:
    probs = torch.tensor([[0.9, 0.06, 0.04]])
    logits = probs.log()
    params = SamplingParams(temperature=1.0, top_p=0.05)  # below best token's prob
    draws = {int(sample(logits, params, make_generator(seed)).item()) for seed in range(50)}
    assert draws == {0}


def test_seeded_sampling_is_reproducible() -> None:
    logits = torch.randn(2, 1000, generator=make_generator(7))
    params = SamplingParams(temperature=0.8, top_p=0.95)
    first = sample(logits, params, make_generator(42))
    second = sample(logits, params, make_generator(42))
    assert torch.equal(first, second)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_output_shape(batch_size: int) -> None:
    logits = torch.randn(batch_size, 32)
    tokens = sample(logits, SamplingParams(temperature=1.0), make_generator())
    assert tokens.shape == (batch_size,)
    assert tokens.dtype == torch.long
