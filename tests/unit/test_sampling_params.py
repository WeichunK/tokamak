import pytest

from tokamak import SamplingParams


def test_defaults_are_valid() -> None:
    params = SamplingParams()
    assert params.temperature == 1.0
    assert not params.is_greedy


def test_zero_temperature_is_greedy() -> None:
    assert SamplingParams(temperature=0.0).is_greedy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"max_new_tokens": 0},
    ],
)
def test_invalid_params_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_frozen() -> None:
    params = SamplingParams()
    with pytest.raises(AttributeError):
        params.temperature = 0.5  # type: ignore[misc]
