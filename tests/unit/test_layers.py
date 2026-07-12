import torch

from tokamak.model.layers import (
    GatedMLP,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_emb,
)


def test_rmsnorm_matches_reference_formula() -> None:
    torch.manual_seed(0)
    norm = RMSNorm(dim=64, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64))
    x = torch.randn(2, 5, 64)

    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-6)


def test_rmsnorm_unit_weight_gives_unit_rms() -> None:
    torch.manual_seed(1)
    norm = RMSNorm(dim=128, eps=1e-8)
    out = norm(torch.randn(10, 128) * 3.0)
    rms = out.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3)


def test_rope_at_position_zero_is_identity() -> None:
    rotary = RotaryEmbedding(head_dim=32, base=10000.0)
    cos, sin = rotary(torch.tensor([0]))
    q = torch.randn(1, 4, 1, 32)
    k = torch.randn(1, 2, 1, 32)
    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    torch.testing.assert_close(q_rot, q)
    torch.testing.assert_close(k_rot, k)


def test_rope_preserves_norm() -> None:
    torch.manual_seed(2)
    rotary = RotaryEmbedding(head_dim=64, base=1e6)
    positions = torch.arange(10)
    cos, sin = rotary(positions)
    q = torch.randn(2, 4, 10, 64)
    k = torch.randn(2, 2, 10, 64)
    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k_rot.norm(dim=-1), k.norm(dim=-1), rtol=1e-4, atol=1e-4)


def test_rope_dot_product_depends_only_on_relative_position() -> None:
    """The defining property of RoPE: <q_m, k_n> is a function of (m - n) only."""
    torch.manual_seed(3)
    head_dim = 32
    rotary = RotaryEmbedding(head_dim=head_dim, base=10000.0)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def rotated_dot(q_pos: int, k_pos: int) -> float:
        cos_q, sin_q = rotary(torch.tensor([q_pos]))
        cos_k, sin_k = rotary(torch.tensor([k_pos]))
        q_rot, _ = apply_rotary_emb(q, q, cos_q, sin_q)
        k_rot, _ = apply_rotary_emb(k, k, cos_k, sin_k)
        return float((q_rot * k_rot).sum())

    assert abs(rotated_dot(5, 3) - rotated_dot(9, 7)) < 1e-4
    assert abs(rotated_dot(10, 0) - rotated_dot(15, 5)) < 1e-4


def test_gated_mlp_shape_and_zero_gate() -> None:
    mlp = GatedMLP(hidden_size=16, intermediate_size=32)
    x = torch.randn(2, 3, 16)
    assert mlp(x).shape == (2, 3, 16)

    # silu(0) = 0, so zeroing the gate projection must zero the output.
    with torch.no_grad():
        mlp.gate_proj.weight.zero_()
    torch.testing.assert_close(mlp(x), torch.zeros(2, 3, 16))
