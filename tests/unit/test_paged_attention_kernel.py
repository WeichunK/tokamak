"""Triton paged-attention kernel vs. a plain-PyTorch reference (GPU only).

The reference gathers each sequence's K/V from the pool and computes softmax
attention directly; the kernel must match it over scattered block tables, GQA
group sizes (including non-power-of-two), partial blocks, and both dtypes.
"""

import pytest
import torch

from tokamak import kernels

pytestmark = pytest.mark.gpu

if not kernels.is_available():  # pragma: no cover - requires CUDA + triton
    pytest.skip("CUDA + triton required", allow_module_level=True)

from tokamak.kernels.paged_attention import paged_attention_decode  # noqa: E402

CUDA = torch.device("cuda")


def reference_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    tables: list[list[int]],
    seq_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """Gather + explicit softmax attention, computed in float64 for headroom."""
    _, block_size, num_kv_heads, _ = k_cache.shape
    num_q_heads = q.shape[1]
    group = num_q_heads // num_kv_heads
    outs = []
    for i, (table, n) in enumerate(zip(tables, seq_lens, strict=True)):
        blocks = table[: -(-n // block_size)]
        k = k_cache[torch.tensor(blocks)].flatten(0, 1)[:n].double()  # [n, Hkv, D]
        v = v_cache[torch.tensor(blocks)].flatten(0, 1)[:n].double()
        k = k.permute(1, 0, 2).repeat_interleave(group, dim=0)  # [Hq, n, D]
        v = v.permute(1, 0, 2).repeat_interleave(group, dim=0)
        scores = torch.einsum("hd,hnd->hn", q[i].double(), k) * scale
        probs = scores.softmax(dim=-1)
        outs.append(torch.einsum("hn,hnd->hd", probs, v))
    return torch.stack(outs)


def build_pool(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=CUDA).manual_seed(seed)
    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    k = torch.randn(shape, generator=generator, device=CUDA, dtype=torch.float32)
    v = torch.randn(shape, generator=generator, device=CUDA, dtype=torch.float32)
    return k.to(dtype), v.to(dtype)


@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads", "head_dim", "block_size"),
    [
        (4, 2, 16, 4),  # tiny, group 2
        (7, 1, 16, 4),  # non-power-of-two group: exercises head padding
        (16, 8, 128, 16),  # Qwen3-0.6B shape
    ],
    ids=["tiny-gqa", "group7", "qwen3-shape"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@torch.inference_mode()
def test_kernel_matches_reference(
    num_q_heads: int, num_kv_heads: int, head_dim: int, block_size: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(1)
    num_blocks = 32
    k_cache, v_cache = build_pool(num_blocks, block_size, num_kv_heads, head_dim, dtype)

    # Scattered, non-monotonic tables; lengths hit partial blocks, exact block
    # boundaries, and a single token.
    tables = [[5, 2, 9, 30], [17, 4, 0, 11], [8, 25, 3, 1]]
    seq_lens = [3 * block_size + 1, block_size, 1]
    scale = head_dim**-0.5

    q = torch.randn(3, num_q_heads, head_dim, device=CUDA, dtype=dtype)
    max_blocks = max(len(t) for t in tables)
    block_tables = torch.zeros(3, max_blocks, dtype=torch.int32, device=CUDA)
    for i, table in enumerate(tables):
        block_tables[i, : len(table)] = torch.tensor(table, dtype=torch.int32)
    lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=CUDA)

    out = paged_attention_decode(q, k_cache, v_cache, block_tables, lens_t, scale)
    expected = reference_attention(q, k_cache, v_cache, tables, seq_lens, scale)

    tolerances = {torch.float32: (1e-4, 1e-5), torch.bfloat16: (2e-2, 1e-2)}
    rtol, atol = tolerances[dtype]
    torch.testing.assert_close(out.float(), expected.float(), rtol=rtol, atol=atol)


@torch.inference_mode()
def test_kernel_handles_long_context() -> None:
    """Many blocks per sequence: the block-table loop must stay numerically stable."""
    torch.manual_seed(2)
    block_size, head_dim = 16, 64
    num_blocks = 260
    k_cache, v_cache = build_pool(num_blocks, block_size, 2, head_dim, torch.float32)

    permutation = torch.randperm(256).tolist()
    seq_len = 256 * block_size - 7  # 4089 tokens over scattered blocks
    q = torch.randn(1, 4, head_dim, device=CUDA)
    block_tables = torch.tensor([permutation], dtype=torch.int32, device=CUDA)
    lens_t = torch.tensor([seq_len], dtype=torch.int32, device=CUDA)

    out = paged_attention_decode(q, k_cache, v_cache, block_tables, lens_t, head_dim**-0.5)
    expected = reference_attention(q, k_cache, v_cache, [permutation], [seq_len], head_dim**-0.5)
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-4, atol=1e-5)


def test_non_power_of_two_shapes_rejected() -> None:
    k_cache, v_cache = build_pool(4, 4, 1, 24, torch.float32)  # head_dim 24
    q = torch.randn(1, 1, 24, device=CUDA)
    with pytest.raises(ValueError, match="powers of 2"):
        paged_attention_decode(
            q,
            k_cache,
            v_cache,
            torch.zeros(1, 4, dtype=torch.int32, device=CUDA),
            torch.tensor([4], dtype=torch.int32, device=CUDA),
            scale=1.0,
        )
