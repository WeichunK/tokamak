"""Triton paged-attention decode kernel and its step context.

This is the M4 replacement for the reference decode path. The reference
(:class:`~tokamak.model.step_context.BatchedDecodeContext`) copies every
sequence's K/V out of the paged pool (gather) into a padded batch tensor before
calling SDPA — roughly double the KV bytes moved per step, plus a Python loop over
rows. This kernel reads the pool *in place*, walking each sequence's block table
inside the attention loop, exactly the trick of vLLM's PagedAttention kernel:

- grid = (sequence, kv_head); each program handles one KV head's query group
  (GQA), so all Q heads sharing a KV head reuse the same K/V loads;
- the sequence loop streams one physical block at a time via the block table,
  with a flash-attention-style online softmax (running max / normalizer), so
  nothing is materialized beyond one ``[block_size, head_dim]`` tile;
- scores and accumulation run in float32 regardless of cache dtype.

Deliberate scope limits for a readable reference-quality kernel: decode only
(one query token per sequence — prefill stays on SDPA, which is already a fused
kernel for the square causal case), no split-K across the sequence (long
contexts serialize within a program), and element-wise score math instead of
``tl.dot`` (query groups are far below tensor-core tile sizes; decode is
bandwidth-bound, so this costs little).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from tokamak.memory.paged_cache import PagedKVCache


@triton.jit
def _paged_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_os,
    stride_oh,
    stride_od,
    stride_bts,
    stride_btb,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + seq)

    g = tl.arange(0, GROUP_PAD)
    d = tl.arange(0, HEAD_DIM)
    s = tl.arange(0, BLOCK_SIZE)
    head_mask = g < GROUP
    q_heads = kv_head * GROUP + g

    q = tl.load(
        q_ptr + seq * stride_qs + q_heads[:, None] * stride_qh + d[None, :] * stride_qd,
        mask=head_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    m = tl.full([GROUP_PAD], float("-inf"), tl.float32)
    acc_norm = tl.zeros([GROUP_PAD], tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], tl.float32)

    for b in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        block_id = tl.load(block_tables_ptr + seq * stride_bts + b * stride_btb).to(tl.int64)
        k_tile = tl.load(
            k_ptr
            + block_id * stride_kb
            + s[:, None] * stride_ks
            + kv_head * stride_kh
            + d[None, :] * stride_kd
        ).to(tl.float32)
        scores = tl.sum(q[:, None, :] * k_tile[None, :, :], axis=2) * scale
        token_idx = b * BLOCK_SIZE + s
        scores = tl.where(token_idx[None, :] < seq_len, scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        rescale = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])
        acc_norm = acc_norm * rescale + tl.sum(p, axis=1)
        v_tile = tl.load(
            v_ptr
            + block_id * stride_vb
            + s[:, None] * stride_vs
            + kv_head * stride_vh
            + d[None, :] * stride_vd
        ).to(tl.float32)
        acc = acc * rescale[:, None] + tl.sum(p[:, :, None] * v_tile[None, :, :], axis=1)
        m = m_new

    out = acc / acc_norm[:, None]
    tl.store(
        out_ptr + seq * stride_os + q_heads[:, None] * stride_oh + d[None, :] * stride_od,
        out.to(out_ptr.dtype.element_ty),
        mask=head_mask[:, None],
    )


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One decode step of paged attention, reading the block pool in place.

    Args:
        q: Queries ``[num_seqs, num_q_heads, head_dim]``.
        k_cache: One layer's key pool ``[num_blocks, block_size, num_kv_heads,
            head_dim]``.
        v_cache: One layer's value pool, same shape as ``k_cache``.
        block_tables: ``[num_seqs, max_blocks_per_seq]`` int32; entries past a
            sequence's block count may be any valid block id (they are masked).
        seq_lens: ``[num_seqs]`` int32 total lengths including the new token,
            whose K/V must already be written to the pool.
        scale: Softmax scale, ``head_dim ** -0.5``.

    Returns:
        Attention output ``[num_seqs, num_q_heads, head_dim]`` in ``q``'s dtype.
    """
    num_seqs, num_q_heads, head_dim = q.shape
    _num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    group = num_q_heads // num_kv_heads
    if head_dim & (head_dim - 1) or block_size & (block_size - 1):
        raise ValueError(f"head_dim ({head_dim}) and block_size ({block_size}) must be powers of 2")

    out = torch.empty_like(q)
    grid = (num_seqs, num_kv_heads)
    _paged_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        out,
        block_tables,
        seq_lens,
        scale,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *out.stride(),
        *block_tables.stride(),
        GROUP=group,
        GROUP_PAD=triton.next_power_of_2(group),
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return out


class TritonPagedDecodeContext:
    """Kernel-backed decode step context over the shared paged pool.

    Satisfies ``StepContextProtocol``. Writes are one batched scatter per layer
    (no per-row Python loop); reads happen inside the kernel via block tables
    (no gather, no padding copies).

    Args:
        cache: The shared paged pool.
        block_tables: ``[num_seqs, max_blocks]`` int32 device tensor; row ``i``
            holds sequence ``i``'s block table, padded with zeros.
        seq_lens: ``[num_seqs]`` int32 device tensor of lengths including the
            token being decoded.
        slots: ``[num_seqs]`` int64 device tensor with each row's write slot,
            ``block_table[pos // block_size] * block_size + pos % block_size``
            for ``pos = seq_len - 1``.
    """

    def __init__(
        self,
        cache: PagedKVCache,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        self._cache = cache
        self._block_tables = block_tables
        self._seq_lens = seq_lens
        self._slots = slots

    def attend(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Scatter this step's K/V into the pool and attend in place."""
        if q.shape[2] != 1:
            raise ValueError(f"decode steps carry exactly 1 token per row, got {q.shape[2]}")
        k_flat = self._cache.k_cache[layer_idx].flatten(0, 1)
        v_flat = self._cache.v_cache[layer_idx].flatten(0, 1)
        k_flat[self._slots] = k[:, :, 0, :]
        v_flat[self._slots] = v[:, :, 0, :]
        out = paged_attention_decode(
            q[:, :, 0, :],
            self._cache.k_cache[layer_idx],
            self._cache.v_cache[layer_idx],
            self._block_tables,
            self._seq_lens,
            scale=q.shape[-1] ** -0.5,
        )
        return out.unsqueeze(2)
