"""The offline batch inference entry point.

Requests are processed one at a time (prefill, then token-by-token decode); the
iteration-level scheduler arrives with continuous batching in M3. KV memory comes
from one of two backends:

- ``"paged"`` (default, M2): fixed-size blocks allocated on demand from a pool that
  is preallocated once at engine startup, vLLM-style. Per-sequence waste is bounded
  by one partial block.
- ``"contiguous"`` (M1 baseline): one buffer per request sized for its worst case
  (``prompt + max_new_tokens``), kept for comparison benchmarks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from tokamak.config import ModelConfig, resolve_device, resolve_dtype
from tokamak.engine.outputs import RequestOutput
from tokamak.engine.sequence import FinishReason, Sequence, SequenceStatus
from tokamak.memory import BlockManager, PagedKVCache, PagedKVCacheView
from tokamak.model.kv_cache import ContiguousKVCache, KVCacheProtocol
from tokamak.model.loader import build_model, load_weights, resolve_model_path
from tokamak.sampling.sampler import sample
from tokamak.sampling_params import SamplingParams

if TYPE_CHECKING:
    from collections.abc import Sequence as AbcSequence
    from pathlib import Path

logger = logging.getLogger(__name__)


class LLM:
    """Loads a model once and serves batch generation requests.

    The API shape mirrors vLLM's offline ``LLM`` class::

        llm = LLM("Qwen/Qwen3-0.6B")
        outputs = llm.generate(["Hello,"], SamplingParams(temperature=0.0))

    Args:
        model: Hugging Face Hub repo id or local checkpoint directory.
        device: ``"auto"`` (default), ``"cuda"``, ``"cpu"``, or an explicit device
            string. ``"auto"`` prefers CUDA.
        dtype: ``"auto"`` (default; bfloat16 on CUDA, float32 on CPU), a torch dtype,
            or its string name.
        max_seq_len: Optional cap on context length (prompt + generation). Defaults
            to the model's trained maximum; lowering it bounds KV-cache memory —
            with the paged backend, the whole pool is preallocated from this value.
        kv_backend: ``"paged"`` (default) allocates KV memory in fixed-size blocks
            from a shared pool; ``"contiguous"`` preallocates one worst-case buffer
            per request (the M1 baseline, kept for comparison).
        block_size: Tokens per KV block for the paged backend.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
        max_seq_len: int | None = None,
        kv_backend: Literal["contiguous", "paged"] = "paged",
        block_size: int = 16,
    ) -> None:
        if kv_backend not in ("contiguous", "paged"):
            raise ValueError(f"Unknown kv_backend: {kv_backend!r}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        model_path = resolve_model_path(model)
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_config = AutoConfig.from_pretrained(model_path)
        self.model_config = ModelConfig.from_hf(
            hf_config, eos_token_ids=self._resolve_eos_ids(model_path)
        )

        model_max = self.model_config.max_position_embeddings
        self.max_seq_len = min(max_seq_len or model_max, model_max)

        self.kv_backend = kv_backend
        self.block_manager: BlockManager | None = None
        self.paged_cache: PagedKVCache | None = None
        if kv_backend == "paged":
            num_blocks = -(-self.max_seq_len // block_size)
            self.block_manager = BlockManager(num_blocks, block_size)
            self.paged_cache = PagedKVCache(
                self.model_config,
                num_blocks,
                block_size,
                device=self.device,
                dtype=self.dtype,
            )

        logger.info(
            "Loading %s (%s) on %s as %s (kv_backend=%s)",
            model,
            self.model_config.architecture,
            self.device,
            self.dtype,
            kv_backend,
        )
        self.model = build_model(self.model_config, self.device, self.dtype)
        load_weights(self.model, model_path)

    def _resolve_eos_ids(self, model_path: Path) -> tuple[int, ...]:
        """Resolve EOS ids from the generation config, falling back to the tokenizer.

        Chat-tuned checkpoints frequently stop on tokens other than the tokenizer's
        ``eos_token_id`` (e.g. Qwen's ``<|im_end|>``), and record them only in
        ``generation_config.json``.
        """
        eos_ids: list[int] = []
        try:
            generation_config = GenerationConfig.from_pretrained(model_path)
            raw = generation_config.eos_token_id
            if isinstance(raw, int):
                eos_ids = [raw]
            elif isinstance(raw, list):
                eos_ids = list(raw)
        except OSError:
            pass  # no generation_config.json in the checkpoint
        if not eos_ids and self.tokenizer.eos_token_id is not None:
            eos_ids = [self.tokenizer.eos_token_id]
        return tuple(eos_ids)

    def generate(
        self,
        prompts: str | AbcSequence[str],
        sampling_params: SamplingParams | AbcSequence[SamplingParams] | None = None,
        *,
        use_tqdm: bool = True,
    ) -> list[RequestOutput]:
        """Generate a completion for each prompt.

        Args:
            prompts: One prompt or a sequence of prompts. Prompts are tokenized
                as-is; apply a chat template first when talking to a chat model.
            sampling_params: One ``SamplingParams`` shared by every prompt, a
                sequence matching ``prompts`` one-to-one, or ``None`` for defaults.
            use_tqdm: Show a progress bar over requests.

        Returns:
            One ``RequestOutput`` per prompt, in input order.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        params_list = self._broadcast_params(sampling_params, len(prompts))

        sequences = [
            Sequence(
                seq_id=i,
                prompt_token_ids=self.tokenizer.encode(prompt),
                sampling_params=params,
            )
            for i, (prompt, params) in enumerate(zip(prompts, params_list, strict=True))
        ]

        for seq in tqdm(sequences, desc="Generating", disable=not use_tqdm):
            self._run_sequence(seq)

        return [
            RequestOutput(
                request_id=seq.seq_id,
                prompt=prompts[seq.seq_id],
                prompt_token_ids=seq.prompt_token_ids,
                output_text=self._decode(seq.output_token_ids),
                output_token_ids=seq.output_token_ids,
                finish_reason=self._require_finish_reason(seq),
            )
            for seq in sequences
        ]

    @torch.inference_mode()
    def _run_sequence(self, seq: Sequence) -> None:
        """Prefill the prompt, then decode until a stop condition is met."""
        params = seq.sampling_params
        if seq.num_prompt_tokens >= self.max_seq_len:
            raise ValueError(
                f"Prompt of {seq.num_prompt_tokens} tokens does not fit in "
                f"max_seq_len={self.max_seq_len} (room for generation is required)"
            )
        max_total_tokens = min(self.max_seq_len, seq.num_prompt_tokens + params.max_new_tokens)

        kv_cache = self._make_cache(seq, max_total_tokens)
        generator: torch.Generator | None = None
        if params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)

        seq.status = SequenceStatus.RUNNING
        try:
            kv_cache.ensure_capacity(seq.num_prompt_tokens)
            input_ids = torch.tensor([seq.prompt_token_ids], dtype=torch.long, device=self.device)
            hidden = self.model(input_ids, kv_cache, start_pos=0)
            token_id = self._sample_next(hidden, params, generator)

            while True:
                seq.append_output_token(token_id)
                if not params.ignore_eos and token_id in self.model_config.eos_token_ids:
                    seq.finish(FinishReason.STOP)
                    return
                if seq.num_tokens >= max_total_tokens:
                    seq.finish(FinishReason.LENGTH)
                    return

                kv_cache.ensure_capacity(seq.num_tokens)
                input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                hidden = self.model(input_ids, kv_cache, start_pos=seq.num_tokens - 1)
                token_id = self._sample_next(hidden, params, generator)
        finally:
            kv_cache.release()

    def _make_cache(self, seq: Sequence, max_total_tokens: int) -> KVCacheProtocol:
        """Build the per-request KV cache for the configured backend.

        Contiguous reserves ``max_total_tokens`` up front; paged starts empty and
        grows block by block via ``ensure_capacity`` as the sequence advances.
        """
        if self.kv_backend == "paged":
            assert self.paged_cache is not None and self.block_manager is not None
            return PagedKVCacheView(self.paged_cache, self.block_manager, seq.seq_id)
        return ContiguousKVCache(
            self.model_config,
            max_seq_len=max_total_tokens,
            device=self.device,
            dtype=self.dtype,
        )

    def _decode(self, token_ids: list[int]) -> str:
        """Detokenize generated ids, stripping special tokens."""
        # decode() is annotated str | list[str] to cover batched input; a flat list
        # of ids always yields str.
        return cast("str", self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def _sample_next(
        self,
        hidden: torch.Tensor,
        params: SamplingParams,
        generator: torch.Generator | None,
    ) -> int:
        """Project the last position to logits and sample one token."""
        logits = self.model.compute_logits(hidden[:, -1])
        return int(sample(logits, params, generator).item())

    @staticmethod
    def _broadcast_params(
        sampling_params: SamplingParams | AbcSequence[SamplingParams] | None,
        num_prompts: int,
    ) -> list[SamplingParams]:
        if sampling_params is None:
            return [SamplingParams()] * num_prompts
        if isinstance(sampling_params, SamplingParams):
            return [sampling_params] * num_prompts
        params_list = list(sampling_params)
        if len(params_list) != num_prompts:
            raise ValueError(f"Got {len(params_list)} SamplingParams for {num_prompts} prompts")
        return params_list

    @staticmethod
    def _require_finish_reason(seq: Sequence) -> FinishReason:
        if seq.finish_reason is None:
            raise RuntimeError(f"Sequence {seq.seq_id} completed without a finish reason")
        return seq.finish_reason
