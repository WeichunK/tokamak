"""The offline batch inference entry point.

The engine is a step loop driven by the iteration-level scheduler: each step is
either one prefill (a newly admitted — or preempted-and-resumed — sequence computes
all its tokens at once) or one batched decode (every running sequence advances one
token). Requests join and leave the running batch at token granularity, which is
what keeps the GPU busy while individual requests start and finish at their own
pace (continuous batching, Orca-style).

KV memory comes from one of two backends:

- ``"paged"`` (default): fixed-size blocks allocated on demand from a pool that is
  preallocated once at engine startup, vLLM-style. When the pool runs dry the
  scheduler preempts the newest requests by recomputation.
- ``"contiguous"`` (M1 baseline): one worst-case buffer per request, kept for
  comparison benchmarks. No preemption; admission is bounded by batch size only.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal, cast

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from tokamak import kernels
from tokamak.config import ModelConfig, resolve_device, resolve_dtype
from tokamak.engine.outputs import RequestOutput
from tokamak.engine.scheduler import ScheduledBatch, Scheduler, StepKind
from tokamak.engine.sequence import FinishReason, Sequence
from tokamak.memory import BlockManager, PagedKVCache, PagedKVCacheView
from tokamak.model.kv_cache import ContiguousKVCache, KVCacheProtocol
from tokamak.model.loader import build_model, load_weights, resolve_model_path
from tokamak.model.step_context import (
    BatchedDecodeContext,
    PrefillContext,
    StepContextProtocol,
)
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

    Not thread-safe: one generate() call at a time.

    Args:
        model: Hugging Face Hub repo id or local checkpoint directory.
        device: ``"auto"`` (default), ``"cuda"``, ``"cpu"``, or an explicit device
            string. ``"auto"`` prefers CUDA.
        dtype: ``"auto"`` (default; bfloat16 on CUDA, float32 on CPU), a torch dtype,
            or its string name.
        max_seq_len: Optional cap on a single request's context length (prompt +
            generation). Defaults to the model's trained maximum.
        kv_backend: ``"paged"`` (default) allocates KV memory in fixed-size blocks
            from a shared pool; ``"contiguous"`` preallocates one worst-case buffer
            per request (the M1 baseline, kept for comparison).
        block_size: Tokens per KV block for the paged backend.
        kv_pool_tokens: Total token capacity of the paged pool, preallocated at
            startup. Defaults to ``max_seq_len``; raise it to admit more
            concurrent sequences. Must be at least ``max_seq_len`` so any single
            request can always run to completion alone (the guarantee that makes
            preemption safe from livelock).
        max_batch_size: Upper bound on concurrently running sequences.
        scheduling: ``"continuous"`` (default) admits requests the moment a slot
            and blocks are free; ``"static"`` fills a batch and lets it drain
            before admitting again (the comparison baseline for benchmarks).
        attention_backend: ``"sdpa"`` is the reference decode path (gather +
            padded SDPA); ``"triton"`` decodes through the paged-attention
            kernel, reading block tables in place (requires the ``triton``
            extra, CUDA, and the paged KV backend). ``"auto"`` (default) picks
            the kernel when available. Prefill always uses SDPA.
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
        kv_pool_tokens: int | None = None,
        max_batch_size: int = 16,
        scheduling: Literal["continuous", "static"] = "continuous",
        attention_backend: Literal["auto", "sdpa", "triton"] = "auto",
    ) -> None:
        if kv_backend not in ("contiguous", "paged"):
            raise ValueError(f"Unknown kv_backend: {kv_backend!r}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
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
        self.max_batch_size = max_batch_size
        self.scheduling = scheduling

        self.kv_backend = kv_backend
        self.block_manager: BlockManager | None = None
        self.paged_cache: PagedKVCache | None = None
        if kv_backend == "paged":
            pool_tokens = kv_pool_tokens or self.max_seq_len
            if pool_tokens < self.max_seq_len:
                raise ValueError(
                    f"kv_pool_tokens ({pool_tokens}) must be >= max_seq_len "
                    f"({self.max_seq_len}) so any single request can complete alone"
                )
            num_blocks = -(-pool_tokens // block_size)
            self.block_manager = BlockManager(num_blocks, block_size)
            self.paged_cache = PagedKVCache(
                self.model_config,
                num_blocks,
                block_size,
                device=self.device,
                dtype=self.dtype,
            )

        self.attention_backend = self._resolve_attention_backend(attention_backend)

        # Per-generate() run state (single-threaded engine).
        self._caches: dict[int, KVCacheProtocol] = {}
        self._generators: dict[int, torch.Generator] = {}

        logger.info(
            "Loading %s (%s) on %s as %s (kv_backend=%s, scheduling=%s, attention=%s)",
            model,
            self.model_config.architecture,
            self.device,
            self.dtype,
            kv_backend,
            scheduling,
            self.attention_backend,
        )
        self.model = build_model(self.model_config, self.device, self.dtype)
        load_weights(self.model, model_path)

    def _resolve_attention_backend(
        self, requested: Literal["auto", "sdpa", "triton"]
    ) -> Literal["sdpa", "triton"]:
        """Pick the decode attention path, validating explicit requests loudly."""
        if requested == "sdpa":
            return "sdpa"
        available = (
            self.kv_backend == "paged" and self.device.type == "cuda" and kernels.is_available()
        )
        if requested == "triton":
            if not available:
                raise ValueError(
                    "attention_backend='triton' requires the paged KV backend, a CUDA "
                    "device, and the triton extra (pip install 'tokamak-llm[triton]')"
                )
            return "triton"
        return "triton" if available else "sdpa"

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
        prompts: str | AbcSequence[str] | None = None,
        sampling_params: SamplingParams | AbcSequence[SamplingParams] | None = None,
        *,
        prompt_token_ids: AbcSequence[AbcSequence[int]] | None = None,
        use_tqdm: bool = True,
    ) -> list[RequestOutput]:
        """Generate a completion for each prompt under continuous batching.

        Args:
            prompts: One prompt or a sequence of prompts. Prompts are tokenized
                as-is; apply a chat template first when talking to a chat model.
            sampling_params: One ``SamplingParams`` shared by every prompt, a
                sequence matching the prompts one-to-one, or ``None`` for defaults.
            prompt_token_ids: Pre-tokenized prompts, mutually exclusive with
                ``prompts`` (used by benchmarks to control lengths exactly).
            use_tqdm: Show a progress bar over completed requests.

        Returns:
            One ``RequestOutput`` per prompt, in input order.
        """
        prompt_texts, token_lists = self._resolve_prompts(prompts, prompt_token_ids)
        params_list = self._broadcast_params(sampling_params, len(token_lists))

        arrival = time.perf_counter()
        sequences = []
        for i, (token_ids, params) in enumerate(zip(token_lists, params_list, strict=True)):
            seq = Sequence(seq_id=i, prompt_token_ids=list(token_ids), sampling_params=params)
            seq.arrival_time = arrival
            self._validate_fits(seq)
            sequences.append(seq)

        scheduler = Scheduler(self.block_manager, self.max_batch_size, self.scheduling)
        for seq in sequences:
            scheduler.add(seq)

        self._caches.clear()
        self._generators.clear()
        try:
            with tqdm(total=len(sequences), desc="Generating", disable=not use_tqdm) as pbar:
                finished = 0
                while scheduler.has_unfinished():
                    batch = scheduler.schedule()
                    if batch is None:
                        raise RuntimeError("scheduler returned no work but is unfinished")
                    self._step(batch)
                    newly_done = sum(s.is_finished for s in sequences) - finished
                    finished += newly_done
                    pbar.update(newly_done)
        finally:
            for cache in self._caches.values():
                cache.release()
            self._caches.clear()
            self._generators.clear()

        return [
            self._to_output(seq, prompt_texts[seq.seq_id] if prompt_texts else None)
            for seq in sequences
        ]

    @torch.inference_mode()
    def _step(self, batch: ScheduledBatch) -> None:
        """Execute one scheduled step (a prefill or a batched decode)."""
        for victim in batch.preempted:
            self._caches.pop(victim.seq_id, None)  # blocks already freed

        if batch.kind is StepKind.PREFILL:
            self._prefill_step(batch.seqs[0])
        else:
            self._decode_step(batch.seqs)

    def _prefill_step(self, seq: Sequence) -> None:
        """Compute a sequence's accumulated tokens and sample its next one."""
        cache = self._make_cache(seq)
        self._caches[seq.seq_id] = cache
        cache.ensure_capacity(seq.num_tokens)

        token_ids = seq.all_token_ids
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(len(token_ids), device=self.device)[None]
        ctx = PrefillContext(cache)
        hidden = self.model(input_ids, positions, ctx)
        logits = self.model.compute_logits(hidden[:, -1])
        self._append_sampled(seq, logits)

    def _decode_step(self, seqs: list[Sequence]) -> None:
        """Advance every running sequence one token in a single forward pass."""
        caches = []
        for seq in seqs:
            cache = self._caches[seq.seq_id]
            cache.ensure_capacity(seq.num_tokens)
            caches.append(cache)

        input_ids = torch.tensor(
            [[seq.last_token_id] for seq in seqs], dtype=torch.long, device=self.device
        )
        positions = torch.tensor(
            [[seq.num_tokens - 1] for seq in seqs], dtype=torch.long, device=self.device
        )
        if self.attention_backend == "triton":
            ctx: StepContextProtocol = self._triton_decode_context(seqs)
        else:
            ctx = BatchedDecodeContext(caches, [seq.num_tokens for seq in seqs], self.device)
        hidden = self.model(input_ids, positions, ctx)
        logits = self.model.compute_logits(hidden[:, -1])
        for i, seq in enumerate(seqs):
            self._append_sampled(seq, logits[i : i + 1])

    def _triton_decode_context(self, seqs: list[Sequence]) -> StepContextProtocol:
        """Build block-table / slot tensors for the kernel-backed decode step."""
        from tokamak.kernels.paged_attention import TritonPagedDecodeContext

        assert self.block_manager is not None and self.paged_cache is not None
        block_size = self.block_manager.block_size
        tables = [self.block_manager.block_table(seq.seq_id) for seq in seqs]
        seq_lens = [seq.num_tokens for seq in seqs]

        max_blocks = max(len(table) for table in tables)
        block_tables = torch.zeros(len(seqs), max_blocks, dtype=torch.int32)
        slots = torch.empty(len(seqs), dtype=torch.int64)
        for i, (table, length) in enumerate(zip(tables, seq_lens, strict=True)):
            block_tables[i, : len(table)] = torch.tensor(table, dtype=torch.int32)
            position = length - 1
            slots[i] = table[position // block_size] * block_size + position % block_size
        return TritonPagedDecodeContext(
            self.paged_cache,
            block_tables.to(self.device),
            torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            slots.to(self.device),
        )

    def _append_sampled(self, seq: Sequence, logits: torch.Tensor) -> None:
        """Sample one token for ``seq``, append it, and apply stop conditions."""
        params = seq.sampling_params
        generator = self._generators.get(seq.seq_id)
        if generator is None and params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)
            self._generators[seq.seq_id] = generator

        token_id = int(sample(logits, params, generator).item())
        seq.append_output_token(token_id)
        if seq.first_token_time is None:
            seq.first_token_time = time.perf_counter()

        if not params.ignore_eos and token_id in self.model_config.eos_token_ids:
            self._finish(seq, FinishReason.STOP)
        elif (
            seq.num_output_tokens >= params.max_new_tokens
            or seq.num_tokens >= self._max_total_tokens(seq)
        ):
            self._finish(seq, FinishReason.LENGTH)

    def _finish(self, seq: Sequence, reason: FinishReason) -> None:
        seq.finish(reason)
        seq.finish_time = time.perf_counter()
        cache = self._caches.pop(seq.seq_id, None)
        if cache is not None:
            cache.release()
        self._generators.pop(seq.seq_id, None)

    def _make_cache(self, seq: Sequence) -> KVCacheProtocol:
        """Build the per-request KV cache for the configured backend.

        Contiguous reserves the request's worst case up front; paged starts empty
        and grows block by block. A fresh view is built on every (re)prefill so
        stale block tables from before a preemption can never be reused.
        """
        if self.kv_backend == "paged":
            assert self.paged_cache is not None and self.block_manager is not None
            return PagedKVCacheView(self.paged_cache, self.block_manager, seq.seq_id)
        return ContiguousKVCache(
            self.model_config,
            max_seq_len=self._max_total_tokens(seq),
            device=self.device,
            dtype=self.dtype,
        )

    def _max_total_tokens(self, seq: Sequence) -> int:
        """A request's context budget, stable across preemptions."""
        return min(self.max_seq_len, seq.num_prompt_tokens + seq.sampling_params.max_new_tokens)

    def _validate_fits(self, seq: Sequence) -> None:
        if seq.num_prompt_tokens >= self.max_seq_len:
            raise ValueError(
                f"Prompt of {seq.num_prompt_tokens} tokens does not fit in "
                f"max_seq_len={self.max_seq_len} (room for generation is required)"
            )
        if self.block_manager is not None:
            needed = self.block_manager.blocks_needed(self._max_total_tokens(seq))
            if needed > self.block_manager.num_blocks:
                raise ValueError(
                    f"Request needs {needed} KV blocks but the pool has "
                    f"{self.block_manager.num_blocks}; raise kv_pool_tokens"
                )

    def _resolve_prompts(
        self,
        prompts: str | AbcSequence[str] | None,
        prompt_token_ids: AbcSequence[AbcSequence[int]] | None,
    ) -> tuple[list[str] | None, list[list[int]]]:
        if (prompts is None) == (prompt_token_ids is None):
            raise ValueError("provide exactly one of `prompts` or `prompt_token_ids`")
        if prompt_token_ids is not None:
            return None, [list(ids) for ids in prompt_token_ids]
        assert prompts is not None  # guaranteed by the exclusivity check above
        if isinstance(prompts, str):
            prompts = [prompts]
        texts = list(prompts)
        return texts, [self.tokenizer.encode(text) for text in texts]

    def _to_output(self, seq: Sequence, prompt_text: str | None) -> RequestOutput:
        if seq.finish_reason is None:
            raise RuntimeError(f"Sequence {seq.seq_id} completed without a finish reason")
        ttft = latency = None
        if seq.arrival_time is not None:
            if seq.first_token_time is not None:
                ttft = seq.first_token_time - seq.arrival_time
            if seq.finish_time is not None:
                latency = seq.finish_time - seq.arrival_time
        return RequestOutput(
            request_id=seq.seq_id,
            prompt=prompt_text if prompt_text is not None else self._decode(seq.prompt_token_ids),
            prompt_token_ids=seq.prompt_token_ids,
            output_text=self._decode(seq.output_token_ids),
            output_token_ids=seq.output_token_ids,
            finish_reason=seq.finish_reason,
            ttft_s=ttft,
            latency_s=latency,
        )

    def _decode(self, token_ids: list[int]) -> str:
        """Detokenize ids, stripping special tokens."""
        # decode() is annotated str | list[str] to cover batched input; a flat list
        # of ids always yields str.
        return cast("str", self.tokenizer.decode(token_ids, skip_special_tokens=True))

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
