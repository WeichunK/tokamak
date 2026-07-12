"""The offline batch inference entry point.

M1 scope: requests are processed one at a time (prefill, then token-by-token decode)
with a contiguous per-sequence KV cache. This is intentionally the simplest correct
engine — it is the baseline that paged KV caching (M2) and continuous batching (M3)
are measured against.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from tokamak.config import ModelConfig, resolve_device, resolve_dtype
from tokamak.engine.outputs import RequestOutput
from tokamak.engine.sequence import FinishReason, Sequence, SequenceStatus
from tokamak.model.kv_cache import KVCache
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
            to the model's trained maximum; lowering it bounds KV-cache memory.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
        max_seq_len: int | None = None,
    ) -> None:
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

        logger.info(
            "Loading %s (%s) on %s as %s",
            model,
            self.model_config.architecture,
            self.device,
            self.dtype,
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

        kv_cache = KVCache(
            self.model_config,
            max_seq_len=max_total_tokens,
            device=self.device,
            dtype=self.dtype,
        )
        generator: torch.Generator | None = None
        if params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)

        seq.status = SequenceStatus.RUNNING
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

            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            hidden = self.model(input_ids, kv_cache, start_pos=seq.num_tokens - 1)
            token_id = self._sample_next(hidden, params, generator)

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
