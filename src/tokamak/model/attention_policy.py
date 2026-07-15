"""Attention visibility policies: which cached positions a query may see.

A policy is a pure value object interpreted by the step contexts
(:mod:`tokamak.model.step_context`, :mod:`tokamak.kernels.paged_attention`);
it owns no tensors and does no math beyond index arithmetic. ``full`` is
today's behavior. ``window(W)`` sees only the last ``W`` positions (the query
included). ``streaming(W, S)`` additionally keeps the first ``S`` positions
always visible — the StreamingLLM refinement, motivated by softmax attention
parking surplus probability mass on early "sink" tokens; evicting them is what
makes a plain window collapse once context outgrows the window.

Positions stay absolute (no StreamingLLM position rolling): policies bound the
*cache*, not context extrapolation, so evaluation belongs inside the model's
trained context length.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionPolicy:
    """Visibility rule for one query at absolute position ``pos``.

    Visible set: ``[0, sinks) plus [band_start(pos), pos]`` where
    ``band_start(pos) = max(sinks, pos - window + 1)`` (``0`` when unwindowed).

    Args:
        window: Number of most recent positions visible, query included;
            ``None`` means unbounded (full attention).
        sinks: Always-visible prefix positions. Only meaningful under a
            window — full attention already sees them.
    """

    window: int | None = None
    sinks: int = 0

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.sinks < 0:
            raise ValueError(f"sinks must be >= 0, got {self.sinks}")
        if self.window is None and self.sinks > 0:
            raise ValueError("sinks require a window (full attention already sees them)")

    @property
    def is_full(self) -> bool:
        """True when every past position is visible (no masking needed)."""
        return self.window is None

    def band_start(self, pos: int) -> int:
        """First visible position of the recency band for a query at ``pos``.

        Clipped to ``sinks`` so the band never overlaps the sink prefix: the
        two pieces of the visible set stay disjoint, which the kernel's
        two-phase block loop relies on.
        """
        if self.window is None:
            return 0
        return max(self.sinks, pos - self.window + 1)

    @classmethod
    def parse(cls, spec: str | AttentionPolicy) -> AttentionPolicy:
        """Parse ``"full"``, ``"window:512"``, or ``"streaming:512+4"``.

        ``streaming:W+S`` reads as "a window of W plus S sinks".
        """
        if isinstance(spec, AttentionPolicy):
            return spec
        if spec == "full":
            return cls()
        kind, sep, arg = spec.partition(":")
        if sep:
            try:
                if kind == "window":
                    return cls(window=int(arg))
                if kind == "streaming":
                    window, plus, sinks = arg.partition("+")
                    if plus:
                        return cls(window=int(window), sinks=int(sinks))
            except ValueError as err:  # int() failures get the uniform message
                raise ValueError(f"Unknown attention_policy: {spec!r}") from err
        raise ValueError(
            f"Unknown attention_policy: {spec!r} (expected 'full', 'window:W', or 'streaming:W+S')"
        )


FULL_ATTENTION = AttentionPolicy()
