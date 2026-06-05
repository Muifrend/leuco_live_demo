from __future__ import annotations

from .models import PoolState, Track


class PoolGate:
    def __init__(self, mode: str) -> None:
        if mode not in {"disabled", "full_frame_stub"}:
            raise ValueError(f"Unsupported pool gate mode: {mode}")
        self.mode = mode

    def evaluate(self, track: Track | None) -> PoolState:
        if self.mode == "disabled":
            return PoolState(
                mode=self.mode,
                in_pool=False,
                active=False,
                label="Pool gate disabled",
            )
        if track is None:
            return PoolState(
                mode=self.mode,
                in_pool=False,
                active=True,
                label="Full-frame pool stub",
            )
        return PoolState(
            mode=self.mode,
            in_pool=True,
            active=True,
            label="Full-frame pool stub",
        )
