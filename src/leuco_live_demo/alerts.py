from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable
from urllib import request

from .config import AppConfig
from .models import AlertState, DecisionState


@dataclass
class AlertManager:
    config: AppConfig
    opener: Callable[[request.Request, float], object] | None = None
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        self.last_alert_at: float | None = None
        self.last_error: str | None = None
        self.state = "idle"

    def maybe_send(self, decision: DecisionState) -> AlertState:
        now = self.clock()
        self.last_error = None

        if not decision.should_alert:
            self.state = "idle"
            return self.snapshot()

        if self.last_alert_at is not None:
            elapsed = now - self.last_alert_at
            if elapsed < self.config.alert_cooldown_seconds:
                self.state = "cooldown"
                return self.snapshot()

        if not self.config.alerts_enabled:
            self.last_alert_at = now
            self.state = "dry_run"
            return self.snapshot()

        try:
            body = (
                "Sustained high-risk pool behavior detected by the Leuco live demo.\n"
                f"High-risk frames: {decision.high_risk_frames}/{decision.window_size}"
            ).encode("utf-8")
            req = request.Request(
                self.config.ntfy_url,
                data=body,
                method="POST",
                headers={"Title": self.config.ntfy_title},
            )
            opener = self.opener or request.urlopen
            opener(req, 3.0)
            self.last_alert_at = now
            self.state = "sent"
        except Exception as exc:  # noqa: BLE001 - alert failures must not kill the demo
            self.last_error = str(exc)
            self.state = "failed"
        return self.snapshot()

    def snapshot(self) -> AlertState:
        return AlertState(
            state=self.state,
            enabled=self.config.alerts_enabled,
            cooldown_seconds=self.config.alert_cooldown_seconds,
            last_alert_at=self.last_alert_at,
            last_error=self.last_error,
        )
