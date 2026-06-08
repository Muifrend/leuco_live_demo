from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Callable
from urllib import request

from .config import AppConfig
from .models import AlertState, DecisionState

LOGGER = logging.getLogger(__name__)


@dataclass
class AlertManager:
    config: AppConfig
    opener: Callable[[request.Request, float], object] | None = None
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        self.last_alert_at: float | None = None
        self.last_attempt_at: float | None = None
        self.last_error: str | None = None
        self.state = "idle"

    def maybe_send(self, decision: DecisionState) -> AlertState:
        now = self.clock()

        if not decision.should_alert:
            self.state = "idle"
            self.last_error = None
            return self.snapshot()

        cooldown_started_at = self.last_attempt_at if self.last_attempt_at is not None else self.last_alert_at
        if cooldown_started_at is not None:
            elapsed = now - cooldown_started_at
            if elapsed < self.config.alert_cooldown_seconds:
                self.state = "cooldown"
                return self.snapshot()

        self.last_attempt_at = now
        if not self.config.alerts_enabled:
            self.last_error = None
            self.last_alert_at = now
            self.state = "dry_run"
            LOGGER.info(
                "alert dry-run reason=%s high_risk=%s/%s lost_visibility=%s",
                decision.alert_reason or "sustained high-risk pool behavior",
                decision.high_risk_frames,
                decision.window_size,
                decision.metrics.lost_visibility_frames,
            )
            return self.snapshot()

        try:
            self.last_error = None
            reason = decision.alert_reason or "sustained high-risk pool behavior"
            body = (
                "Sustained high-risk pool behavior detected by the Leuco live demo.\n"
                f"Reason: {reason}\n"
                f"High-risk frames: {decision.high_risk_frames}/{decision.window_size}\n"
                f"Lost visibility frames: {decision.metrics.lost_visibility_frames}"
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
            LOGGER.info(
                "ntfy alert sent url=%s reason=%s high_risk=%s/%s lost_visibility=%s",
                self.config.ntfy_url,
                reason,
                decision.high_risk_frames,
                decision.window_size,
                decision.metrics.lost_visibility_frames,
            )
        except Exception as exc:  # noqa: BLE001 - alert failures must not kill the demo
            self.last_error = str(exc)
            self.state = "failed"
            LOGGER.warning(
                "ntfy alert failed url=%s error=%s",
                self.config.ntfy_url,
                self.last_error,
            )
        return self.snapshot()

    def snapshot(self) -> AlertState:
        return AlertState(
            state=self.state,
            enabled=self.config.alerts_enabled,
            cooldown_seconds=self.config.alert_cooldown_seconds,
            last_alert_at=self.last_alert_at,
            last_error=self.last_error,
        )
