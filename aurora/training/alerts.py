from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any
from urllib.parse import urlparse

from aurora.provenance import utc_now

ALERT_WEBHOOK_ENV = "AURORA_ALERT_WEBHOOK"


class AlertNotifier:
    """Post JSON alerts to an optional webhook without failing training."""

    def __init__(
        self,
        url: str,
        logger: logging.Logger,
        timeout: float = 5.0,
    ) -> None:
        self.url = url.strip()
        self.logger = logger
        self.timeout = timeout
        if self.url:
            parsed = urlparse(self.url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("alert webhook must be an http(s) URL")
        if self.timeout <= 0:
            raise ValueError("alert timeout must be positive")

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def send(self, event: str, payload: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        body = json.dumps(
            {"event": event, "created_utc": utc_now(), **payload},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read(1024)
        except (OSError, ValueError) as error:
            self.logger.warning("Alert %r delivery failed: %s", event, error)
            return False
        return True
