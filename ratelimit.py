"""
Per-session sliding-window rate limiter for the public HF Space.

Goals
-----
1. Protect a user from accidentally burning through their own Anthropic / Tavily
   quota if they spam-click "Generate".
2. Protect the shared Gradio process from being overwhelmed by a single session
   (each agent invocation blocks a queue worker).
3. Enforce a reasonable upper bound per session per hour.

Limits are sliding windows (not fixed buckets) so a burst right before a window
boundary can't slip through.
"""

from __future__ import annotations

from collections import deque
from time import time
from typing import Deque, Dict, Iterable, Tuple


# Action name → list of (window_seconds, max_requests)
DEFAULT_LIMITS: Dict[str, Iterable[Tuple[int, int]]] = {
    # Agent invocations are expensive (multi-tool LLM calls). Keep burst low.
    "analyze": ((60, 3), (300, 8), (3600, 30)),
    "rank": ((60, 2), (300, 5), (3600, 15)),
    # PDF indexing is CPU-bound on the Space. Tight cap.
    "upload": ((300, 3), (3600, 10)),
}


class SessionRateLimiter:
    """
    Keeps a deque of timestamps per action and enforces sliding-window limits.

    Not thread-safe in the strict sense — Gradio runs callbacks in a threadpool,
    but each session's state object is only touched by that session's own
    callbacks, and within a session Gradio serializes by default. Good enough.
    """

    def __init__(self, limits: Dict[str, Iterable[Tuple[int, int]]] = None):
        self.limits = limits or DEFAULT_LIMITS
        self._timestamps: Dict[str, Deque[float]] = {
            action: deque() for action in self.limits
        }

    def check(self, action: str) -> Tuple[bool, str]:
        """
        Returns (allowed, human-readable reason). Records the timestamp if allowed.
        """
        if action not in self.limits:
            return True, ""

        now = time()
        ts = self._timestamps[action]
        rules = list(self.limits[action])
        # Prune entries older than the longest window — they can't matter anymore.
        longest = max(w for w, _ in rules)
        while ts and ts[0] < now - longest:
            ts.popleft()

        for window, max_req in sorted(rules):
            count = sum(1 for t in ts if t > now - window)
            if count >= max_req:
                # Time until the oldest-in-window entry expires.
                retry_in = int((ts[0] + window) - now) if ts else window
                retry_in = max(retry_in, 1)
                return (
                    False,
                    f"Rate limit reached ({max_req} {action} requests per "
                    f"{_humanize(window)}). Try again in {_humanize(retry_in)}.",
                )

        ts.append(now)
        return True, ""

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        """Returns current usage counters per action, useful for UI display."""
        now = time()
        out: Dict[str, Dict[str, int]] = {}
        for action, rules in self.limits.items():
            ts = self._timestamps[action]
            out[action] = {
                _humanize(window): sum(1 for t in ts if t > now - window)
                for window, _ in rules
            }
        return out


def _humanize(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"
