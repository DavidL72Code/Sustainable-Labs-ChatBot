"""Supabase-backed storage for the staff dashboard.

Talks to PostgREST and GoTrue over HTTPS with the standard library, so the
deployment does not need a Postgres driver in the image.

Two rules shape this module:
  * Nothing here may break a chat response. Every network call is wrapped, and
    writes are dispatched on a background thread so a slow or unreachable
    Supabase never adds latency to a user's answer.
  * Ordinary chats leave no transcript. Only numbers go to chat_metrics; the
    question and answer are written only when an answer is flagged for review.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

_TIMEOUT_SECONDS = float(os.getenv("SUPABASE_TIMEOUT_SECONDS", "8"))


class SupabaseStore:
    def __init__(
        self,
        url: str = "",
        service_key: str = "",
        anon_key: str = "",
        timeout: float = _TIMEOUT_SECONDS,
    ) -> None:
        self.url = (url or os.getenv("SUPABASE_URL", "")).strip().rstrip("/")
        self.service_key = (service_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")).strip()
        self.anon_key = (anon_key or os.getenv("SUPABASE_ANON_KEY", "")).strip()
        self.timeout = timeout
        self._warned = False

    @property
    def enabled(self) -> bool:
        """True when writes and reads should go to Supabase."""
        return bool(self.url and self.service_key)

    @property
    def auth_enabled(self) -> bool:
        """True when employees sign in through Supabase Auth."""
        return bool(self.url and self.anon_key)

    # -- plumbing ----------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        key: str,
        body: Optional[dict | list] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Optional[Any]:
        request_headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        request_headers.update(headers or {})
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self.url}{path}", data=data, headers=request_headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw.strip() else True
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            self._warn(f"Supabase {method} {path} failed ({exc.code}): {detail}")
        except Exception as exc:
            self._warn(f"Supabase {method} {path} failed: {type(exc).__name__}: {exc}")
        return None

    def _warn(self, message: str) -> None:
        # Log the first failure per process. A broken Supabase should be
        # visible in the logs without flooding them on every request.
        if not self._warned:
            print(message, flush=True)
            self._warned = True

    def _insert_async(self, table: str, row: dict) -> None:
        if not self.enabled:
            return

        def worker() -> None:
            self._request(
                "POST",
                f"/rest/v1/{table}",
                key=self.service_key,
                body=[row],
                headers={"Prefer": "return=minimal,resolution=merge-duplicates"},
            )

        threading.Thread(target=worker, name=f"supabase-{table}", daemon=True).start()

    # -- writes ------------------------------------------------------------
    def record_chat_metrics(self, row: dict) -> None:
        """One content-free row per answer."""
        self._insert_async("chat_metrics", row)

    def record_flagged_chat(self, row: dict) -> None:
        """Full transcript, written only for answers flagged for review."""
        self._insert_async("flagged_chats", row)

    def record_audit_event(self, row: dict) -> None:
        self._insert_async("admin_audit_events", row)

    # -- reads -------------------------------------------------------------
    def fetch_flagged_chats(self, limit: int = 50) -> list[dict]:
        if not self.enabled:
            return []
        # flagged_chats.id references chat_metrics.id, so PostgREST can embed
        # the numbers alongside the transcript in one request.
        query = urllib.parse.urlencode(
            {
                "select": "*,chat_metrics(*)",
                "order": "created_at.desc",
                "limit": max(1, min(limit, 500)),
            }
        )
        result = self._request("GET", f"/rest/v1/flagged_chats?{query}", key=self.service_key)
        return result if isinstance(result, list) else []

    def fetch_chat_metrics(self, limit: int = 200) -> list[dict]:
        if not self.enabled:
            return []
        query = urllib.parse.urlencode(
            {"select": "*", "order": "created_at.desc", "limit": max(1, min(limit, 2000))}
        )
        result = self._request("GET", f"/rest/v1/chat_metrics?{query}", key=self.service_key)
        return result if isinstance(result, list) else []

    def fetch_daily_metrics(self, days: int = 30) -> list[dict]:
        if not self.enabled:
            return []
        query = urllib.parse.urlencode(
            {"select": "*", "order": "day.desc", "limit": max(1, min(days, 365))}
        )
        result = self._request("GET", f"/rest/v1/daily_metrics?{query}", key=self.service_key)
        return result if isinstance(result, list) else []

    def fetch_audit_events(self, limit: int = 50) -> list[dict]:
        if not self.enabled:
            return []
        query = urllib.parse.urlencode(
            {"select": "*", "order": "created_at.desc", "limit": max(1, min(limit, 500))}
        )
        result = self._request("GET", f"/rest/v1/admin_audit_events?{query}", key=self.service_key)
        return result if isinstance(result, list) else []

    def mark_reviewed(self, event_id: str, username: str, note: str = "") -> bool:
        if not self.enabled or not event_id:
            return False
        query = urllib.parse.urlencode({"id": f"eq.{event_id}"})
        body = {"reviewed_by": username, "reviewed_at": "now()", "review_note": note or None}
        result = self._request(
            "PATCH",
            f"/rest/v1/flagged_chats?{query}",
            key=self.service_key,
            body=body,
            headers={"Prefer": "return=minimal"},
        )
        return result is not None

    # -- auth --------------------------------------------------------------
    def sign_in(self, email: str, password: str) -> Optional[dict]:
        """Verify one employee against Supabase Auth.

        Returns the user record on success and None otherwise. Staff are
        managed from the Supabase dashboard, so adding or removing an employee
        needs no restart and no redeploy.
        """
        if not self.auth_enabled or not email or not password:
            return None
        result = self._request(
            "POST",
            "/auth/v1/token?grant_type=password",
            key=self.anon_key,
            body={"email": email, "password": password},
        )
        if not isinstance(result, dict):
            return None
        user = result.get("user")
        return user if isinstance(user, dict) else None
