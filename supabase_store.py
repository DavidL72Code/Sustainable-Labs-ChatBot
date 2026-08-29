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

    # -- visitor accounts and history --------------------------------------
    #
    # These use the visitor's own access token, never the service role key, so
    # the row-level policies decide what each visitor can see. Staff tables are
    # never touched here, and these tables are never read with the service role.
    def visitor_sign_up(self, email: str, password: str) -> Optional[dict]:
        if not self.auth_enabled or not email or not password:
            return None
        result = self._request(
            "POST", "/auth/v1/signup", key=self.anon_key,
            body={"email": email, "password": password},
        )
        return result if isinstance(result, dict) else None

    def visitor_sign_in(self, email: str, password: str) -> Optional[dict]:
        """Return the visitor's session, including the access token."""
        if not self.auth_enabled or not email or not password:
            return None
        result = self._request(
            "POST", "/auth/v1/token?grant_type=password", key=self.anon_key,
            body={"email": email, "password": password},
        )
        if not isinstance(result, dict) or not result.get("access_token"):
            return None
        return result

    def visitor_recover(self, email: str, redirect_to: str = "") -> bool:
        """Ask Supabase to email a password reset link.

        Returns True whenever the request was accepted. The caller must not
        expose whether the address actually has an account.
        """
        if not self.auth_enabled or not email:
            return False
        path = "/auth/v1/recover"
        if redirect_to:
            path += "?" + urllib.parse.urlencode({"redirect_to": redirect_to})
        return self._request("POST", path, key=self.anon_key, body={"email": email}) is not None

    def visitor_update_password(self, access_token: str, password: str) -> Optional[dict]:
        """Set a new password using the recovery token from the emailed link.

        Returns the updated user so the caller can open a full session; the
        recovery token is already a valid one.
        """
        if not self.auth_enabled or not access_token or not password:
            return None
        result = self._as_visitor(
            "PUT", "/auth/v1/user", access_token, body={"password": password}
        )
        return result if isinstance(result, dict) else None

    def visitor_refresh(self, refresh_token: str) -> Optional[dict]:
        """Exchange a refresh token so a visitor is not signed out every hour."""
        if not self.auth_enabled or not refresh_token:
            return None
        result = self._request(
            "POST", "/auth/v1/token?grant_type=refresh_token", key=self.anon_key,
            body={"refresh_token": refresh_token},
        )
        if not isinstance(result, dict) or not result.get("access_token"):
            return None
        return result

    def _as_visitor(
        self,
        method: str,
        path: str,
        access_token: str,
        body: Optional[dict | list] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> Optional[Any]:
        if not self.auth_enabled or not access_token:
            return None
        headers = {"Authorization": f"Bearer {access_token}"}
        headers.update(extra_headers or {})
        # The apikey stays the public anon key; the bearer token is the
        # visitor's, so PostgREST evaluates the policies as that visitor.
        return self._request(method, path, key=self.anon_key, body=body, headers=headers)

    def list_visitor_conversations(self, access_token: str, limit: int = 30) -> list[dict]:
        query = urllib.parse.urlencode(
            {"select": "*", "order": "updated_at.desc", "limit": max(1, min(limit, 200))}
        )
        result = self._as_visitor("GET", f"/rest/v1/visitor_conversations?{query}", access_token)
        return result if isinstance(result, list) else []

    def create_visitor_conversation(self, access_token: str, user_id: str, title: str) -> Optional[str]:
        result = self._as_visitor(
            "POST", "/rest/v1/visitor_conversations", access_token,
            body=[{"user_id": user_id, "title": (title or "New chat")[:120]}],
            extra_headers={"Prefer": "return=representation"},
        )
        if isinstance(result, list) and result:
            return str(result[0].get("id", "")) or None
        return None

    def append_visitor_messages(self, access_token: str, rows: list[dict]) -> bool:
        if not rows:
            return False
        result = self._as_visitor(
            "POST", "/rest/v1/visitor_messages", access_token,
            body=rows, extra_headers={"Prefer": "return=minimal"},
        )
        return result is not None

    def fetch_visitor_messages(self, access_token: str, conversation_id: str) -> list[dict]:
        query = urllib.parse.urlencode(
            {"select": "*", "conversation_id": f"eq.{conversation_id}", "order": "created_at.asc"}
        )
        result = self._as_visitor("GET", f"/rest/v1/visitor_messages?{query}", access_token)
        return result if isinstance(result, list) else []

    def touch_visitor_conversation(self, access_token: str, conversation_id: str) -> None:
        query = urllib.parse.urlencode({"id": f"eq.{conversation_id}"})
        self._as_visitor(
            "PATCH", f"/rest/v1/visitor_conversations?{query}", access_token,
            body={"updated_at": "now()"}, extra_headers={"Prefer": "return=minimal"},
        )

    def delete_visitor_conversation(self, access_token: str, conversation_id: str) -> bool:
        """A visitor can always delete their own history."""
        query = urllib.parse.urlencode({"id": f"eq.{conversation_id}"})
        result = self._as_visitor(
            "DELETE", f"/rest/v1/visitor_conversations?{query}", access_token,
            extra_headers={"Prefer": "return=minimal"},
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
        if not isinstance(user, dict):
            return None
        if not self.is_staff(user):
            print(
                f"Rejected dashboard sign-in for {user.get('email', 'unknown')}: "
                "the account is not marked as staff.",
                flush=True,
            )
            return None
        return user

    @staticmethod
    def is_staff(user: dict) -> bool:
        """Only accounts explicitly marked as staff may reach the dashboard.

        Supabase projects allow public email signup by default, and visitor
        accounts may later share this project, so a valid login is not by
        itself proof of employment. app_metadata is writable only with the
        service role key, so a user cannot grant themselves this role.
        """
        metadata = user.get("app_metadata")
        if not isinstance(metadata, dict):
            return False
        if str(metadata.get("role", "")).strip().lower() == "staff":
            return True
        roles = metadata.get("roles")
        if isinstance(roles, list):
            return any(str(role).strip().lower() == "staff" for role in roles)
        return False
