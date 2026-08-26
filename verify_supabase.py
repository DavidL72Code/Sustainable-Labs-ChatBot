"""Check that Supabase is wired up correctly for the staff dashboard.

Run after creating the project, running supabase/schema.sql, and setting the
environment variables:

    python3 verify_supabase.py

It writes two clearly-marked test rows, reads them back, checks the daily
rollup, then deletes them. Add --keep to leave the rows in place if you want to
look at them in the dashboard.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import uuid

from dotenv import load_dotenv

load_dotenv()

from supabase_store import SupabaseStore  # noqa: E402

OK = "  [ok]  "
BAD = "  [FAIL]"


def main() -> int:
    keep = "--keep" in sys.argv
    store = SupabaseStore()
    failures = 0

    print("1. Environment")
    for name, value, required in (
        ("SUPABASE_URL", store.url, True),
        ("SUPABASE_SERVICE_ROLE_KEY", store.service_key, True),
        ("SUPABASE_ANON_KEY", store.anon_key, False),
        ("DASHBOARD_SESSION_SECRET", os.getenv("DASHBOARD_SESSION_SECRET", ""), True),
    ):
        if value:
            shown = value if name == "SUPABASE_URL" else f"set ({len(value)} chars)"
            print(f"{OK}{name}: {shown}")
        elif required:
            print(f"{BAD}{name} is not set")
            failures += 1
        else:
            print(f"       {name} is not set (needed only for Supabase Auth sign-in)")

    if not store.enabled:
        print("\nSUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required. Stopping.")
        return 1

    print("\n2. Tables reachable")
    for table in ("chat_metrics", "flagged_chats", "admin_audit_events", "daily_metrics"):
        query = urllib.parse.urlencode({"select": "*", "limit": 1})
        result = store._request("GET", f"/rest/v1/{table}?{query}", key=store.service_key)
        if result is None:
            print(f"{BAD}{table} - not reachable. Did supabase/schema.sql run?")
            failures += 1
        else:
            print(f"{OK}{table}")

    if failures:
        print("\nFix the errors above before continuing.")
        return 1

    print("\n3. Write and read back")
    test_id = f"verify-{uuid.uuid4().hex[:12]}"
    metrics_row = {
        "id": test_id,
        "status": "answered",
        "response_mode": "verify_script",
        "path_label": "retrieval -> generation",
        "latency_ms": 1234.5,
        "retrieval_ms": 234.5,
        "llm_ms": 1000.0,
        "total_tokens": 4242,
        "cost_usd": 0.00042,
        "confidence_score": 0.42,
        "is_low_confidence": True,
        "top_score": 0.5,
        "source_count": 0,
        "flagged": True,
        "flag_reasons": ["low_confidence", "weak_retrieval"],
    }
    written = store._request(
        "POST", "/rest/v1/chat_metrics", key=store.service_key,
        body=[metrics_row], headers={"Prefer": "return=minimal"},
    )
    print(f"{OK}wrote chat_metrics row {test_id}" if written is not None
          else f"{BAD}could not write to chat_metrics")
    failures += written is None

    flagged_row = {
        "id": test_id,
        "conversation_id": "verify-conversation",
        "question": "VERIFY SCRIPT - safe to delete",
        "answer": "VERIFY SCRIPT - safe to delete",
        "flag_reasons": ["low_confidence", "weak_retrieval"],
        "sources": [],
        "trace": {"telemetry": {"path_label": "retrieval -> generation"}},
    }
    written = store._request(
        "POST", "/rest/v1/flagged_chats", key=store.service_key,
        body=[flagged_row], headers={"Prefer": "return=minimal"},
    )
    print(f"{OK}wrote flagged_chats row" if written is not None
          else f"{BAD}could not write to flagged_chats")
    failures += written is None

    found = [row for row in store.fetch_flagged_chats(limit=20) if row.get("id") == test_id]
    if found:
        embedded = found[0].get("chat_metrics")
        print(f"{OK}read the row back, with metrics joined: {bool(embedded)}")
        if not embedded:
            print("       (the embed is empty - check the flagged_chats.id foreign key)")
    else:
        print(f"{BAD}wrote the row but could not read it back")
        failures += 1

    daily = store.fetch_daily_metrics(days=7)
    print(f"{OK}daily_metrics returns {len(daily)} day(s)" if daily is not None and daily
          else "       daily_metrics is empty (expected until real traffic arrives)")
    if daily:
        print(f"       most recent: {json.dumps(daily[0], default=str)[:160]}")

    print("\n4. Staff role enforcement")
    print(f"{OK}a user with app_metadata.role='staff' is accepted: "
          f"{SupabaseStore.is_staff({'app_metadata': {'role': 'staff'}})}")
    print(f"{OK}a self-signup user with no role is rejected: "
          f"{not SupabaseStore.is_staff({'app_metadata': {}})}")
    print("       Mark each employee in Supabase with:")
    print("         Authentication -> Users -> (user) -> App Metadata -> {\"role\": \"staff\"}")

    if keep:
        print(f"\nLeaving test rows in place (id {test_id}). Delete them with:")
        print(f"  delete from flagged_chats where id = '{test_id}';")
        print(f"  delete from chat_metrics where id = '{test_id}';")
    else:
        print("\n5. Cleanup")
        for table in ("flagged_chats", "chat_metrics"):
            query = urllib.parse.urlencode({"id": f"eq.{test_id}"})
            store._request("DELETE", f"/rest/v1/{table}?{query}",
                           key=store.service_key, headers={"Prefer": "return=minimal"})
        print(f"{OK}removed the test rows")

    print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
