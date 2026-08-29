"""Grant or revoke dashboard access for an employee.

The staff role is what separates an employee from a visitor who signed up, so
it is granted deliberately rather than inferred from how the account was made.
Revoking it takes effect on that person's next request.

    python3 make_staff.py someone@umb.edu           # grant
    python3 make_staff.py someone@umb.edu --remove  # revoke
    python3 make_staff.py --list                    # show everyone and their role

Needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY, which are read from .env.
The role lives in app_metadata, which only the service role key can write, so a
user can never grant it to themselves.
"""

from __future__ import annotations

import sys
import urllib.parse

from dotenv import load_dotenv

load_dotenv()

from supabase_store import SupabaseStore  # noqa: E402

STAFF_ROLE = "staff"


def find_user(store: SupabaseStore, email: str) -> dict | None:
    query = urllib.parse.urlencode({"page": 1, "per_page": 200})
    result = store._request("GET", f"/auth/v1/admin/users?{query}", key=store.service_key)
    users = (result or {}).get("users", []) if isinstance(result, dict) else []
    target = email.strip().lower()
    return next((u for u in users if str(u.get("email", "")).lower() == target), None)


def list_users(store: SupabaseStore) -> int:
    query = urllib.parse.urlencode({"page": 1, "per_page": 200})
    result = store._request("GET", f"/auth/v1/admin/users?{query}", key=store.service_key)
    if not isinstance(result, dict):
        print("Could not reach Supabase. Check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
        return 1
    users = result.get("users", [])
    if not users:
        print("No accounts yet.")
        return 0
    print(f"{'email':42} {'role':8} confirmed")
    for user in sorted(users, key=lambda u: str(u.get("email", ""))):
        role = (user.get("app_metadata") or {}).get("role") or "-"
        confirmed = "yes" if user.get("email_confirmed_at") else "no"
        print(f"{str(user.get('email','')):42} {role:8} {confirmed}")
    staff = sum(1 for u in users if (u.get("app_metadata") or {}).get("role") == STAFF_ROLE)
    print(f"\n{len(users)} account(s), {staff} with dashboard access.")
    return 0


def set_role(store: SupabaseStore, email: str, grant: bool) -> int:
    user = find_user(store, email)
    if not user:
        print(f"No account found for {email}.")
        print("Create it first under Authentication -> Users -> Add user.")
        return 1

    metadata = dict(user.get("app_metadata") or {})
    current = metadata.get("role")
    if grant and current == STAFF_ROLE:
        print(f"{email} already has dashboard access.")
        return 0
    if not grant and current != STAFF_ROLE:
        print(f"{email} does not have dashboard access.")
        return 0

    # Send the whole app_metadata back with role added or dropped; Supabase
    # merges top-level keys, so removing one means writing the rest without it.
    if grant:
        metadata["role"] = STAFF_ROLE
    else:
        metadata.pop("role", None)
        metadata["role"] = None  # an explicit null clears it

    updated = store._request(
        "PUT", f"/auth/v1/admin/users/{user['id']}",
        key=store.service_key, body={"app_metadata": metadata},
    )
    if not isinstance(updated, dict):
        print("Supabase rejected the change. Is SUPABASE_SERVICE_ROLE_KEY the service_role key?")
        return 1

    now = (updated.get("app_metadata") or {}).get("role")
    if grant and now == STAFF_ROLE:
        print(f"{email} can now sign in to the dashboard.")
        return 0
    if not grant and now != STAFF_ROLE:
        print(f"{email} no longer has dashboard access. It ends on their next request.")
        return 0
    print(f"The change did not stick: role is now {now!r}.")
    return 1


def main() -> int:
    args = [a for a in sys.argv[1:] if a not in ("--remove", "--list")]
    remove = "--remove" in sys.argv[1:]
    listing = "--list" in sys.argv[1:]

    store = SupabaseStore()
    if not store.url or not store.service_key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (in .env) first.")
        return 1
    if listing:
        return list_users(store)
    if not args:
        print(__doc__)
        return 1
    return max(set_role(store, email, grant=not remove) for email in args)


if __name__ == "__main__":
    raise SystemExit(main())
