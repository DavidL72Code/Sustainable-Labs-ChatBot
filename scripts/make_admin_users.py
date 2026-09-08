"""Build the ADMIN_USERS_JSON secret for the staff dashboard.

Each dashboard user gets their own username and password so sign-ins are
attributable and one person can be removed without disrupting everyone else.

Usage:
    python3 make_admin_users.py alice bob carol

You are prompted for each password; nothing is echoed, stored on disk, or kept
in shell history. The script prints one line to paste into the Space's
Settings -> Variables and secrets as a SECRET named ADMIN_USERS_JSON.

To add someone later, rerun with the full list of users -- the secret holds
every account, so regenerate it whenever the roster changes. Passing an
existing ADMIN_USERS_JSON value on stdin merges the new users into it:

    pbpaste | python3 make_admin_users.py --merge dave
"""

from __future__ import annotations

import getpass
import json
import sys

from werkzeug.security import generate_password_hash

MIN_PASSWORD_LENGTH = 12
# scrypt is werkzeug's default but is missing from some Python builds (macOS
# system Python against LibreSSL). pbkdf2 hashes verify everywhere.
HASH_METHOD = "pbkdf2:sha256:600000"


def prompt_password(username: str) -> str:
    while True:
        password = getpass.getpass(f"Password for {username}: ")
        if len(password) < MIN_PASSWORD_LENGTH:
            print(f"  Too short - use at least {MIN_PASSWORD_LENGTH} characters.")
            continue
        if password != getpass.getpass(f"Confirm password for {username}: "):
            print("  Passwords did not match, try again.")
            continue
        return password


def main() -> None:
    args = [arg for arg in sys.argv[1:] if arg != "--merge"]
    merge = "--merge" in sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)

    users: dict[str, str] = {}
    if merge:
        raw = sys.stdin.read().strip()
        if raw:
            raw = raw.split("=", 1)[1] if raw.startswith("ADMIN_USERS_JSON=") else raw
            users = json.loads(raw)
        print(f"Merging into {len(users)} existing account(s).")

    for username in args:
        name = username.strip()
        if not name:
            continue
        if name in users:
            print(f"Replacing the password for existing user {name}.")
        users[name] = generate_password_hash(prompt_password(name), method=HASH_METHOD)

    print("\nAdd this as a SECRET (not a Variable) named ADMIN_USERS_JSON:\n")
    print(json.dumps(users, separators=(",", ":")))
    print(f"\n{len(users)} account(s). Password hashes only - no plaintext passwords are stored.")


if __name__ == "__main__":
    main()
