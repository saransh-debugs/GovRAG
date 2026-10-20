"""Mint a dev JWT signed with keys/jwt_private.pem (made by scripts/dev_setup.py).

    python -m scripts.mint_jwt --tenant acme --user u-hr-1 --roles hr
"""

from __future__ import annotations

import argparse
from pathlib import Path

from govrag.auth import mint
from govrag.config import get_settings

ROOT = Path(__file__).resolve().parent.parent


def token(tenant: str, user: str, roles: list[str], lifetime_s: int = 900) -> str:
    s = get_settings()
    return mint((ROOT / "keys" / "jwt_private.pem").read_bytes(), tenant_id=tenant,
                user_id=user, roles=roles, issuer=s.jwt_issuer, audience=s.jwt_audience,
                lifetime_s=lifetime_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--roles", required=True, help="comma-separated")
    ap.add_argument("--ttl", type=int, default=900)
    a = ap.parse_args()
    print(token(a.tenant, a.user, [r for r in a.roles.split(",") if r], a.ttl))


if __name__ == "__main__":
    main()
