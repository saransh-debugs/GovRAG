"""JWT verification -> Principal(tenant_id, user_id, roles).

Tenant and roles come only from a verified token, never from the request body.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt

from govrag.db import IDENT, TENANT
from govrag.errors import Unauthorized

# Opaque user IDs only. A user ID like "john.smith@acme.com" would put PII in every audit
# row and span, so emails are rejected here.
_SUB = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
MAX_ROLES = 32


@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: str
    user_id: str
    roles: frozenset[str]

    def has(self, role: str) -> bool:
        return role in self.roles


class JwtVerifier:
    def __init__(
        self,
        public_key_pem: bytes,
        *,
        issuer: str,
        audience: str,
        leeway_s: int = 30,
        max_lifetime_s: int = 3600,
    ) -> None:
        self._key = public_key_pem
        self._issuer = issuer
        self._audience = audience
        self._leeway = leeway_s
        self._max_lifetime = max_lifetime_s

    @classmethod
    def from_file(cls, path: Path, **kw: Any) -> JwtVerifier:
        return cls(path.read_bytes(), **kw)

    def verify(self, token: str) -> Principal:
        try:
            # algorithms is pinned. Taking the algorithm from the token header is how
            # `alg=none` and the RS256->HS256 swap (signing with the public key as an HMAC
            # secret) attacks work.
            claims = jwt.decode(
                token,
                self._key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp", "iat", "iss", "aud", "sub", "tid", "roles"]},
            )
        except jwt.ExpiredSignatureError as e:
            raise Unauthorized("token_expired") from e
        except jwt.PyJWTError as e:
            raise Unauthorized("token_invalid") from e

        now = time.time()
        iat, exp = claims["iat"], claims["exp"]
        if not isinstance(iat, int | float) or not isinstance(exp, int | float):
            raise Unauthorized("token_invalid")
        if iat > now + self._leeway:
            raise Unauthorized("token_not_yet_valid")
        # A stolen long-lived token is a long-lived breach. Revocation here is "wait for
        # exp", so exp has to be short.
        if exp - iat > self._max_lifetime:
            raise Unauthorized("token_lifetime_too_long")

        sub, tid, roles = claims["sub"], claims["tid"], claims["roles"]
        if not isinstance(sub, str) or not _SUB.match(sub):
            raise Unauthorized("bad_sub")
        if not isinstance(tid, str) or not TENANT.match(tid):
            raise Unauthorized("bad_tid")
        if (
            not isinstance(roles, list)
            or len(roles) > MAX_ROLES
            or not all(isinstance(r, str) and IDENT.match(r) for r in roles)
        ):
            raise Unauthorized("bad_roles")
        return Principal(tenant_id=tid, user_id=sub, roles=frozenset(roles))


def mint(
    private_key_pem: bytes,
    *,
    tenant_id: str,
    user_id: str,
    roles: list[str],
    issuer: str,
    audience: str,
    lifetime_s: int = 900,
    now: float | None = None,
) -> str:
    """Dev/test token issuer. In production the IdP does this."""
    t = int(now if now is not None else time.time())
    return jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "sub": user_id,
            "tid": tenant_id,
            "roles": roles,
            "iat": t,
            "exp": t + lifetime_s,
        },
        private_key_pem,
        algorithm="RS256",
    )
