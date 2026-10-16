"""JWT attacks. Every one must end in Unauthorized, never in a Principal."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from govrag.errors import Unauthorized
from tests.conftest import AUD, ISS, JwtKit


def _b64(d: dict[str, object] | bytes) -> str:
    raw = d if isinstance(d, bytes) else json.dumps(d, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _claims(**over: object) -> dict[str, object]:
    now = int(time.time())
    c: dict[str, object] = {
        "iss": ISS, "aud": AUD, "sub": "u1", "tid": "acme", "roles": ["intern"],
        "iat": now, "exp": now + 600,
    }
    c.update(over)
    return c


def test_valid_token(jwtkit: JwtKit) -> None:
    p = jwtkit.verifier.verify(jwtkit.token(roles=["hr", "intern"]))
    assert (p.tenant_id, p.user_id, p.roles) == ("acme", "u1", frozenset({"hr", "intern"}))


def test_alg_none(jwtkit: JwtKit) -> None:
    tok = f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64(_claims(roles=['admin']))}."
    with pytest.raises(Unauthorized):
        jwtkit.verifier.verify(tok)


def test_hs256_with_public_key_as_secret(jwtkit: JwtKit) -> None:
    # The classic RS->HS confusion: the attacker knows the public key and uses it as an
    # HMAC secret. A verifier that trusts the header's alg would accept this.
    head = _b64({"alg": "HS256", "typ": "JWT"})
    body = _b64(_claims(roles=["admin"]))
    sig = hmac.new(jwtkit.public_pem, f"{head}.{body}".encode(), hashlib.sha256).digest()
    with pytest.raises(Unauthorized):
        jwtkit.verifier.verify(f"{head}.{body}.{_b64(sig)}")


def test_tampered_roles_keep_signature(jwtkit: JwtKit) -> None:
    head, _, sig = jwtkit.token(roles=["intern"]).split(".")
    with pytest.raises(Unauthorized):
        jwtkit.verifier.verify(f"{head}.{_b64(_claims(roles=['hr']))}.{sig}")


def test_tampered_tenant_keep_signature(jwtkit: JwtKit) -> None:
    head, _, sig = jwtkit.token().split(".")
    with pytest.raises(Unauthorized):
        jwtkit.verifier.verify(f"{head}.{_b64(_claims(tid='globex'))}.{sig}")


def test_signed_by_other_key(jwtkit: JwtKit) -> None:
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    with pytest.raises(Unauthorized):
        jwtkit.verifier.verify(jwt.encode(_claims(), other, algorithm="RS256"))


@pytest.mark.parametrize(
    ("over", "code"),
    [
        ({"exp": int(time.time()) - 3600}, "token_expired"),
        ({"aud": "someone-else"}, "token_invalid"),
        ({"iss": "https://evil"}, "token_invalid"),
        ({"iat": int(time.time()) + 3600, "exp": int(time.time()) + 4000}, None),
        ({"exp": int(time.time()) + 86_400}, "token_lifetime_too_long"),
        ({"sub": "john.smith@acme.com"}, "bad_sub"),
        ({"tid": "ACME"}, "bad_tid"),
        ({"tid": "acme' OR 1=1"}, "bad_tid"),
        ({"roles": "hr"}, "bad_roles"),
        ({"roles": ["hr}", "x"]}, "bad_roles"),
        ({"roles": [f"r{i}" for i in range(33)]}, "bad_roles"),
    ],
)
def test_bad_claims(jwtkit: JwtKit, over: dict[str, object], code: str | None) -> None:
    tok = jwt.encode(_claims(**over), jwtkit.private_pem, algorithm="RS256")
    with pytest.raises(Unauthorized) as e:
        jwtkit.verifier.verify(tok)
    if code:
        assert e.value.code == code


@pytest.mark.parametrize("missing", ["exp", "iat", "sub", "tid", "roles", "aud", "iss"])
def test_missing_claim(jwtkit: JwtKit, missing: str) -> None:
    c = _claims()
    del c[missing]
    with pytest.raises(Unauthorized):
        jwtkit.verifier.verify(jwt.encode(c, jwtkit.private_pem, algorithm="RS256"))


@pytest.mark.parametrize("junk", ["", "abc", "a.b.c", "Bearer x", "." * 3])
def test_garbage(jwtkit: JwtKit, junk: str) -> None:
    with pytest.raises(Unauthorized):
        jwtkit.verifier.verify(junk)
