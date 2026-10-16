"""Create local dev secrets: .env (DB passwords, master key) and an RSA keypair for JWTs.

Both are gitignored. Re-running never overwrites existing secrets.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    keys = ROOT / "keys"
    keys.mkdir(exist_ok=True)
    priv_path, pub_path = keys / "jwt_private.pem", keys / "jwt_public.pem"
    if not priv_path.exists():
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        priv_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        priv_path.chmod(0o600)
        pub_path.write_bytes(
            key.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )
        print(f"wrote {priv_path.relative_to(ROOT)}, {pub_path.relative_to(ROOT)}")

    env = ROOT / ".env"
    if not env.exists():
        env.write_text(
            "\n".join(
                [
                    "GOVRAG_DB_OWNER_PASSWORD=owner_dev_pw",  # matches docker-compose.yml
                    f"GOVRAG_DB_APP_PASSWORD={secrets.token_urlsafe(24)}",
                    f"GOVRAG_DB_VAULT_PASSWORD={secrets.token_urlsafe(24)}",
                    f"GOVRAG_DB_INGEST_PASSWORD={secrets.token_urlsafe(24)}",
                    f"GOVRAG_MASTER_KEYS={Fernet.generate_key().decode()}",
                    "GOVRAG_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces",
                    "",
                ]
            )
        )
        env.chmod(0o600)
        print("wrote .env")


if __name__ == "__main__":
    main()
