"""
Week 10: verifies Clerk-issued session JWTs so the public API resolves a
real, provable identity instead of trusting a client-supplied username --
the old as_user request field let any caller simply claim to be
bob_admin (compliance_admin) with zero proof, which is fine for a CLI
trusted local operator but not for a public HTTP endpoint.

CLERK_ISSUER is the Clerk instance's Frontend API URL, e.g.
"https://your-app.clerk.accounts.dev" (Clerk dashboard -> API Keys). The
JWKS used to verify the RS256 signature lives at
{CLERK_ISSUER}/.well-known/jwks.json -- no Clerk secret key is needed here,
since JWKS verification only uses Clerk's public keys.
"""

import os

import jwt
from jwt import PyJWKClient

CLERK_ISSUER = os.environ.get("CLERK_ISSUER")

_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        if not CLERK_ISSUER:
            raise RuntimeError("CLERK_ISSUER is not configured")
        _jwks_client = PyJWKClient(f"{CLERK_ISSUER}/.well-known/jwks.json")
    return _jwks_client


def verify_clerk_token(token: str) -> str:
    """Returns the Clerk user ID (the `sub` claim) for a valid, unexpired,
    correctly-signed session token issued by CLERK_ISSUER. Raises
    jwt.PyJWTError (or a subclass) otherwise -- callers should treat any
    exception here as "not authenticated," not retry it."""
    signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        issuer=CLERK_ISSUER,
        options={"verify_aud": False},  # Clerk session tokens don't set aud by default
    )
    return claims["sub"]
