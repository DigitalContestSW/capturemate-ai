import time
import uuid
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from google.auth.transport import requests
from google.oauth2 import id_token

from app.config import settings


JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"


@dataclass(frozen=True)
class AuthenticatedUser:
    subject: str
    email: str | None = None
    name: str | None = None
    picture: str | None = None


def verify_google_id_token(token: str) -> dict[str, Any]:
    if not settings.google_web_client_id:
        raise HTTPException(status_code=500, detail="GOOGLE_WEB_CLIENT_ID is not configured")

    try:
        claims = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            settings.google_web_client_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="invalid Google ID token") from exc

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(status_code=401, detail="invalid Google ID token")
    return claims


def issue_token_pair(google_claims: dict[str, Any]) -> tuple[str, str]:
    subject = google_claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(status_code=401, detail="invalid Google ID token")

    base_claims = {
        "sub": subject,
        "email": google_claims.get("email"),
        "name": google_claims.get("name"),
        "picture": google_claims.get("picture"),
    }
    access_token = _encode_token(base_claims, ACCESS_TOKEN_TYPE, settings.jwt_access_ttl_seconds)
    refresh_token = _encode_token(base_claims, REFRESH_TOKEN_TYPE, settings.jwt_refresh_ttl_seconds)
    return access_token, refresh_token


def issue_access_token_from_refresh(refresh_token: str) -> str:
    user = decode_refresh_token(refresh_token)
    claims = {
        "sub": user.subject,
        "email": user.email,
        "name": user.name,
        "picture": user.picture,
    }
    return _encode_token(claims, ACCESS_TOKEN_TYPE, settings.jwt_access_ttl_seconds)


def decode_access_token(token: str) -> AuthenticatedUser:
    claims = _decode_token(token, ACCESS_TOKEN_TYPE, _access_secret())
    return _user_from_claims(claims)


def decode_refresh_token(token: str) -> AuthenticatedUser:
    claims = _decode_token(token, REFRESH_TOKEN_TYPE, _refresh_secret())
    return _user_from_claims(claims)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None,
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="missing bearer token")
    return decode_access_token(credentials.credentials)


def _encode_token(claims: dict[str, Any], token_type: str, ttl_seconds: int) -> str:
    now = int(time.time())
    payload = {
        **{key: value for key, value in claims.items() if value is not None},
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": uuid.uuid4().hex,
        "type": token_type,
    }
    secret = _access_secret() if token_type == ACCESS_TOKEN_TYPE else _refresh_secret()
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def _decode_token(token: str, expected_type: str, secret: str) -> dict[str, Any]:
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[JWT_ALGORITHM],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="invalid bearer token") from exc

    if claims.get("type") != expected_type:
        raise HTTPException(status_code=401, detail="invalid token type")
    return claims


def _user_from_claims(claims: dict[str, Any]) -> AuthenticatedUser:
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(status_code=401, detail="invalid bearer token")

    return AuthenticatedUser(
        subject=subject,
        email=_optional_string(claims.get("email")),
        name=_optional_string(claims.get("name")),
        picture=_optional_string(claims.get("picture")),
    )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _access_secret() -> str:
    if not settings.jwt_access_secret:
        raise HTTPException(status_code=500, detail="JWT_ACCESS_SECRET is not configured")
    return settings.jwt_access_secret


def _refresh_secret() -> str:
    if not settings.jwt_refresh_secret:
        raise HTTPException(status_code=500, detail="JWT_REFRESH_SECRET is not configured")
    return settings.jwt_refresh_secret
