import os
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


def verify_basic_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
) -> str:
    """Verify credentials from environment variables."""
    expected_user = os.getenv("BASIC_AUTH_USER", "admin")
    expected_pass = os.getenv("BASIC_AUTH_PASS", "changeme")

    user_ok = secrets.compare_digest(credentials.username, expected_user)
    pass_ok = secrets.compare_digest(credentials.password, expected_pass)

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


def require_basic_auth(username: str = Depends(verify_basic_credentials)) -> str:
    """Use as a dependency on protected routes."""
    return username
