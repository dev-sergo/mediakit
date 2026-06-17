from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mediakit.config import settings

_bearer = HTTPBearer(auto_error=False)


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
) -> None:
    if not settings.api_token:
        return  # auth disabled when token not configured
    if credentials is None or credentials.credentials != settings.api_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
