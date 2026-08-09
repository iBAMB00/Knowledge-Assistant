from pydantic import BaseModel


class TokenResponse(BaseModel):
    """JWT 登录响应。"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
