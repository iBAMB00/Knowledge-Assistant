from pydantic import BaseModel, EmailStr, Field


class UserLoginRequest(BaseModel):
    """用户登录请求。"""

    email: EmailStr
    password: str = Field(
        min_length=1,
        max_length=128,
    )
