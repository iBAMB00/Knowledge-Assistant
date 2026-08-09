from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    """用户注册请求。"""

    email: EmailStr
    password: str = Field(
        min_length=8,
        max_length=128,
    )
