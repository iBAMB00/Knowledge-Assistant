from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.constants.user_role import UserRole


@dataclass(frozen=True)
class ToolAccessPrincipal:
    """Tool 权限校验使用的最小可信主体。"""

    id: int
    role: str


class ToolExecutionContext(BaseModel):
    """
    Tool 执行时由服务端注入的可信上下文。

    这些字段不属于 LLM 可生成的 Tool 参数，
    Tool 必须使用该上下文完成资源隔离和请求关联。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    user_id: int = Field(strict=True, gt=0)
    role: UserRole
    knowledge_base_id: int = Field(strict=True, gt=0)
    request_id: str = Field(min_length=1, max_length=128)
    agent_run_id: int | str | None = None

    def to_access_principal(self) -> ToolAccessPrincipal:
        """转换为权限策略真正需要的最小身份信息。"""

        return ToolAccessPrincipal(
            id=self.user_id,
            role=self.role.value,
        )
