from enum import Enum


class UserRole(str, Enum):
    """系统内置的最小用户角色。"""

    USER = "user"
    ADMIN = "admin"
