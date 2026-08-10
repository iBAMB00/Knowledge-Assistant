from sqlalchemy.orm import Session

from app.models.database.user import User


class UserRepository:
    """用户数据访问层，不负责事务提交。"""

    def create(
        self,
        db: Session,
        user: User,
    ) -> User:
        """新增用户并 flush，事务由 Service 控制。"""
        db.add(user)
        db.flush()
        return user

    def find_by_id(
        self,
        db: Session,
        user_id: int,
    ) -> User | None:
        """根据用户 ID 查询用户。"""
        return db.get(User, user_id)

    def find_by_email(
        self,
        db: Session,
        email: str,
    ) -> User | None:
        """根据规范化邮箱查询用户。"""
        return (
            db.query(User)
            .filter(User.email == email)
            .one_or_none()
        )
