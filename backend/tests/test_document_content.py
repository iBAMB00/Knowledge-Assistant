from app.models.database.document_content import DocumentContent
from app.repositories.document_contents_repository import DocumentContentRepository
from app.core.database import SessionLocal


def test_create_document_content():
    content = DocumentContent(
        document_id=1,
        content="测试文本",
        parser_type="test",
    )

    assert content.content == "测试文本"

def test_document_content_repository():
    """
    测试文档解析内容Repository。

    验证：
    1. 创建解析内容
    2. 根据文档ID查询
    """

    db = SessionLocal()

    try:
        repository = DocumentContentRepository()

        document_content = DocumentContent(
            document_id=999,
            content="测试解析后的文本内容",
            parser_type="test",
        )

        saved_content = repository.create(
            db=db,
            document_content=document_content,
        )

        result = repository.find_by_document_id(
            db=db,
            document_id=999,
        )

        assert saved_content.id is not None

        assert result is not None

        assert result.content == "测试解析后的文本内容"

        assert result.parser_type == "test"

    finally:
        db.close()