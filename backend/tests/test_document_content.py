from app.models.database.document_content import DocumentContent


def test_create_document_content():
    content = DocumentContent(
        document_id=1,
        content="测试文本",
        parser_type="test",
    )

    assert content.content == "测试文本"