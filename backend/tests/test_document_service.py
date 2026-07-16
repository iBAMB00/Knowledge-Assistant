from app.services.document_service import DocumentService
from pathlib import Path

def test_create_upload_dir():
    service = DocumentService()

    print(service.upload_dir)

    assert service.upload_dir.exists()

def test_upload_document(tmp_path: Path):
    """
    测试文档上传功能。
    """

    service = DocumentService(upload_dir=str(tmp_path))

    filename = "员工手册.pdf"
    content = b"Hello Secure Assistant"

    document = service.upload_document(
        filename=filename,
        content=content,
    )

    # 原始文件名
    assert document.filename == filename

    # 文件大小
    assert document.size == len(content)

    # 服务端文件存在
    stored_file = tmp_path / document.stored_name

    assert stored_file.exists()

    # 文件内容一致
    assert stored_file.read_bytes() == content