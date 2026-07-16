from app.services.document_service import DocumentService


def test_create_upload_dir():
    service = DocumentService()

    print(service.upload_dir)

    assert service.upload_dir.exists()