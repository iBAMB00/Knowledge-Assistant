from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.processing_job_status import (
    ProcessingJobStatus,
)
from app.constants.processing_job_type import (
    ProcessingJobType,
)
from app.models.database.document import Document
from app.models.database.processing_job import ProcessingJob


def test_processing_job_defaults_and_persistence(
    db: Session,
) -> None:
    """
    验证任务默认状态和基础持久化。
    """

    document = Document(
        filename="processing-job-test.txt",
        stored_name="processing-job-test-stored.txt",
        path=(
            "tests/uploads/"
            "processing-job-test-stored.txt"
        ),
        size=100,
        status=DocumentStatus.UPLOADED.value,
    )

    db.add(document)
    db.flush()

    job = ProcessingJob(
        document_id=document.id,
        job_type=(
            ProcessingJobType
            .DOCUMENT_PROCESSING
            .value
        ),
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    assert job.id is not None
    assert job.document_id == document.id

    assert (
        job.job_type
        == ProcessingJobType
        .DOCUMENT_PROCESSING
        .value
    )

    assert (
        job.status
        == ProcessingJobStatus.PENDING.value
    )

    assert job.progress == 0
    assert job.error_message is None
    assert job.started_at is None
    assert job.finished_at is None
    assert job.created_at is not None
    assert job.updated_at is not None