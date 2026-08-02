from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.schemas.processing_job_response import ProcessingJobResponse
from app.services.processing_job_service import (
    ProcessingJobNotFoundError,
    ProcessingJobService,
)


router = APIRouter(
    tags=["Processing Jobs"],
)


processing_job_service = ProcessingJobService(
    document_repository=DocumentRepository(),
    processing_job_repository=(
        ProcessingJobRepository()
    ),
)


@router.get(
    "/processing-jobs/{job_id}",
    response_model=ProcessingJobResponse,
)
def get_processing_job(
    job_id: int,
    db: Session = Depends(get_db),
) -> ProcessingJobResponse:
    """
    根据任务ID查询处理状态。
    """

    try:
        return processing_job_service.get_job(
            db=db,
            job_id=job_id,
        )

    except ProcessingJobNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="任务查询失败",
        ) from exc


@router.get(
    (
        "/documents/{document_id}"
        "/processing-jobs/latest"
    ),
    response_model=ProcessingJobResponse,
)
def get_latest_document_processing_job(
    document_id: int,
    db: Session = Depends(get_db),
) -> ProcessingJobResponse:
    """
    查询文档最近一次处理任务。
    """

    try:
        return (
            processing_job_service
            .get_latest_document_job(
                db=db,
                document_id=document_id,
            )
        )

    except ProcessingJobNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        detail = str(exc)

        if detail == "document not found":
            status_code = 404
        else:
            status_code = 400

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="任务查询失败",
        ) from exc