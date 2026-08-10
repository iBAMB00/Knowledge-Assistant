from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.core.database import get_db
from app.models.database.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.processing_job_response import ProcessingJobResponse
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)
from app.services.processing_job_service import (
    ProcessingJobNotFoundError,
    ProcessingJobService,
)


router = APIRouter(
    tags=["Processing Jobs"],
)


document_repository = DocumentRepository()
processing_job_service = ProcessingJobService(
    document_repository=document_repository,
    processing_job_repository=ProcessingJobRepository(),
)
knowledge_base_access_policy = KnowledgeBaseAccessPolicy(
    knowledge_base_repository=KnowledgeBaseRepository(),
    document_repository=document_repository,
)


@router.get(
    "/processing-jobs/{job_id}",
    response_model=ProcessingJobResponse,
)
def get_processing_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ProcessingJobResponse:
    """
    根据任务ID查询处理状态。
    """

    try:
        job = processing_job_service.get_job(
            db=db,
            job_id=job_id,
        )
        knowledge_base_access_policy.get_accessible_document(
            db=db,
            document_id=job.document_id,
            user=current_user,
        )
        return job

    except ResourceAccessNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

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
    "/documents/{document_id}/processing-jobs",
    response_model=list[ProcessingJobResponse],
)
def list_document_processing_jobs(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ProcessingJobResponse]:
    """
    查询文档全部处理任务。

    按最新任务优先排列；
    文档存在但没有任务时返回空列表。
    """

    try:
        knowledge_base_access_policy.get_accessible_document(
            db=db, document_id=document_id, user=current_user
        )
        return processing_job_service.list_document_jobs(
            db=db,
            document_id=document_id,
        )

    except ResourceAccessNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

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
            detail="任务列表查询失败",
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
    current_user: User = Depends(get_current_user),
) -> ProcessingJobResponse:
    """
    查询文档最近一次处理任务。
    """

    try:
        knowledge_base_access_policy.get_accessible_document(
            db=db, document_id=document_id, user=current_user
        )
        return (
            processing_job_service
            .get_latest_document_job(
                db=db,
                document_id=document_id,
            )
        )

    except ResourceAccessNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

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