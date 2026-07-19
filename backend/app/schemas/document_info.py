from dataclasses import dataclass

from app.constants.document_status import DocumentStatus


@dataclass(frozen=True)
class DocumentInfo:
    """
    文档上传结果。

    用于向客户端返回新建文档的公开信息，
    不包含存储文件名、服务器路径和解析内容。
    """

    id: int
    filename: str
    size: int
    status: DocumentStatus