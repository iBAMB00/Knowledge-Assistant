from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentInfo:
    """
    文档基础信息。

    用于描述文档上传后的元数据，
    不包含文档正文和解析后的内容。
    """

    filename: str
    stored_name: str
    path: str
    size: int