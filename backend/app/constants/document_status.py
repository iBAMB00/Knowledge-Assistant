from enum import Enum

class DocumentStatus(str, Enum):
    """
    文档生命周期状态。

    用于管理文档从上传到完成处理的状态流转。
    """

    # 文件已经保存成功
    UPLOADED = "uploaded"

    # 正在解析原始文件
    PARSING = "parsing"

    # 文档解析完成，已有文本内容
    PARSED = "parsed"

    # 正在文本切片
    CHUNKING = "chunking"

    # 文档切片完成，已有文本切片
    CHUNKED = "chunked"

    # 正在生成向量
    EMBEDDING = "embedding"

    # 知识库处理完成
    COMPLETED = "completed"

    # 任意阶段失败
    FAILED = "failed"