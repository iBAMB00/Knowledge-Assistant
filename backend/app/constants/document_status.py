from enum import Enum


class DocumentStatus(str, Enum):
    """
    文档知识加工生命周期状态。

    状态按照解析、切片、向量化三个阶段流转。
    每个阶段使用独立失败状态，便于定位问题和执行重试。
    """

    # 文件已经上传并保存
    UPLOADED = "uploaded"

    # 正在解析原始文件
    PARSING = "parsing"

    # 文档解析完成，文本内容已保存
    PARSED = "parsed"

    # 正在生成文本切片
    CHUNKING = "chunking"

    # 文档切片完成，Chunk已保存
    CHUNKED = "chunked"

    # 正在为Chunk生成向量
    EMBEDDING = "embedding"

    # 所有知识加工步骤完成，可以进入检索
    COMPLETED = "completed"

    # 文档解析失败
    PARSE_FAILED = "parse_failed"

    # 文档切片失败
    CHUNK_FAILED = "chunk_failed"

    # 文档向量化失败
    EMBEDDING_FAILED = "embedding_failed"