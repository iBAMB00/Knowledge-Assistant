from dataclasses import dataclass


@dataclass(frozen=True)
class ParseResult:
    """
    文档解析结果。

    用于在解析服务和业务服务之间传递解析后的文本，以及解析器类型和版本。
    不包含数据库模型和存储位置信息。
    """

    content: str
    parser_type: str
    parser_version: str