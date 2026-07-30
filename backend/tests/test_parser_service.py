import fitz
import pytest

from app.services.parser_service import (
    ParserService,
)


def test_parse_pdf_uses_visual_reading_order() -> None:
    """
    验证PDF文本按照页面坐标顺序提取，
    而不是按照PDF对象写入顺序提取。
    """

    document = fitz.open()
    page = document.new_page()

    # 故意先写页面下方文本。
    page.insert_text(
        (72, 144),
        "Second paragraph",
    )

    # 再写页面上方文本。
    page.insert_text(
        (72, 72),
        "First paragraph",
    )

    pdf_content = document.tobytes()
    document.close()

    result = ParserService().parse(
        filename="reading-order.pdf",
        content=pdf_content,
    )

    assert (
        result.content.index(
            "First paragraph"
        )
        < result.content.index(
            "Second paragraph"
        )
    )

    assert result.parser_type == "pymupdf"
    assert result.parser_version == "1.1.0"


def test_parse_pdf_rejects_empty_document() -> None:
    """
    验证无可提取文本的PDF不会进入后续流程。
    """

    document = fitz.open()
    document.new_page()

    pdf_content = document.tobytes()
    document.close()

    with pytest.raises(
        ValueError,
        match="no extractable text",
    ):
        ParserService().parse(
            filename="empty.pdf",
            content=pdf_content,
        )


def test_pdf_garbled_detection_rejects_cid_text() -> None:
    """
    验证PDF字符映射异常文本会被识别。
    """

    text = (
        "(cid:123)(cid:456)"
        "(cid:789)(cid:101)"
    )

    assert (
        ParserService._looks_garbled(text)
        is True
    )


def test_pdf_garbled_detection_accepts_normal_chinese() -> None:
    """
    验证正常中英文文本不会被误判为乱码。
    """

    text = (
        "企业知识库支持PDF文档解析，"
        "并通过向量检索辅助大模型回答问题。"
    )

    assert (
        ParserService._looks_garbled(text)
        is False
    )


def test_parse_txt_normalizes_line_breaks() -> None:
    """
    验证TXT解析统一换行并压缩多余空行。
    """

    result = ParserService().parse(
        filename="example.txt",
        content=(
            "第一行\r\n\r\n\r\n\r\n第二行"
            .encode("utf-8")
        ),
    )

    assert result.content == (
        "第一行\n\n第二行"
    )