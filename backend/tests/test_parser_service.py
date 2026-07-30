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
    assert result.parser_version == "1.2.0"


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

def test_garbled_detection_rejects_broken_font_mapping() -> None:
    """
    验证由错误PDF字体映射产生的多文字脚本乱码。
    """

    garbled_text = (
        "Java चᏐ໐ஞ௛ᕮ\n"
        "Ջԍฎ JavaҘ\n"
        "Java ጱᇙᅩ\n"
        "Java ୏ݎሾह\n"
        "ᦢᳯഴګ๦ᴴ"
    )

    assert (
        ParserService._looks_garbled(
            garbled_text
        )
        is True
    )

def test_parse_pdf_falls_back_to_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证文字层异常时会使用OCR结果。
    """

    document = fitz.open()
    page = document.new_page()

    page.insert_text(
        (72, 72),
        "extractable but mapped incorrectly",
    )

    pdf_content = document.tobytes()
    document.close()

    monkeypatch.setattr(
        ParserService,
        "_looks_garbled",
        classmethod(
            lambda cls, text: True
        ),
    )

    monkeypatch.setattr(
        ParserService,
        "_ocr_pdf_page",
        lambda self, page, page_number: (
            "Java 是一门面向对象的编程语言。"
        ),
    )

    result = ParserService().parse(
        filename="garbled.pdf",
        content=pdf_content,
    )

    assert result.content == (
        "Java 是一门面向对象的编程语言。"
    )
    assert result.parser_type == (
        "pymupdf_ocr"
    )
    assert result.parser_version == "1.2.0"