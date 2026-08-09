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

def test_parse_markdown_preserves_heading_structure() -> None:
    """验证 Markdown Heading 会生成统一章节层级与全文偏移。"""

    source = (
        "项目简介\n\n"
        "# 部署指南\n"
        "部署前先检查环境。\n\n"
        "## 数据库\n"
        "数据库使用 PostgreSQL。\n\n"
        "# 故障处理\n"
        "先查看日志。"
    )

    result = ParserService().parse(
        filename="guide.md",
        content=source.encode("utf-8"),
    )

    assert result.parser_type == "markdown"
    assert result.parser_version == "1.1.0"
    assert result.source_format == "markdown"

    assert [section.title for section in result.sections] == [
        None,
        "部署指南",
        "数据库",
        "故障处理",
    ]
    assert result.sections[2].heading_path == (
        "部署指南",
        "数据库",
    )
    assert result.sections[3].heading_path == (
        "故障处理",
    )

    for section in result.sections:
        section_text = result.content[
            section.start_offset:section.end_offset
        ]
        assert section_text.strip()


def test_parse_markdown_ignores_headings_inside_code_fence() -> None:
    """验证 fenced code block 内的 # 不会被误识别为章节。"""

    source = (
        "# API 示例\n\n"
        "```python\n"
        "# 这只是代码注释\n"
        "print('hello')\n"
        "```\n\n"
        "## 返回值\n"
        "返回 JSON。"
    )

    result = ParserService().parse(
        filename="api.markdown",
        content=source.encode("utf-8"),
    )

    assert [section.title for section in result.sections] == [
        "API 示例",
        "返回值",
    ]


def test_parse_html_extracts_semantic_text_and_sections() -> None:
    """验证 HTML 保留标题、正文和列表，并过滤脚本样式。"""

    source = """
    <html>
      <head>
        <style>.hidden { color: red; }</style>
        <script>window.secret = 'ignore-me';</script>
      </head>
      <body>
        <h1>部署指南</h1>
        <p>先准备 PostgreSQL。</p>
        <h2>检查项</h2>
        <ul>
          <li>检查 Redis</li>
          <li>检查 Qdrant</li>
        </ul>
      </body>
    </html>
    """

    result = ParserService().parse(
        filename="guide.html",
        content=source.encode("utf-8"),
    )

    assert result.parser_type == "html"
    assert result.parser_version == "1.1.0"
    assert result.source_format == "html"
    assert "# 部署指南" in result.content
    assert "## 检查项" in result.content
    assert "检查 Redis" in result.content
    assert "ignore-me" not in result.content
    assert "color: red" not in result.content
    assert [section.title for section in result.sections] == [
        "部署指南",
        "检查项",
    ]
    assert result.sections[1].heading_path == (
        "部署指南",
        "检查项",
    )


def test_parse_result_builds_compact_structure_metadata() -> None:
    """验证持久化结构只保存索引，不复制章节正文。"""

    result = ParserService().parse(
        filename="guide.md",
        content=(
            "# 标题\n正文内容\n## 子标题\n更多正文"
        ).encode("utf-8"),
    )

    metadata = result.to_structure_metadata()

    assert metadata is not None
    assert metadata["version"] == "1.1"
    assert metadata["source_format"] == "markdown"
    assert metadata["sections"][0]["title"] == "标题"
    assert "content" not in metadata["sections"][0]



def test_parse_markdown_extracts_code_and_table_blocks() -> None:
    """验证 Markdown 代码块与表格会生成紧凑 Block 索引。"""

    source = (
        "# API 示例\n\n"
        "```python\n"
        "# 代码注释，不是 Heading\n"
        "print('hello')\n"
        "```\n\n"
        "| 字段 | 含义 |\n"
        "| --- | --- |\n"
        "| status | 状态码 |"
    )

    result = ParserService().parse(
        filename="api.md",
        content=source.encode("utf-8"),
    )

    assert [block.block_type for block in result.blocks] == [
        "code",
        "table",
    ]

    code_block = result.blocks[0]
    table_block = result.blocks[1]

    assert code_block.language == "python"
    assert code_block.section_index == 0
    assert "print('hello')" in result.content[
        code_block.start_offset:code_block.end_offset
    ]

    assert table_block.section_index == 0
    assert table_block.row_count == 2
    assert table_block.column_count == 2
    assert table_block.has_header is True
    assert "| status | 状态码 |" in result.content[
        table_block.start_offset:table_block.end_offset
    ]


def test_parse_markdown_does_not_treat_table_inside_code_as_table() -> None:
    """验证 fenced code 内的 pipe table 文本不会被重复识别为表格。"""

    source = (
        "# 示例\n\n"
        "```text\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "```"
    )

    result = ParserService().parse(
        filename="example.md",
        content=source.encode("utf-8"),
    )

    assert len(result.blocks) == 1
    assert result.blocks[0].block_type == "code"


def test_parse_html_extracts_pre_and_table_blocks() -> None:
    """验证 HTML 的 pre/code 与 table 能映射为统一 Block 索引。"""

    source = """
    <html>
      <body>
        <h1>接口说明</h1>
        <pre><code>curl /health
# comment</code></pre>
        <table>
          <tr><th>字段</th><th>含义</th></tr>
          <tr><td>status</td><td>状态码</td></tr>
        </table>
      </body>
    </html>
    """

    result = ParserService().parse(
        filename="api.html",
        content=source.encode("utf-8"),
    )

    assert [block.block_type for block in result.blocks] == [
        "code",
        "table",
    ]
    assert all(block.section_index == 0 for block in result.blocks)

    code_block = result.blocks[0]
    table_block = result.blocks[1]

    assert "curl /health" in result.content[
        code_block.start_offset:code_block.end_offset
    ]
    assert table_block.row_count == 2
    assert table_block.column_count == 2
    assert "status" in result.content[
        table_block.start_offset:table_block.end_offset
    ]


def test_structure_metadata_persists_blocks_without_copying_block_content() -> None:
    """验证结构元数据保存 Block 索引与属性，但不复制正文。"""

    result = ParserService().parse(
        filename="guide.md",
        content=(
            "# 示例\n\n"
            "```json\n"
            '{"ok": true}\n'
            "```\n\n"
            "| key | value |\n"
            "| --- | --- |\n"
            "| ok | true |"
        ).encode("utf-8"),
    )

    metadata = result.to_structure_metadata()

    assert metadata is not None
    assert metadata["version"] == "1.1"
    assert [block["block_type"] for block in metadata["blocks"]] == [
        "code",
        "table",
    ]
    assert metadata["blocks"][0]["language"] == "json"
    assert metadata["blocks"][1]["row_count"] == 2
    assert all(
        "content" not in block
        for block in metadata["blocks"]
    )
