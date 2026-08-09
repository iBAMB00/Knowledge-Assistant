import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

import fitz

from app.schemas.parse_result import ParseResult, ParsedSection


class _HTMLTextExtractor(HTMLParser):
    """
    将 HTML 转换为适合知识库处理的轻量文本。

    不尝试还原浏览器布局，只保留标题、正文、列表、代码块等
    对 RAG 结构有直接价值的语义信息。
    """

    HEADING_LEVELS = {
        "h1": 1,
        "h2": 2,
        "h3": 3,
        "h4": 4,
        "h5": 5,
        "h6": 6,
    }
    IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
    }
    BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "aside",
        "nav",
        "ul",
        "ol",
        "blockquote",
        "table",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0
        self._pre_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs,
    ) -> None:
        normalized_tag = tag.lower()

        if normalized_tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
            return

        if self._ignored_depth:
            return

        heading_level = self.HEADING_LEVELS.get(
            normalized_tag
        )

        if heading_level is not None:
            self._parts.append("\n\n")
            self._parts.append(
                "#" * heading_level + " "
            )
            return

        if normalized_tag == "br":
            self._parts.append("\n")
            return

        if normalized_tag == "li":
            self._parts.append("\n- ")
            return

        if normalized_tag == "pre":
            self._parts.append("\n\n```\n")
            self._pre_depth += 1
            return

        if normalized_tag in {"th", "td"}:
            self._parts.append(" | ")
            return

        if normalized_tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()

        if normalized_tag in self.IGNORED_TAGS:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return

        if self._ignored_depth:
            return

        if normalized_tag == "pre":
            if self._pre_depth:
                self._pre_depth -= 1
            self._parts.append("\n```\n\n")
            return

        if (
            normalized_tag in self.HEADING_LEVELS
            or normalized_tag in self.BLOCK_TAGS
            or normalized_tag == "li"
        ):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth or not data:
            return

        if self._pre_depth:
            self._parts.append(data)
            return

        normalized_data = re.sub(
            r"\s+",
            " ",
            data,
        )

        if normalized_data.strip():
            self._parts.append(normalized_data)

    def render(self) -> str:
        """返回 HTML 转换后的文本。"""

        return "".join(self._parts)


class ParserService:
    """
    文档解析服务。

    负责从文件二进制内容中提取文本和基础结构。
    不负责读取存储系统，也不依赖本地文件路径。

    当前支持：
    - PDF
    - TXT
    - Markdown
    - HTML
    """

    PDF_PARSER_VERSION = "1.2.0"
    TXT_PARSER_VERSION = "1.1.0"
    MARKDOWN_PARSER_VERSION = "1.0.0"
    HTML_PARSER_VERSION = "1.0.0"

    PDF_OCR_LANGUAGE = "chi_sim+eng"
    PDF_OCR_DPI = 200

    MAX_SUSPICIOUS_CHARACTER_RATIO = 0.02
    MAX_UNEXPECTED_SCRIPT_RATIO = 0.15

    CONTROL_CHARACTER_PATTERN = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
    )

    CID_CHARACTER_PATTERN = re.compile(
        r"\(cid:\d+\)",
        re.IGNORECASE,
    )

    MARKDOWN_HEADING_PATTERN = re.compile(
        r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$"
    )

    FENCE_PATTERN = re.compile(
        r"^[ \t]*(```+|~~~+)"
    )

    def parse(
        self,
        filename: str,
        content: bytes,
    ) -> ParseResult:
        """解析文档二进制内容。"""

        cleaned_filename = filename.strip()

        if not cleaned_filename:
            raise ValueError("filename cannot be empty")

        if not content:
            raise ValueError("file content cannot be empty")

        suffix = Path(cleaned_filename).suffix.lower()

        if suffix == ".pdf":
            parsed_text, used_ocr = self._parse_pdf(
                content
            )

            return ParseResult(
                content=parsed_text,
                parser_type=(
                    "pymupdf_ocr"
                    if used_ocr
                    else "pymupdf"
                ),
                parser_version=self.PDF_PARSER_VERSION,
                source_format="pdf",
            )

        if suffix == ".txt":
            return ParseResult(
                content=self._parse_txt(content),
                parser_type="plain_text",
                parser_version=self.TXT_PARSER_VERSION,
                source_format="txt",
            )

        if suffix in {".md", ".markdown"}:
            parsed_text = self._parse_markdown(content)
            return ParseResult(
                content=parsed_text,
                parser_type="markdown",
                parser_version=(
                    self.MARKDOWN_PARSER_VERSION
                ),
                source_format="markdown",
                sections=self._extract_sections(
                    parsed_text
                ),
            )

        if suffix in {".html", ".htm"}:
            parsed_text = self._parse_html(content)
            return ParseResult(
                content=parsed_text,
                parser_type="html",
                parser_version=self.HTML_PARSER_VERSION,
                source_format="html",
                sections=self._extract_sections(
                    parsed_text
                ),
            )

        raise ValueError(
            f"unsupported file type: {suffix or 'unknown'}"
        )

    def _parse_pdf(
        self,
        content: bytes,
    ) -> tuple[str, bool]:
        """
        从PDF二进制内容中提取文本。

        优先使用PDF文本层；页面为空或文字层乱码时，降级为整页OCR。
        """

        try:
            page_texts: list[str] = []
            used_ocr = False

            with fitz.open(
                stream=content,
                filetype="pdf",
            ) as document:
                if document.needs_pass:
                    raise ValueError(
                        "encrypted pdf is not supported"
                    )

                for page_number, page in enumerate(
                    document,
                    start=1,
                ):
                    page_text = page.get_text(
                        "text",
                        sort=True,
                    )

                    normalized_page_text = (
                        self._normalize_text(
                            page_text
                        )
                    )

                    if not normalized_page_text:
                        has_images = bool(
                            page.get_images(full=True)
                        )

                        if not has_images:
                            continue

                        normalized_page_text = (
                            self._ocr_pdf_page(
                                page=page,
                                page_number=page_number,
                            )
                        )
                        used_ocr = True

                    elif self._looks_garbled(
                        normalized_page_text
                    ):
                        normalized_page_text = (
                            self._ocr_pdf_page(
                                page=page,
                                page_number=page_number,
                            )
                        )
                        used_ocr = True

                    if normalized_page_text:
                        page_texts.append(
                            normalized_page_text
                        )

            parsed_text = "\n\n".join(
                page_texts
            ).strip()

            if not parsed_text:
                raise ValueError(
                    "pdf contains no extractable text"
                )

            return parsed_text, used_ocr

        except ValueError:
            raise

        except Exception as exc:
            raise ValueError(
                "failed to parse pdf document"
            ) from exc

    def _parse_txt(
        self,
        content: bytes,
    ) -> str:
        """从TXT二进制内容中读取并清理文本。"""

        parsed_text = self._decode_utf8(
            content=content,
            document_type="txt",
        )

        normalized_text = self._normalize_text(
            parsed_text
        )

        if not normalized_text:
            raise ValueError(
                "txt document contains no text"
            )

        return normalized_text

    def _parse_markdown(
        self,
        content: bytes,
    ) -> str:
        """
        解析 Markdown。

        Markdown 本身已经是结构化纯文本，因此保留原始标记，
        只做统一编码和文本清理；Heading 在后续生成结构索引。
        """

        decoded_text = self._decode_utf8(
            content=content,
            document_type="markdown",
        )
        normalized_text = self._normalize_text(
            decoded_text
        )

        if not normalized_text:
            raise ValueError(
                "markdown document contains no text"
            )

        return normalized_text

    def _parse_html(
        self,
        content: bytes,
    ) -> str:
        """
        将 HTML 转换为适合检索的轻量 Markdown 风格文本。

        script/style/noscript 不进入知识正文；H1-H6 转换为对应的
        Markdown Heading，方便与 Markdown 共用统一章节模型。
        """

        decoded_html = self._decode_utf8(
            content=content,
            document_type="html",
        )

        try:
            extractor = _HTMLTextExtractor()
            extractor.feed(decoded_html)
            extractor.close()
        except Exception as exc:
            raise ValueError(
                "failed to parse html document"
            ) from exc

        normalized_text = self._normalize_text(
            extractor.render()
        )

        if not normalized_text:
            raise ValueError(
                "html document contains no text"
            )

        return normalized_text

    @staticmethod
    def _decode_utf8(
        content: bytes,
        document_type: str,
    ) -> str:
        """以 UTF-8 / UTF-8 BOM 解码文本类文档。"""

        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{document_type} document must use utf-8 encoding"
            ) from exc

    @classmethod
    def _extract_sections(
        cls,
        content: str,
    ) -> tuple[ParsedSection, ...]:
        """
        从 Markdown 风格 Heading 中生成统一章节索引。

        Markdown fenced code block 内的 ``#`` 不会被误识别为标题。
        每个章节只保存全文偏移和 heading_path，不复制章节正文。
        """

        headings: list[tuple[int, int, str]] = []
        current_offset = 0
        active_fence: str | None = None

        for line in content.splitlines(keepends=True):
            line_without_newline = line.rstrip("\r\n")
            fence_match = cls.FENCE_PATTERN.match(
                line_without_newline
            )

            if fence_match:
                fence_token = fence_match.group(1)
                fence_marker = fence_token[0]

                if active_fence is None:
                    active_fence = fence_marker
                elif active_fence == fence_marker:
                    active_fence = None

                current_offset += len(line)
                continue

            if active_fence is None:
                heading_match = (
                    cls.MARKDOWN_HEADING_PATTERN.match(
                        line_without_newline
                    )
                )

                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()
                    if title:
                        headings.append(
                            (
                                current_offset,
                                level,
                                title,
                            )
                        )

            current_offset += len(line)

        sections: list[ParsedSection] = []
        next_section_index = 0

        if not headings:
            return (
                ParsedSection(
                    section_index=0,
                    title=None,
                    level=0,
                    heading_path=(),
                    start_offset=0,
                    end_offset=len(content),
                ),
            )

        first_heading_offset = headings[0][0]
        if content[:first_heading_offset].strip():
            sections.append(
                ParsedSection(
                    section_index=next_section_index,
                    title=None,
                    level=0,
                    heading_path=(),
                    start_offset=0,
                    end_offset=first_heading_offset,
                )
            )
            next_section_index += 1

        heading_stack: dict[int, str] = {}

        for heading_index, (
            start_offset,
            level,
            title,
        ) in enumerate(headings):
            for existing_level in list(
                heading_stack
            ):
                if existing_level >= level:
                    del heading_stack[existing_level]

            heading_stack[level] = title

            heading_path = tuple(
                heading_stack[path_level]
                for path_level in sorted(
                    heading_stack
                )
            )

            end_offset = (
                headings[heading_index + 1][0]
                if heading_index + 1 < len(headings)
                else len(content)
            )

            sections.append(
                ParsedSection(
                    section_index=next_section_index,
                    title=title,
                    level=level,
                    heading_path=heading_path,
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
            )
            next_section_index += 1

        return tuple(sections)

    @classmethod
    def _normalize_text(
        cls,
        text: str,
    ) -> str:
        """清理解析文本并统一换行。"""

        normalized_text = unicodedata.normalize(
            "NFC",
            text,
        )

        normalized_text = (
            normalized_text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        normalized_text = (
            cls.CONTROL_CHARACTER_PATTERN.sub(
                "",
                normalized_text,
            )
        )

        normalized_lines = [
            line.rstrip()
            for line in normalized_text.splitlines()
        ]

        normalized_text = "\n".join(
            normalized_lines
        )

        normalized_text = re.sub(
            r"\n{3,}",
            "\n\n",
            normalized_text,
        )

        return normalized_text.strip()

    @classmethod
    def _looks_garbled(
        cls,
        text: str,
    ) -> bool:
        """检测明显乱码和异常字体字符映射。"""

        compact_text = "".join(
            character
            for character in text
            if not character.isspace()
        )

        if not compact_text:
            return True

        common_garbled_markers = (
            "锟斤拷",
            "ï¿½",
        )

        if any(
            marker in compact_text
            for marker in common_garbled_markers
        ):
            return True

        if cls.CID_CHARACTER_PATTERN.search(
            compact_text
        ):
            return True

        suspicious_character_count = 0
        unexpected_script_count = 0

        for character in compact_text:
            code_point = ord(character)

            if character == "\ufffd":
                suspicious_character_count += 1
                continue

            is_private_use_character = (
                0xE000 <= code_point <= 0xF8FF
                or 0xF0000 <= code_point <= 0xFFFFD
                or 0x100000 <= code_point <= 0x10FFFD
            )

            if is_private_use_character:
                suspicious_character_count += 1
                continue

            category = unicodedata.category(
                character
            )

            if (
                category.startswith(("L", "M"))
                and not cls._is_expected_character(
                    character
                )
            ):
                unexpected_script_count += 1

        character_count = len(compact_text)

        suspicious_ratio = (
            suspicious_character_count
            / character_count
        )

        unexpected_script_ratio = (
            unexpected_script_count
            / character_count
        )

        return (
            suspicious_ratio
            > cls.MAX_SUSPICIOUS_CHARACTER_RATIO
            or unexpected_script_ratio
            > cls.MAX_UNEXPECTED_SCRIPT_RATIO
        )

    def _ocr_pdf_page(
        self,
        page: fitz.Page,
        page_number: int,
    ) -> str:
        """对单个PDF页面执行整页OCR。"""

        try:
            tessdata = fitz.get_tessdata()

            text_page = page.get_textpage_ocr(
                language=self.PDF_OCR_LANGUAGE,
                dpi=self.PDF_OCR_DPI,
                full=True,
                tessdata=tessdata,
            )

            ocr_text = page.get_text(
                "text",
                textpage=text_page,
                sort=True,
            )

        except Exception as exc:
            raise ValueError(
                f"pdf page {page_number} requires OCR, "
                "but OCR failed; ensure Tesseract and "
                "chi_sim language data are installed"
            ) from exc

        normalized_ocr_text = self._normalize_text(
            ocr_text
        )

        if not normalized_ocr_text:
            raise ValueError(
                f"pdf page {page_number} OCR result "
                "is empty"
            )

        return normalized_ocr_text

    @staticmethod
    def _is_expected_character(
        character: str,
    ) -> bool:
        """判断字符是否属于中文、英文及常见符号范围。"""

        if character.isascii():
            return True

        code_point = ord(character)

        is_cjk_character = (
            0x3400 <= code_point <= 0x4DBF
            or 0x4E00 <= code_point <= 0x9FFF
            or 0xF900 <= code_point <= 0xFAFF
            or 0x20000 <= code_point <= 0x2FA1F
        )

        if is_cjk_character:
            return True

        if (
            0x3000 <= code_point <= 0x303F
            or 0xFF00 <= code_point <= 0xFFEF
        ):
            return True

        category = unicodedata.category(
            character
        )

        return category.startswith(
            ("N", "P", "S")
        )
