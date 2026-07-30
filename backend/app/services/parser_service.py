import re
import unicodedata
from pathlib import Path

import fitz

from app.schemas.parse_result import ParseResult


class ParserService:
    """
    文档解析服务。

    负责从文件二进制内容中提取文本。
    不负责读取存储系统，也不依赖本地文件路径。
    当前支持PDF和TXT文件。
    """

    PDF_PARSER_VERSION = "1.2.0"
    TXT_PARSER_VERSION = "1.1.0"

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

    def parse(
        self,
        filename: str,
        content: bytes,
    ) -> ParseResult:
        """
        解析文档二进制内容。

        Args:
            filename:
                文档原始文件名，用于判断文件类型。

            content:
                由存储服务读取的文件二进制内容。

        Returns:
            包含解析文本和解析器类型的结果对象。

        Raises:
            ValueError:
                文件名为空、内容为空、文件类型不支持，
                或文件内容无法正常解析。
        """

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
            )

        if suffix == ".txt":
            return ParseResult(
                content=self._parse_txt(content),
                parser_type="plain_text",
                parser_version=self.TXT_PARSER_VERSION,
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

        优先使用PDF文本层；
        页面为空或文字层乱码时，降级为整页OCR。

        Returns:
            解析文本，以及是否使用过OCR。
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

                    # 没有文本时，先区分空白页和图片扫描页。
                    if not normalized_page_text:
                        has_images = bool(
                            page.get_images(full=True)
                        )

                        # 空白页直接跳过，不需要调用OCR。
                        if not has_images:
                            continue

                        normalized_page_text = (
                            self._ocr_pdf_page(
                                page=page,
                                page_number=page_number,
                            )
                        )
                        used_ocr = True

                    # 有文本但质量异常，说明可能存在字体映射问题，
                    # 此时对整页执行OCR。
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
        """
        从TXT二进制内容中读取并清理文本。
        """

        try:
            # utf-8-sig同时兼容普通UTF-8和带BOM的UTF-8。
            decoded_text = content.decode(
                "utf-8-sig"
            )

        except UnicodeDecodeError as exc:
            raise ValueError(
                "txt document must use utf-8 encoding"
            ) from exc

        normalized_text = self._normalize_text(
            decoded_text
        )

        if not normalized_text:
            raise ValueError(
                "txt document contains no text"
            )

        return normalized_text

    @classmethod
    def _normalize_text(
        cls,
        text: str,
    ) -> str:
        """
        清理解析文本。

        处理：
        - Unicode标准化
        - 统一换行符
        - 移除非法控制字符
        - 移除行尾空白
        - 压缩过多空行
        """

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

        # 连续三个及以上换行压缩成一个空行。
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
        """
        检测明显乱码和异常字体字符映射。
        """

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

            # 对中文+英文文档来说，
            # 非预期文字或组合标记大量出现，
            # 通常表示PDF字体映射异常。
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
        """
        对单个PDF页面执行整页OCR。
        """

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
        """
        判断字符是否属于中文、英文及常见符号范围。
        """

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

        # 中文标点与全角字符。
        if (
            0x3000 <= code_point <= 0x303F
            or 0xFF00 <= code_point <= 0xFFEF
        ):
            return True

        category = unicodedata.category(
            character
        )

        # 数字、标点和数学/技术符号允许出现。
        return category.startswith(
            ("N", "P", "S")
        )