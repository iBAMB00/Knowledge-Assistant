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
    当前支持 PDF 和 TXT 文件。
    """
    PDF_PARSER_VERSION = "1.1.0"

    TXT_PARSER_VERSION = "1.0.0"

    CONTROL_CHARACTER_PATTERN = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
    )

    CID_CHARACTER_PATTERN = re.compile(
        r"\(cid:\d+\)",
        re.IGNORECASE,
    )

    MAX_SUSPICIOUS_CHARACTER_RATIO = 0.02

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
            return ParseResult(
                content=self._parse_pdf(content),
                parser_type="pymupdf",
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
    ) -> str:
        """
        从PDF二进制内容中提取文本。

        普通文本型PDF使用PyMuPDF提取。
        扫描件或字符映射异常的PDF明确提示需要OCR，
        不允许乱码继续进入Chunk和Embedding。
        """

        try:
            page_texts: list[str] = []
            image_only_pages: list[int] = []

            # 直接从内存打开PDF，不依赖本地文件路径。
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
                    # sort=True按照页面坐标重新排列文本，
                    # 比默认的PDF对象存储顺序更接近阅读顺序。
                    page_text = page.get_text(
                        "text",
                        sort=True,
                    )

                    normalized_page_text = (
                        self._normalize_text(page_text)
                    )

                    if normalized_page_text:
                        page_texts.append(
                            normalized_page_text
                        )
                        continue

                    # 当前页面无可提取文本但包含图片，
                    # 通常说明该页属于扫描件。
                    if page.get_images(full=True):
                        image_only_pages.append(
                            page_number
                        )

            parsed_text = "\n\n".join(
                page_texts
            ).strip()

            if not parsed_text:
                if image_only_pages:
                    page_numbers = ", ".join(
                        str(page_number)
                        for page_number
                        in image_only_pages
                    )

                    raise ValueError(
                        "pdf contains image-only pages "
                        "and requires OCR: "
                        f"pages {page_numbers}"
                    )

                raise ValueError(
                    "pdf contains no extractable text"
                )

            if self._looks_garbled(parsed_text):
                raise ValueError(
                    "pdf text extraction quality is too low; "
                    "OCR is required"
                )

            return parsed_text

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
        检测明显乱码。

        当前检测：
        - Unicode替换字符
        - Unicode私有区字符
        - PDF缺失字符映射产生的(cid:xxx)
        - 常见错误解码标记

        该方法只负责质量拦截，不负责修复乱码。
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

        suspicious_character_count = 0

        for character in compact_text:
            if character == "\ufffd":
                suspicious_character_count += 1
                continue

            if unicodedata.category(
                character
            ) == "Co":
                suspicious_character_count += 1

        cid_matches = (
            cls.CID_CHARACTER_PATTERN.findall(
                compact_text
            )
        )

        cid_character_count = sum(
            len(match)
            for match in cid_matches
        )

        suspicious_character_count += (
            cid_character_count
        )

        suspicious_ratio = (
            suspicious_character_count
            / len(compact_text)
        )

        return (
            suspicious_ratio
            > cls.MAX_SUSPICIOUS_CHARACTER_RATIO
        )