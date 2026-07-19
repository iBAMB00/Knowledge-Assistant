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
            )

        if suffix == ".txt":
            return ParseResult(
                content=self._parse_txt(content),
                parser_type="plain_text",
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

        Args:
            content:
                PDF文件二进制内容。

        Returns:
            PDF中的文本内容。

        Raises:
            ValueError:
                PDF文件损坏或格式不正确。
        """

        try:
            texts: list[str] = []

            # 直接从内存打开PDF，不依赖本地文件路径。
            with fitz.open(
                stream=content,
                filetype="pdf",
            ) as document:
                for page in document:
                    texts.append(
                        page.get_text()
                    )

            return "\n".join(texts)

        except Exception as exc:
            raise ValueError(
                "failed to parse pdf document"
            ) from exc

    def _parse_txt(
        self,
        content: bytes,
    ) -> str:
        """
        从TXT二进制内容中读取文本。

        Args:
            content:
                TXT文件二进制内容。

        Returns:
            解码后的文本内容。

        Raises:
            ValueError:
                TXT文件不是UTF-8编码。
        """

        try:
            # utf-8-sig 同时兼容普通UTF-8和带BOM的UTF-8文本。
            return content.decode(
                "utf-8-sig"
            )

        except UnicodeDecodeError as exc:
            raise ValueError(
                "txt document must use utf-8 encoding"
            ) from exc