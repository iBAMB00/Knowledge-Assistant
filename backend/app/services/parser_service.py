from pathlib import Path

import fitz


class ParserService:
    """
    文档解析服务。

    负责读取文件并提取文本内容。
    当前支持 PDF 和 TXT 文件。
    """


    def parse(
        self,
        file_path: str,
    ) -> str:
        """
        解析文档内容。

        Args:
            file_path:
                文档文件路径。

        Returns:
            提取后的文本内容。

        Raises:
            ValueError:
                文件不存在或类型不支持。
        """

        path = Path(file_path)

        if not path.exists():
            raise ValueError(
                "document file not found"
            )

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(path)

        if suffix == ".txt":
            return self._parse_txt(path)

        raise ValueError(
            f"unsupported file type: {suffix}"
        )


    def _parse_pdf(
        self,
        path: Path,
    ) -> str:
        """
        解析PDF文件。

        Args:
            path:
                PDF文件路径。

        Returns:
            PDF文本内容。
        """

        texts: list[str] = []

        with fitz.open(path) as document:
            for page in document:
                texts.append(
                    page.get_text()
                )

        return "\n".join(texts)


    def _parse_txt(
        self,
        path: Path,
    ) -> str:
        """
        解析TXT文件。

        Args:
            path:
                TXT文件路径。

        Returns:
            TXT文本内容。
        """

        return path.read_text(
            encoding="utf-8",
        )