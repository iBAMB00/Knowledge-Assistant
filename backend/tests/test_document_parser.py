def test_pdf_parser(tmp_path):
    """
    测试PDF解析。
    """

    import fitz

    from app.services.parser_service import ParserService


    pdf_path = tmp_path / "test.pdf"


    document = fitz.open()

    page = document.new_page()

    page.insert_text(
        (50, 50),
        "Secure Assistant Test",
    )

    document.save(pdf_path)

    document.close()


    parser_service = ParserService()

    text = parser_service.parse(
        str(pdf_path),
    )


    assert "Secure Assistant Test" in text