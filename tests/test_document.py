import pytest

from src.data.document import Document


def test_document_valid_creation():
    doc = Document(id="123", text="Hello world", title="Test Title")
    assert doc.id == "123"
    assert doc.text == "Hello world"
    assert doc.title == "Test Title"


def test_document_empty_id_raises_error():
    with pytest.raises(ValueError):
        Document(id="", text="Some text")


def test_document_empty_text_raises_error():
    with pytest.raises(ValueError):
        Document(id="123", text="")
