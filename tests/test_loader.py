import json
from pathlib import Path

import pytest

from src.data.loader import StreamingJSONLLoader
from src.data.document import Document


BASE_TMP = Path(__file__).parent / "tmp_data"


def make_tmp_file(name: str) -> Path:
    BASE_TMP.mkdir(exist_ok=True)
    return BASE_TMP / name


def write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            if isinstance(rec, str):
                f.write(rec + "\n")
            else:
                f.write(json.dumps(rec) + "\n")


def test_loader_raises_if_file_missing():
    missing = make_tmp_file("missing.jsonl")
    if missing.exists():
        missing.unlink()
    with pytest.raises(FileNotFoundError):
        StreamingJSONLLoader(str(missing))


def test_parse_year_from_date_formats():
    tmp = make_tmp_file("parse_year.jsonl")
    write_jsonl(tmp, [])
    loader = StreamingJSONLLoader(str(tmp))

    # RFC 2822 style
    assert loader._parse_year_from_date("Sun, 1 Apr 2007 13:06:50 GMT") == 2007
    # Fallback regex extraction
    assert loader._parse_year_from_date("published on 2020-01-01") == 2020
    # No year
    assert loader._parse_year_from_date("no year here") is None
    # None / empty
    assert loader._parse_year_from_date("") is None
    assert loader._parse_year_from_date(None) is None


def test_basic_load_yields_document_with_metadata_and_year():
    tmp = make_tmp_file("basic.jsonl")
    write_jsonl(
        tmp,
        [
            {"id": 123, "abstract": "hello", "title": "T", "published_date": "2021-05-01", "extra": "x"}
        ],
    )

    loader = StreamingJSONLLoader(str(tmp))
    docs = list(loader.load())
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    # id coerced to string
    assert doc.id == "123"
    assert doc.text == "hello"
    assert doc.title == "T"
    assert doc.published_year == 2021
    # extra field moved to metadata
    assert doc.metadata == {"published_date": "2021-05-01", "extra": "x"}


def test_load_auto_id_and_skips_empty_and_invalid_lines(capsys):
    tmp = make_tmp_file("auto_id.jsonl")
    write_jsonl(
        tmp,
        [
            "",  # empty line
            "{not json}",  # invalid json
            {"abstract": "text without id"},  # no id -> auto_3
            {"id": "ok", "abstract": ""},  # empty text -> skipped by default
        ],
    )

    loader = StreamingJSONLLoader(str(tmp))
    docs = list(loader.load())

    # Only the third logical line yields a document
    assert [d.id for d in docs] == ["auto_3"]
    assert [d.text for d in docs] == ["text without id"]

    captured = capsys.readouterr().out
    assert "Skipping invalid JSON at line 2" in captured


def test_load_with_metadata_template_and_lists():
    tmp = make_tmp_file("metadata_template.jsonl")
    write_jsonl(
        tmp,
        [
            {
                "id": "1",
                "abstract": "Body",
                "title": "My Title",
                "categories": ["cs.AI", "cs.LG"],
                "authors": ["Alice", "Bob"],
            }
        ],
    )

    template = "Title: {title} | Categories: {categories} | Authors: {authors} | Abstract: {abstract}"
    loader = StreamingJSONLLoader(
        str(tmp),
        use_metadata=True,
        metadata_template=template,
    )

    doc = next(iter(loader.load()))
    # Text should be built from metadata template; order and separators matter
    assert doc.text == "Title: My Title | Categories: cs.AI, cs.LG | Authors: Alice, Bob | Abstract: Body"


def test_load_with_metadata_template_and_scalar_categories_authors():
    tmp = make_tmp_file("metadata_template_scalar.jsonl")
    write_jsonl(
        tmp,
        [
            {
                "id": "1",
                "abstract": "Body",
                "title": "My Title",
                "categories": "cs.AI cs.LG",
                "authors": "Alice;Bob",
            }
        ],
    )

    template = "Title: {title} | Categories: {categories} | Authors: {authors} | Abstract: {abstract}"
    loader = StreamingJSONLLoader(
        str(tmp),
        use_metadata=True,
        metadata_template=template,
    )

    doc = next(iter(loader.load()))
    # When categories/authors are not lists, they should be stringified directly
    assert doc.text == "Title: My Title | Categories: cs.AI cs.LG | Authors: Alice;Bob | Abstract: Body"


def test_metadata_template_when_categories_and_authors_keys_disabled():
    tmp = make_tmp_file("metadata_no_keys.jsonl")
    write_jsonl(
        tmp,
        [
            {
                "id": "1",
                "abstract": "Body",
                "title": "My Title",
                # categories/authors present in data but keys are disabled on loader
                "categories": ["cs.AI", "cs.LG"],
                "authors": ["Alice", "Bob"],
            }
        ],
    )

    template = "Title: {title} | Categories: {categories} | Authors: {authors} | Abstract: {abstract}"
    loader = StreamingJSONLLoader(
        str(tmp),
        use_metadata=True,
        metadata_template=template,
        categories_key=None,
        authors_key=None,
    )

    doc = next(iter(loader.load()))
    # When keys are disabled, categories/authors should be empty strings
    assert doc.text == "Title: My Title | Categories:  | Authors:  | Abstract: Body"


def test_metadata_template_missing_key_falls_back_to_original_text(capsys):
    tmp = make_tmp_file("metadata_missing_key.jsonl")
    write_jsonl(
        tmp,
        [
            {"id": "1", "abstract": "Original", "title": "T"},
        ],
    )

    # Template refers to unknown key {unknown}
    template = "{unknown}"
    loader = StreamingJSONLLoader(
        str(tmp),
        use_metadata=True,
        metadata_template=template,
    )

    doc = next(iter(loader.load()))
    # Because formatting failed, loader should keep original text
    assert doc.text == "Original"

    captured = capsys.readouterr().out
    assert "Missing key" in captured


def test_skip_empty_false_includes_empty_text():
    tmp = make_tmp_file("skip_empty_false.jsonl")
    write_jsonl(
        tmp,
        [
            {"id": "1", "abstract": ""},
        ],
    )

    loader = StreamingJSONLLoader(str(tmp), skip_empty=False)
    # Document enforces non-empty text, so this should raise
    with pytest.raises(ValueError):
        list(loader.load())


def test_count_documents_counts_non_empty_lines():
    tmp = make_tmp_file("count_docs.jsonl")
    # Two non-empty, one empty line
    write_jsonl(tmp, [{"a": 1}, "", {"b": 2}])

    loader = StreamingJSONLLoader(str(tmp))
    assert loader.count_documents() == 2
