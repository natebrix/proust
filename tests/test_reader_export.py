import json
from types import SimpleNamespace

from proust import create_session
from proust.export import ReaderEditionSpec, build_reader_chapter, export_reader_dataset
import proust.export as export_module


HTML = """
<html>
  <body>
    <h1>001 : Combray</h1>
    <div class="field-item">
      <p>Premier paragraphe. Deuxieme phrase.</p>
      <p>Troisieme phrase.</p>
    </div>
  </body>
</html>
"""


class FakePipeline:
    def __call__(self, text):
        sentences = [segment.strip() for segment in text.split(".") if segment.strip()]
        return SimpleNamespace(
            sents=[SimpleNamespace(text=segment + ".") for segment in sentences],
            ents=[],
        )


def test_build_reader_chapter_uses_stable_paragraph_and_sentence_ids():
    session = create_session(aliases={}, nlp=FakePipeline())
    session.get_proust_page = lambda chapter_number, source=None: HTML

    chapter = build_reader_chapter(session, edition="fr-original", chapter_number=1, source="file")

    assert chapter["chapterId"] == "001"
    assert chapter["paragraphCount"] == 2
    assert chapter["sentenceCount"] == 3
    assert chapter["paragraphs"][0]["id"] == "p-1"
    assert chapter["paragraphs"][0]["sentences"] == [
        {"id": "s-1", "index": 1, "text": "Premier paragraphe."},
        {"id": "s-2", "index": 2, "text": "Deuxieme phrase."},
    ]
    assert chapter["paragraphs"][1]["id"] == "p-2"


def test_export_reader_dataset_writes_manifest_and_chapter_files(monkeypatch, tmp_path):
    chapter_template = {
        "edition": "fr-original",
        "title": "Chapter",
        "paragraphCount": 1,
        "sentenceCount": 1,
        "paragraphs": [{"id": "p-1", "index": 1, "text": "Text.", "sentences": []}],
    }

    monkeypatch.setattr(export_module, "create_session", lambda **kwargs: object())
    monkeypatch.setattr(
        export_module,
        "build_reader_chapter",
        lambda session, edition, chapter_number, source=None: {
            **chapter_template,
            "chapterId": f"{chapter_number:03}",
            "chapterNumber": chapter_number,
            "title": f"Chapter {chapter_number}",
        },
    )

    manifest = export_reader_dataset(
        tmp_path,
        edition_specs=[
            ReaderEditionSpec(
                id="fr-original",
                title="French Original",
                language="fr",
                chapter_start=1,
                chapter_end=2,
            )
        ],
    )

    manifest_path = tmp_path / "manifest.json"
    chapter_path = tmp_path / "editions" / "fr-original" / "chapters" / "001.json"

    assert manifest["defaultEdition"] == "fr-original"
    assert manifest_path.exists()
    assert chapter_path.exists()

    manifest_data = json.loads(manifest_path.read_text())
    assert manifest_data["editions"][0]["chapters"] == [
        {
            "id": "001",
            "number": 1,
            "title": "Chapter 1",
            "paragraphCount": 1,
            "sentenceCount": 1,
            "prevChapterId": None,
            "nextChapterId": "002",
        },
        {
            "id": "002",
            "number": 2,
            "title": "Chapter 2",
            "paragraphCount": 1,
            "sentenceCount": 1,
            "prevChapterId": "001",
            "nextChapterId": None,
        },
    ]
