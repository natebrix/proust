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
                source="file",
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


def test_export_reader_dataset_writes_canonical_manifest_and_structure(monkeypatch, tmp_path):
    canonical_structure = [
        {
            "id": "v1-p1-combray",
            "volumeNumber": 1,
            "volumeTitle": "Du Cote de Chez Swann",
            "partNumber": 1,
            "partTitle": "Combray",
            "sectionTitle": None,
            "title": "Du Cote de Chez Swann - I. Combray",
            "startChapterId": "001",
            "endChapterId": "002",
        },
        {
            "id": "v2-p1",
            "volumeNumber": 2,
            "volumeTitle": "A l'Ombre des Jeunes Filles en Fleurs",
            "partNumber": 1,
            "partTitle": "Autour de Mme Swann",
            "sectionTitle": None,
            "title": "A l'Ombre des Jeunes Filles en Fleurs - I. Autour de Mme Swann",
            "startChapterId": "003",
            "endChapterId": "004",
        },
    ]
    chapters = {
        "v1-p1-combray": {
            "edition": "fr-original",
            "chapterId": "v1-p1-combray",
            "chapterNumber": 1,
            "title": "Du Cote de Chez Swann - I. Combray",
            "volumeNumber": 1,
            "volumeTitle": "Du Cote de Chez Swann",
            "partNumber": 1,
            "partTitle": "Combray",
            "sectionTitle": None,
            "sourceChapterRange": {"start": "001", "end": "002"},
            "paragraphCount": 1,
            "sentenceCount": 1,
            "prevChapterId": None,
            "nextChapterId": "v2-p1",
            "paragraphs": [{"id": "p-1", "index": 1, "text": "Text 1.", "sentences": []}],
        },
        "v2-p1": {
            "edition": "fr-original",
            "chapterId": "v2-p1",
            "chapterNumber": 2,
            "title": "A l'Ombre des Jeunes Filles en Fleurs - I. Autour de Mme Swann",
            "volumeNumber": 2,
            "volumeTitle": "A l'Ombre des Jeunes Filles en Fleurs",
            "partNumber": 1,
            "partTitle": "Autour de Mme Swann",
            "sectionTitle": None,
            "sourceChapterRange": {"start": "003", "end": "004"},
            "paragraphCount": 1,
            "sentenceCount": 1,
            "prevChapterId": "v1-p1-combray",
            "nextChapterId": None,
            "paragraphs": [{"id": "p-1", "index": 1, "text": "Text 2.", "sentences": []}],
        },
    }

    monkeypatch.setattr(export_module, "get_canonical_structure", lambda edition="fr-original": canonical_structure)
    monkeypatch.setattr(export_module, "get_canonical_chapter", lambda chapter_id, edition="fr-original": chapters[chapter_id])

    manifest = export_reader_dataset(
        tmp_path,
        edition_specs=[
            ReaderEditionSpec(
                id="fr-original",
                title="French Original",
                language="fr",
                description="Canonical French text",
                source="canonical",
            )
        ],
    )

    manifest_path = tmp_path / "manifest.json"
    chapter_path = tmp_path / "editions" / "fr-original" / "chapters" / "v1-p1-combray.json"

    assert manifest["defaultEdition"] == "fr-original"
    assert manifest_path.exists()
    assert chapter_path.exists()

    manifest_data = json.loads(manifest_path.read_text())
    edition = manifest_data["editions"][0]
    assert edition["chapterCount"] == 2
    assert edition["isComplete"] is True
    assert edition["structure"] == [
        {
            "id": "v1",
            "number": 1,
            "title": "Du Cote de Chez Swann",
            "chapterIds": ["v1-p1-combray"],
            "parts": [
                {
                    "id": "v1-p1",
                    "number": 1,
                    "title": "Combray",
                    "chapterIds": ["v1-p1-combray"],
                }
            ],
        },
        {
            "id": "v2",
            "number": 2,
            "title": "A l'Ombre des Jeunes Filles en Fleurs",
            "chapterIds": ["v2-p1"],
            "parts": [
                {
                    "id": "v2-p1",
                    "number": 1,
                    "title": "Autour de Mme Swann",
                    "chapterIds": ["v2-p1"],
                }
            ],
        },
    ]
    assert edition["chapters"][0]["id"] == "v1-p1-combray"
    assert edition["chapters"][0]["sourceChapterRange"] == {"start": "001", "end": "002"}
