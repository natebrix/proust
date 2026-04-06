import json

from proust.export import ReaderEditionSpec, export_reader_dataset
import proust.export as export_module


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
        },
        {
            "id": "v2-p1",
            "volumeNumber": 2,
            "volumeTitle": "A l'Ombre des Jeunes Filles en Fleurs",
            "partNumber": 1,
            "partTitle": "Autour de Mme Swann",
            "sectionTitle": None,
            "title": "A l'Ombre des Jeunes Filles en Fleurs - I. Autour de Mme Swann",
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
    assert edition["chapters"][0]["partTitle"] == "Combray"
