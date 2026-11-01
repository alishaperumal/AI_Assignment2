import hashlib
import json

from pptx import Presentation

from delivery.build_template_deck import ROOT, clone_slide, fill


TEMPLATE = ROOT / "delivery" / "templates" / "COS711_A2_Presentation_Template.pptx"
DECK = ROOT / "delivery" / "COS711_A2_team_template.pptx"


def test_replacing_template_text_keeps_one_paragraph_properties_element():
    prs = Presentation(TEMPLATE)
    shape = prs.slides[2].placeholders[1]
    fill(shape, ["First replacement", "Second replacement"])
    for paragraph in shape.text_frame.paragraphs:
        assert len(paragraph._p.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}pPr")) == 1
    assert "buChar" in shape.text_frame.paragraphs[0]._p.xml


def test_cloned_template_slide_preserves_layout_images_and_original(tmp_path):
    prs = Presentation(TEMPLATE)
    source = prs.slides[7]
    source_text = source.placeholders[1].text
    clone = clone_slide(prs, source)
    assert clone.slide_layout is source.slide_layout
    fill(clone.placeholders[1], ["New conclusion"])
    assert source.placeholders[1].text == source_text
    output = tmp_path / "clone.pptx"
    prs.save(output)
    reloaded = Presentation(output)
    original_images = [shape.image.sha1 for shape in source.shapes if shape.shape_type == 13]
    cloned_images = [shape.image.sha1 for shape in reloaded.slides[-1].shapes if shape.shape_type == 13]
    assert original_images == cloned_images


def test_delivered_deck_follows_template_and_has_notes_and_placeholders():
    prs, template = Presentation(DECK), Presentation(TEMPLATE)
    assert len(prs.slides) == 14
    assert (prs.slide_width, prs.slide_height) == (template.slide_width, template.slide_height)
    assert len(prs.slide_masters) == len(template.slide_masters)
    text_parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                text_parts.append(shape.text)
            if shape.has_table:
                text_parts.extend(cell.text for row in shape.table.rows for cell in row.cells)
    texts = "\n".join(text_parts)
    assert "COS711 Assignment 2" in texts
    assert "Assignment 3" not in texts
    assert "Public PQ: [ADD]" in texts
    assert "[name / student number]" in texts
    assert "0.3370 +/- 0.0076" in texts
    for slide in prs.slides:
        assert "Evidence:" in slide.notes_slide.notes_text_frame.text


def test_receipt_matches_evidence_and_video_budget():
    receipt = json.loads(DECK.with_suffix(".json").read_text(encoding="utf-8"))
    final = json.loads((ROOT / "results" / "person_c" / "final_submission.json").read_text(encoding="utf-8"))
    assert receipt["template_sha256"] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    assert receipt["final_csv_sha256"] == final["csv_sha256"]
    assert receipt["final_validation_PQ"] == final["validation_PQ"]
    assert receipt["target_narration_seconds"] == 510
    assert receipt["target_narration_seconds"] < 600
    assert len(receipt["test_illustrations"]) == 2
    assert all(case["filaments"] > 0 for case in receipt["test_illustrations"])
