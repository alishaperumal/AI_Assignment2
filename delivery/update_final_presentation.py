"""Add the template items the final presentation was missing (python-pptx).

Reads the team's designed deck and writes a NEW file (the input is never modified).
New content reuses the deck's own shapes (cards, badges, bar rows, reference rows)
and copies each base slide's header, background, presenter label and Morph
transition, so the design stays consistent. Kaggle placeholders are left alone.

Changes:
  * slides 1-2: hidden "[u00000000]" student number corrected
  * "Our Choices": image resolution, standardisation and the augmentation list
  * NEW "Noisy Labels": noisy/faint/disagreeing examples and annotator agreement
  * "Experimental Setup": problem formulation and the reasons for each choice
  * NEW "Tuning Results": the 11-recipe six-epoch screen
  * NEW "Kinds of Errors": the final model's error breakdown on validation
  * "Example Predictions" -> best and worst validation cases of the final model
  * NEW "What Else We Tried": ideas that did not help, retrained and combined models
  * "Conclusions": roadblocks and what we would do differently
  * NEW bibliography slide for data and libraries

Run delivery/final_deck_figures.py first. Usage:
  python delivery/update_final_presentation.py --input IN.pptx --output OUT.pptx
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "delivery" / ".build" / "final_additions"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
CHROME = {"Google Shape;40;p3", "Oval 11", "Oval 12", "Oval 13", "Oval 14", "Oval 16", "Oval 21", "Group 17",
          "Google Shape;42;p3", "Section Title", "Presenter Icon", "Presenter Name", "Rectangle 9"}


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def by_name(slide):
    return {shape.name: shape for shape in slide.shapes}


def _remap(element, source_part, target_part):
    for node in element.iter():
        for attribute, value in list(node.attrib.items()):
            if attribute.startswith(RNS) and value in source_part.rels:
                rel = source_part.rels[value]
                target = rel.target_ref if rel.is_external else rel.target_part
                node.set(attribute, target_part.relate_to(target, rel.reltype, rel.is_external))


def _fresh_ids(element, slide):
    used = [int(n.get("id")) for n in slide._element.iter(qn("p:cNvPr"))]
    next_id = max(used + [1]) + 1
    for node in element.iter(qn("p:cNvPr")):
        node.set("id", str(next_id))
        next_id += 1


def _shape_for(slide, element):
    return next(shape for shape in slide.shapes if shape._element is element)


def place(shape, x=None, y=None, w=None, h=None):
    if x is not None:
        shape.left = Inches(x)
    if y is not None:
        shape.top = Inches(y)
    if w is not None:
        shape.width = Inches(w)
    if h is not None:
        shape.height = Inches(h)
    return shape


def clone(source_slide, name, slide, new_name, x=None, y=None, w=None, h=None):
    """Copy a named shape (with its images/links) below the header band of ``slide``."""
    element = copy.deepcopy(by_name(source_slide)[name]._element)
    _remap(element, source_slide.part, slide.part)
    _fresh_ids(element, slide)
    element.find(".//" + qn("p:cNvPr")).set("name", new_name)
    by_name(slide)["Presenter Icon"]._element.addprevious(element)
    return place(_shape_for(slide, element), x, y, w, h)


def picture(slide, path, new_name, x, y, w, h):
    shape = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    shape._element.find(".//" + qn("p:cNvPr")).set("name", new_name)
    by_name(slide)["Presenter Icon"]._element.addprevious(shape._element)
    return shape


def replace_image(shape, path):
    blip = shape._element.find(".//" + qn("a:blip"))
    old = blip.get(qn("r:embed"))
    _, rid = shape.part.get_or_add_image_part(str(path))
    blip.set(qn("r:embed"), rid)
    for crop in shape._element.iter(qn("a:srcRect")):
        crop.getparent().remove(crop)
    shape.part.drop_rel(old)


class Styles:
    """Run properties copied from runs already in the deck."""

    def __init__(self, prs):
        s = prs.slides

        def run(slide, name, paragraph=0, index=0):
            return copy.deepcopy(by_name(s[slide - 1])[name].text_frame.paragraphs[paragraph].runs[index]._r
                                 .find(qn("a:rPr")))

        self.rpr = dict(
            accent=run(8, "Sampling Text", 0, 0), body=run(8, "Sampling Text", 0, 1), grey=run(8, "Split Note"),
            muted=run(12, "Examples Legend", 1), strong=run(9, "Flow Model Text"), warn=run(16, "Conclusion 3 Text"),
            label=run(12, "Grid Head 1"), row=run(14, "Post Row 1 Label"), row_strong=run(14, "Post Row 4 Label"),
            value=run(14, "Post Row 1 Value"), value_teal=run(14, "Post Row 4 Value"),
            stat_value=run(7, "Stat 1 Text", 0), stat_label=run(7, "Stat 1 Text", 1),
            ref_bold=run(18, "Ref 7 Text", 0, 0), ref_grey=run(18, "Ref 7 Text", 0, 1))

    def get(self, style, size=None):
        rpr = copy.deepcopy(self.rpr[style])
        if size is not None:
            rpr.set("sz", str(int(round(size * 100))))
        return rpr


def write(shape, styles, paragraphs, size=None, align=None, space_before=None):
    """Replace a text frame's content. Paragraph = list of (text, style[, size]) runs."""
    body = shape.text_frame._txBody
    existing = body.findall(qn("a:p"))
    templates = [copy.deepcopy(p.find(qn("a:pPr"))) for p in existing] or [None]
    for p in existing:
        body.remove(p)
    for i, runs in enumerate(paragraphs):
        p = etree.SubElement(body, qn("a:p"))
        ppr = copy.deepcopy(templates[min(i, len(templates) - 1)])
        if ppr is not None:
            if align:
                ppr.set("algn", align)
            if space_before is not None and i:
                spc = ppr.find(qn("a:spcBef"))
                if spc is not None and spc.find(qn("a:spcPts")) is not None:
                    spc.find(qn("a:spcPts")).set("val", str(int(space_before * 100)))
            p.append(ppr)
        for spec in runs:
            text, style = spec[0], spec[1]
            r = etree.SubElement(p, qn("a:r"))
            r.append(styles.get(style, spec[2] if len(spec) > 2 else size))
            etree.SubElement(r, qn("a:t")).text = text
    return shape


def set_plain(shape, text):
    """Keep the first run's formatting, replace all text."""
    paragraphs = shape.text_frame.paragraphs
    first = paragraphs[0]
    first.runs[0].text = text
    for extra in first.runs[1:]:
        extra._r.getparent().remove(extra._r)
    for para in paragraphs[1:]:
        para._p.getparent().remove(para._p)


def notes(slide, owner, seconds, narration, evidence):
    slide.notes_slide.notes_text_frame.text = (
        f"{owner} | Target: {seconds} seconds\n\n{narration}\n\nEvidence:\n" + "\n".join(evidence))


def new_slide(prs, base, title, presenter):
    """Header, background, decorations, presenter label and transition of ``base``; no content."""
    slide = prs.slides.add_slide(base.slide_layout)
    tree = slide.shapes._spTree
    for shape in list(slide.shapes):
        tree.remove(shape._element)
    for shape in base.shapes:
        if shape.name in CHROME:
            element = copy.deepcopy(shape._element)
            _remap(element, base.part, slide.part)
            tree.insert_element_before(element, "p:extLst")
    sld = slide._element
    for child in list(sld):
        if child.tag != qn("p:cSld"):
            sld.remove(child)
    for child in base._element:
        if child.tag not in (qn("p:cSld"), qn("p:extLst")):
            sld.append(copy.deepcopy(child))
    shapes = by_name(slide)
    set_plain(shapes["Section Title"], title)
    set_plain(shapes["Presenter Name"], presenter)
    return slide


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
def fix_student_number(prs):
    changed = 0
    for slide in list(prs.slides)[:2]:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        if "u00000000" in run.text:
                            run.text = run.text.replace("u00000000", "u22512285")
                            changed += 1
    return changed


def our_choices(slide, st):
    s = by_name(slide)
    write(s["Sampling Text"], st, [[("Full 2048 × 2048 images: ", "accent"),
                                    ("train on random 512 × 512 patches (70% centred on filaments); "
                                     "predict whole images.", "body")]], size=13)
    write(s["Preprocessing Text"], st, [
        [("Scaling: ", "accent"), ("pixels standardised with the training-set mean and spread.", "body")],
        [("Augmentation ", "accent"), ("(flips, rotations, scaling, brightness, noise, blur) scored lower, "
                                      "so it was switched off: 0.3201 → 0.3381.", "body")]], size=13)
    notes(slide, "Alisha Perumal", 35,
          "All annotation sets of one image stay on the same side of the split, and we split by 14-day blocks "
          "with a one-day gap so near-identical images do not leak; validation matches the test instrument mix. "
          "Images are 2048 by 2048 pixels: we train on random 512-pixel patches, mostly centred on filaments, and "
          "predict whole images. Pixels are standardised with training-set statistics. Extra augmentation, meaning "
          "flips, rotations, scaling, brightness, noise and blur, scored lower, so we switched it off.",
          ["splits/split_v1.json", "configs/person_c/seed45_selected.yaml",
           "results/person_b_study/confirmation.csv", "results/person_b_study/methodology.md"])


def noisy_labels(prs, base, src, st, info):
    slide = new_slide(prs, base, "Dataset Preparation: Noisy Labels", "Alisha Perumal")
    panels = [("NOISY IMAGE", "noisy_noisy.png", "Most image noise"),
              ("FAINT FILAMENT", "noisy_faint.png", "Lowest filament contrast"),
              ("ANNOTATORS DISAGREE", "noisy_disagree.png", "Teal / orange = two annotators")]
    for i, (head, image, caption) in enumerate(panels):
        x = 0.6 + i * 2.32
        write(clone(src[12], "Grid Head 1", slide, f"Noisy Head {i + 1}", x, 2.1, 2.15, 0.3), st, [[(head, "label")]])
        picture(slide, FIG / image, f"Noisy Panel {i + 1}", x, 2.45, 2.15, 2.15)
        write(clone(src[8], "Split Note", slide, f"Noisy Caption {i + 1}", x, 4.64, 2.15, 0.34), st,
              [[(caption, "grey", 11)]])
    clone(src[9], "Loss", slide, "Handling Card", 0.6, 5.12, 6.79, 1.6)
    clone(src[9], "Loss Badge Ring", slide, "Handling Badge Ring", 0.8, 5.67, 0.5, 0.5)
    clone(src[9], "Loss Badge Icon", slide, "Handling Badge Icon", 0.92, 5.79, 0.26, 0.26)
    write(clone(src[9], "Loss Text", slide, "Handling Text", 1.5, 5.22, 5.7, 1.4), st, [[
        ("How we handled it: ", "accent"),
        ("every annotation set is its own training target, and scoring uses all 200 validation sets. "
         "Voted labels were tested (0.272 vs 0.255, 6-epoch screen) but not promoted.", "body")]], size=13)
    stats = [("0.34", "PQ when one annotator is scored against another (100 images)"),
             ("296 of 707", "images have 2–3 different annotation sets"),
             ("5", "sets with overlapping outlines, kept as drawn; 0 empty sets")]
    for i, (value, label) in enumerate(stats):
        y = 2.1 + i * 1.57
        clone(src[7], "Stat 1", slide, f"Label Stat {i + 1}", 7.8, y, 4.93, 1.4)
        write(clone(src[7], "Stat 1 Text", slide, f"Label Stat {i + 1} Text", 8.0, y + 0.08, 4.55, 1.24), st,
              [[(value, "stat_value")], [(label, "stat_label", 13)]])
    notes(slide, "Alisha Perumal", 30,
          "Labels are noisy. These are the noisiest image, the faintest filament and the biggest annotator "
          "disagreement from our data analysis. One annotator scored against another gets a PQ of only 0.34, a "
          "reference point rather than a ceiling. 296 of 707 images have two or three annotation sets. We kept every "
          "annotation set as its own training target and score against all 200 validation sets; voted labels were "
          "tested but not promoted.",
          ["results/eda_summary.json (annotator_agreement, sets_per_observation, overlap_sets, noisy_samples)",
           "results/person_b_study/screening.csv", f"Panels: {json.dumps(info)}"])
    return slide


def setup(slide, st):
    s = by_name(slide)
    write(s["Flow Model Text"], st, [[("U-Net with a ResNet34 encoder", "strong")],
                                     [("ImageNet-pretrained (only 584 training images); "
                                       "skip links keep thin detail", "grey", 12)]])
    write(s["Flow Output Text"], st, [[("Filament score per pixel", "strong")],
                                      [("> 0.5 → connected regions = filaments", "grey", 12)]])
    write(s["Loss Text"], st, [[("Training loss: ", "accent"),
                                ("binary cross entropy + Dice. Dice copes with filaments covering only "
                                 "0.36% of pixels.", "body")]])
    write(s["Hardware Text"], st, [[("Vansh: ", "accent"), ("RTX 5060 Ti. ", "body"), ("Denis: ", "accent"),
                                    ("RTX 3070 (8 GB). Mixed precision saves memory; ONNX Runtime and "
                                     "torch-pruning for compression.", "body")]], size=13)
    notes(slide, "Vansh Sood", 35,
          "We treat it as pixel classification: every pixel gets a filament score, we keep pixels above 0.5, and "
          "each connected region becomes one filament. U-Net's skip connections keep the fine detail thin filaments "
          "need, and the ResNet34 encoder starts from ImageNet because we only have 584 training images. Dice loss "
          "copes with filaments being only 0.36 percent of pixels. AdamW with a cosine schedule trains it, and "
          "mixed precision saves memory.",
          ["results/person_b_study/methodology.md", "models.py", "losses.py", "results/eda_summary.json"])


SCREEN = [("Whole image shrunk to 1024 px", 0.3104, True), ("No added transformations", 0.2856, True),
          ("U-Net++ architecture", 0.2791, False), ("Mask outside the Sun", 0.2783, False),
          ("Contrast boost (CLAHE)", 0.2768, False), ("Voted labels", 0.2723, False),
          ("Local brightness normalisation", 0.2597, False), ("Baseline: learning rate 0.0003", 0.2554, False),
          ("Learning rate 0.0001", 0.2465, False), ("Learning rate 0.001", 0.2424, False),
          ("Tversky loss", 0.2416, False)]


def bar_rows(slide, src, st, prefix, rows, x, y, label_w, track_w, value_w, row_h, scale, fmt):
    """Rows like the deck's 'Post Row' bars. rows = (label, value, highlight[, value_style])."""
    clone(src, "Post Rule 0", slide, f"{prefix} Rule 0", x, y, label_w + track_w + value_w + 0.3, 0)
    for i, row in enumerate(rows, start=1):
        label, value, strong = row[:3]
        top = y + (i - 1) * row_h
        write(clone(src, "Post Row 1 Label", slide, f"{prefix} Row {i} Label", x, top, label_w, row_h), st,
              [[(label, "row_strong" if strong else "row")]])
        bx = x + label_w + 0.1
        clone(src, "Post Row 1 Track", slide, f"{prefix} Row {i} Track", bx, top + row_h / 2 - 0.08, track_w, 0.16)
        clone(src, "Post Row 4 Bar" if strong else "Post Row 1 Bar", slide, f"{prefix} Row {i} Bar",
              bx, top + row_h / 2 - 0.08, max(track_w * value / scale, 0.04), 0.16)
        style = row[3] if len(row) > 3 else ("value_teal" if strong else "value")
        write(clone(src, "Post Row 1 Value", slide, f"{prefix} Row {i} Value",
                    bx + track_w + 0.1, top, value_w, row_h), st, [[(fmt(value), style)]],
              size=12.5 if style == "warn" else None)
        clone(src, "Post Rule 1", slide, f"{prefix} Rule {i}", x, top + row_h, label_w + track_w + value_w + 0.3, 0)


def card(slide, src, st, name, x, y, w, h, badge, runs, size=13):
    """Rounded card with a badge (from slide 9 / 7 / 8 / 10 shapes) and text."""
    ring_source, ring, icon = badge
    clone(src[9], "Loss", slide, f"{name} Card", x, y, w, h)
    clone(src[ring_source], ring, slide, f"{name} Badge Ring", x + 0.2, y + h / 2 - 0.25, 0.5, 0.5)
    clone(src[ring_source], icon, slide, f"{name} Badge Icon", x + 0.32, y + h / 2 - 0.13, 0.26, 0.26)
    write(clone(src[9], "Loss Text", slide, f"{name} Text", x + 0.9, y + 0.08, w - 1.05, h - 0.16), st, [runs],
          size=size)


def tuning_results(prs, base, src, st):
    slide = new_slide(prs, base, "Methodology: Tuning Results", "Vansh Sood")
    write(clone(src[14], "Post Head L", slide, "Screen Head L", 0.6, 2.08, 4.6, 0.3), st,
          [[("6-EPOCH SCREEN · SEED 42 · SAME PATCHES FOR EACH", "label", 10)]])
    write(clone(src[14], "Post Head R", slide, "Screen Head R", 6.2, 2.08, 1.3, 0.3), st,
          [[("VALIDATION PQ", "label", 10)]], align="r")
    bar_rows(slide, src[14], st, "Screen", SCREEN, 0.6, 2.42, 3.3, 2.1, 0.8, 0.385, 0.32, lambda v: f"{v:.4f}")
    badges = [(10, "Evaluation Badge Ring", "Evaluation Badge Icon"), (9, "Training Badge Ring", "Training Badge Icon"),
              (8, "Preprocessing Badge Ring", "Preprocessing Badge Icon"), (7, "Challenges Badge Ring", "Challenges Badge Icon")]
    texts = [[("Top two (teal) ", "accent"), ("went to 20-epoch runs: 0.3375 vs 0.3381, a near-tie.", "body")],
             [("Learning rate: ", "accent"), ("0.0003 beat 0.0001 and 0.001.", "body")],
             [("Preprocessing ", "accent"), ("helped short runs slightly (+0.004 to +0.023) but was not top two.", "body")],
             [("Caveat: ", "warn", 13), ("one seed, short runs; gaps under ~0.01 are within run-to-run noise.", "body")]]
    for i, (badge, runs) in enumerate(zip(badges, texts)):
        card(slide, src, st, f"Tuning {i + 1}", 8.05, 2.1 + i * 1.15, 4.68, 1.0, badge, runs)
    notes(slide, "Vansh Sood", 30,
          "These are the six-epoch screening runs, all with seed 42 and the same training budget. The top two, "
          "1024-pixel whole images and no extra transformations, got full 20-epoch runs and nearly tied. A learning "
          "rate of 0.0003 was best. Preprocessing helped short runs slightly but was not top two, and the Tversky "
          "loss was lowest. These are single short runs, so small gaps should not be over-read.",
          ["results/person_b_study/screening.csv", "results/person_b_study/figures/screening.png",
           "results/person_b_study/confirmation.csv", "results/person_b_study/fresh_seed_summary.csv"])
    return slide


def kinds_of_errors(prs, base, src, st, err):
    slide = new_slide(prs, base, "Results: Kinds of Errors", "Vansh Sood")
    blocks = [(f"THE {err['n_gt']:,} MARKED FILAMENTS", [
                  ("Found (IoU > 0.5)", err["tp"], True),
                  ("Overlapped, but outline too different", err["gt_partial"], False),
                  ("Completely missed", err["gt_none"], False, "warn")]),
              (f"THE {err['n_pred']:,} PREDICTED FILAMENTS", [
                  ("Correct", err["tp"], True),
                  ("Touch a filament, wrong shape or split", err["fp_overlap"], False),
                  ("False alarm in empty background", err["fp_pure"], False, "warn")])]
    for b, (head, rows) in enumerate(blocks):
        y = 2.08 + b * 2.05
        write(clone(src[14], "Post Head L", slide, f"Error Head {b + 1}", 0.6, y, 6.9, 0.3), st,
              [[(head, "label", 10)]])
        bar_rows(slide, src[14], st, f"Error {b + 1}", rows, 0.6, y + 0.36, 3.6, 2.0, 0.7, 0.45,
                 err["n_pred"], lambda v: f"{int(v):,}")
    write(clone(src[8], "Split Note", slide, "Error Scope", 0.6, 6.3, 6.9, 0.45), st,
          [[("Final model (30% pruned + FP16) with cleanup, 122 validation images; counts pooled over 200 "
             "annotation sets.", "muted", 10.5)]])
    texts = [
        [("Small = missed: ", "accent"), (f"completely missed filaments have median {err['median_none']:,} px vs "
                                         f"{err['median_found']:,} px for found ones; {err['none_lt500']:.0%} are under "
                                         f"500 px (found: {err['found_lt500']:.0%}).", "body")],
        [("Split and merged: ", "accent"), (f"{err['one_to_many']} filaments broken into pieces; "
                                            f"{err['many_to_one']} predictions covering two or more.", "body")],
        [("Shape × detection: ", "accent"), (f"PQ {err['pq']:.2f} = outline quality {err['sq']:.2f} × "
                                             f"detection quality {err['rq']:.2f}.", "body")]]
    badges = [(7, "Challenges Badge Ring", "Challenges Badge Icon"), (8, "Sampling Badge Ring", "Sampling Badge Icon"),
              (10, "Evaluation Badge Ring", "Evaluation Badge Icon")]
    for i, (badge, runs) in enumerate(zip(badges, texts)):
        card(slide, src, st, f"Error Card {i + 1}", 7.95, 2.1 + i * 1.55, 4.78, 1.4, badge, runs)
    notes(slide, "Vansh Sood", 35,
          f"These are the final model's validation errors after cleanup. Of {err['n_gt']:,} marked filaments, "
          f"{err['tp']} were found, {err['gt_partial']} were overlapped with the wrong outline, and {err['gt_none']} "
          f"were missed completely. The completely missed ones are small, median {err['median_none']:,} pixels "
          f"against {err['median_found']:,} for found ones, so fine-scale structure is our main weakness. The wrong "
          f"predictions split evenly between wrong shapes and false alarms in empty background. PQ is outline "
          f"quality {err['sq']:.2f} times detection quality {err['rq']:.2f}.",
          ["runs/person_c/compression/bench_pruned30_fp16/validation/summary.json",
           "runs/person_c/compression/bench_pruned30_fp16/validation/filaments.csv",
           "runs/person_c/compression/bench_pruned30_fp16/validation/overlap_pairs.csv",
           "runs/person_c/compression/bench_pruned30_fp16/validation/annotation_sets.csv"])
    return slide


def best_worst(slide, st, cases):
    s = by_name(slide)
    set_plain(s["Section Title"], "Results: Best and Worst Cases")
    set_plain(s["Grid Head 2"], "BEFORE CLEANUP")
    set_plain(s["Grid Head 3"], "FINAL MODEL")
    set_plain(s["Grid Row 1 Label"], cases["best"]["observation"])
    set_plain(s["Grid Row 2 Label"], cases["worst"]["observation"])
    for i, name in enumerate(["case_best_reference", "case_best_before", "case_best_after",
                              "case_worst_reference", "case_worst_before", "case_worst_after"], start=1):
        replace_image(s[f"Example Panel {i}"], FIG / f"{name}.png")
    for chip, text, width in (("2", f"PQ {cases['best']['before']:.2f}", 0.82),
                              ("3", f"PQ {cases['best']['after']:.2f} · best", 1.32),
                              ("5", f"PQ {cases['worst']['before']:.2f}", 0.82),
                              ("6", f"PQ {cases['worst']['after']:.2f} · worst", 1.38)):
        set_plain(s[f"Example Chip {chip} Text"], text)
        s[f"Example Chip {chip}"].width = s[f"Example Chip {chip} Text"].width = Inches(width)
    write(s["Examples Intro"], st, [[("Final model, labelled validation images: ", "strong", 14.5),
                                     ("highest and lowest image PQ.", "body", 14.5)]])
    set_plain(s["Outcome 1 Label"], "BEST CASE")
    set_plain(s["Outcome 1 Value"], f"PQ {cases['best']['after']:.2f}")
    set_plain(s["Outcome 1 Text"], "Clear, elongated filaments: all 4 found in both annotation sets; no extras.")
    set_plain(s["Outcome 2 Label"], "WORST CASE")
    set_plain(s["Outcome 2 Value"], f"PQ {cases['worst']['after']:.2f}")
    set_plain(s["Outcome 2 Text"], "Tiny, faint filaments (225–396 px) beside bright regions: scores ≈ 0, none found.")
    for name in ("Outcome 1 Text", "Outcome 2 Text"):
        s[name].height = Inches(0.5)
    write(s["Examples Legend"], st, [
        [("Colours = object IDs; red = removed by cleanup. Worst row is zoomed on one missed filament.", "grey", 11)],
        [("Best = highest PQ with ≥ 3 filaments per set; worst = lowest PQ, most filaments.", "muted", 10.5)]])
    place(s["Examples Legend"], y=5.98, h=0.86)
    notes(slide, "Denis Woolley", 35,
          "Best and worst validation images for the final model. Best: four clear, elongated filaments, all found "
          "in both annotation sets with no extras, PQ 0.66. We skipped single-filament images because they are too "
          "easy. Worst: tiny, faint filaments beside bright active regions. The model's scores there are near zero, "
          "so it found nothing, and three of them are below our 320-pixel cleanup limit anyway.",
          ["runs/person_c/compression/bench_pruned30_fp16/validation/observations.csv",
           "runs/improvements/deck/final_model_before_cleanup/observations.csv",
           "delivery/final_deck_figures.py", f"Cases: {json.dumps(cases)}"])


def what_else(prs, base, src, st):
    slide = new_slide(prs, base, "Segmentation Enhancement: What Else We Tried", "Denis Woolley")
    write(clone(src[14], "Diag Head", slide, "Else Head L", 0.6, 2.08, 5.75, 0.3), st,
          [[("POST-PROCESSING IDEAS · CHANGE IN PQ", "label", 10)]])
    rows_l = [("Keep weak pixels next to strong ones (hysteresis)", "+0.023", "value_teal"),
              ("Close small gaps (morphological closing)", "+0.014", "value_teal"),
              ("Change only the 0.5 pixel cutoff", "±0.000", "value"),
              ("Fill holes inside regions", "±0.000", "value"),
              ("Stricter 4-neighbour connections", "−0.002", "warn"),
              ("Split touching filaments (watershed)", "−0.002 to −0.010", "warn"),
              ("Average flipped copies (TTA), 3.8× slower", "−0.004", "warn")]
    write(clone(src[14], "Diag Head", slide, "Else Head R", 6.97, 2.08, 5.76, 0.3), st,
          [[("RETRAINED OR COMBINED MODELS · VALIDATION PQ", "label", 10)]])
    rows_r = [("Plain images (selected recipe, seed 45)", "0.4034", "value"),
              ("Contrast boost + outside-Sun mask (seed 46)", "0.3996", "warn"),
              ("Local brightness norm. + mask (seed 47)", "0.3808", "warn"),
              ("Average of 2 models (+ distilled ResNet18)", "0.4143", "value_teal"),
              ("Average of 4 models (incl. both above)", "0.4080", "value_teal")]
    for side, (x, rows, label_w, value_w) in enumerate(((0.6, rows_l, 4.25, 1.5), (6.97, rows_r, 4.36, 1.4))):
        clone(src[14], "Diag Rule 0", slide, f"Else {side} Rule 0", x, 2.44, label_w + value_w, 0)
        for i, (label, value, style) in enumerate(rows, start=1):
            y = 2.44 + (i - 1) * 0.44
            write(clone(src[14], "Diag 1 Label", slide, f"Else {side} Row {i} Label", x, y, label_w, 0.44), st,
                  [[(label, "row", 12)]])
            write(clone(src[14], "Diag 1 Value", slide, f"Else {side} Row {i} Value", x + label_w, y, value_w, 0.44),
                  st, [[(value, style, 13)]], align="r")
            clone(src[14], "Diag Rule 1", slide, f"Else {side} Rule {i}", x, y + 0.44, label_w + value_w, 0)
    write(clone(src[8], "Split Note", slide, "Else Note L", 0.6, 5.58, 5.75, 0.5), st,
          [[("Hysteresis and closing helped alone but added nothing meaningful on top of the selected pair.",
             "muted", 10.5)]])
    clone(src[14], "Gain Card", slide, "Else Verdict Card", 6.97, 4.78, 5.76, 1.32)
    write(clone(src[14], "Gain Text", slide, "Else Verdict Text", 7.17, 4.84, 5.4, 1.2), st, [
        [("Not adopted: ", "accent", 13), ("the 2-model gain (+0.011; 95% interval −0.001 to +0.025) is uncertain "
                                           "and needs a 6.5× larger file and 17% more time per image.", "body", 12)],
        [("Preprocessing models: single runs, different seeds.", "muted", 10.5)]])
    clone(src[14], "Check 1", slide, "Else Fine Icon", 0.6, 6.28, 0.24, 0.24)
    write(clone(src[14], "Check 1 Text", slide, "Else Fine Text", 0.96, 6.22, 11.7, 0.37), st, [[
        ("Fine-scale structure: ", "accent", 12),
        ("no dedicated method; the Tversky loss, which weights missed pixels more, was lowest in screening (0.242).",
         "grey", 12)]])
    notes(slide, "Denis Woolley", 35,
          "Other ideas did not make the final pipeline. Hysteresis and closing helped alone but added nothing on "
          "top of our two rules. Cutoff changes and hole filling did nothing; watershed splitting and averaging "
          "flipped predictions made PQ worse. Retraining with contrast enhancement or brightness normalisation did "
          "not beat plain images. Averaging two models reached 0.414, but the gain is uncertain and the file is 6.5 "
          "times larger, so we kept the compressed model.",
          ["results/person_c/postprocess_stage1.csv", "results/person_c/person_c_results.md",
           "results/improvements/README.md", "results/improvements/score_with_localnorm.json",
           "results/person_b_study/screening.csv"])
    return slide


def shorten_existing_notes(enhancement, compression):
    """Tighter narration for the two longest original slides (keeps their evidence lists)."""
    for slide, owner, seconds, narration in (
            (enhancement, "Denis Woolley", 45,
             "Post-processing changes the output after the network predicts. Using saved predictions, we first "
             "reproduced Vansh's score exactly. About half of the extra regions were under 320 pixels, against 4 "
             "percent of correct ones. So we give fragments within about 16 pixels the same ID, without adding "
             "pixels, then remove groups under 320 pixels. That roughly halves extra detections and raises PQ by "
             "0.059; the bootstrap interval and both held-out halves agree. The cost is 17 fewer matches and a few "
             "more merges."),
            (compression, "Denis Woolley", 45,
             "Pruning removes internal feature channels, then we fine-tune what is left. FP16 stores weights with "
             "16-bit numbers. Distillation trains a smaller student from the labels plus the teacher's predictions, "
             "and a label-only control shows its effect. 50 percent pruning lost quality, so we added a milder 30 "
             "percent version. Our rule picks the smallest GPU model within 0.002 PQ of the teacher: 30 percent "
             "pruning plus FP16, 22.9 megabytes instead of 93.4, at 475 milliseconds per image end to end on an RTX "
             "3070, including reading, prediction, cleanup and writing the CSV. The CPU-only INT8 version is a "
             "separate result.")):
        old = slide.notes_slide.notes_text_frame.text
        evidence = old.split("Evidence:\n", 1)[1].splitlines() if "Evidence:\n" in old else []
        notes(slide, owner, seconds, narration, evidence)


def conclusions(slide, st):
    s = by_name(slide)
    write(s["Conclusion 3 Text"], st, [[("Roadblocks: ", "warn"),
                                        ("tiny, faint filaments; labels that disagree (annotators: PQ 0.34); and "
                                         "local validation (≈0.40) scoring higher than Kaggle.", "body")]], size=14)
    write(s["Conclusion 4 Text"], st, [[("Do differently: ", "accent"),
                                        ("submit to Kaggle early to check our validation set; repeat final-model "
                                         "runs; try a loss for thin, connected shapes. Contrast enhancement was "
                                         "tested: no gain.", "body")]], size=14)
    notes(slide, "Denis Woolley / all", 30,
          "Our main lesson: the biggest gain came from understanding object-level errors and cleaning up "
          "predictions, not from a bigger network. Compression cut the file by about 75 percent with almost no "
          "loss. Roadblocks were tiny faint filaments, annotators who disagree, and a validation score higher than "
          "our Kaggle score. Next time we would submit to Kaggle earlier, repeat final-model runs and try losses "
          "designed for thin, connected shapes.",
          ["results/person_c/person_c_results.md", "results/improvements/README.md", "results/eda_summary.json"])


REFERENCES = [
    ("Kaggle (2026).", " Solar Filament Segmentation Challenge 2026: data, rules and evaluation.",
     "https://www.kaggle.com/competitions/filament-segmentation-2026"),
    ("Zuiderveld (1994).", " Contrast Limited Adaptive Histogram Equalization (CLAHE). Graphics Gems IV.", None),
    ("Buslaev et al. (2020).", " Albumentations: Fast and Flexible Image Augmentations.",
     "https://doi.org/10.3390/info11020125"),
    ("Paszke et al. (2019).", " PyTorch: An Imperative Style, High-Performance Deep Learning Library.",
     "https://arxiv.org/abs/1912.01703"),
    ("Lin et al. (2014).", " Microsoft COCO: Common Objects in Context (RLE masks, pycocotools).",
     "https://arxiv.org/abs/1405.0312"),
    ("Bradski (2000).", " The OpenCV Library. Dr. Dobb's Journal of Software Tools.", "https://opencv.org"),
]


def bibliography(prs, base, st):
    slide = new_slide(prs, base, "Bibliography: Data and Libraries", "Reference only")
    for i, (bold, rest, url) in enumerate(REFERENCES):
        number, dy = 13 + i, i * 0.72
        clone(base, "Ref 7 Ring", slide, f"Ref {number} Ring", y=2.28 + dy)
        write(clone(base, "Ref 7 Number", slide, f"Ref {number} Number", y=2.28 + dy), st,
              [[(str(number), "label", 13)]], align="ctr")
        set_plain(by_name(slide)[f"Ref {number} Number"], str(number))
        text = clone(base, "Ref 7 Text", slide, f"Ref {number} Text", y=2.15 + dy)
        write(text, st, [[(bold, "ref_bold"), (rest, "ref_grey")]])
        if url:
            link = clone(base, "Ref 7 Link 1", slide, f"Ref {number} Link 1", y=2.37 + dy)
            rid = slide.part.relate_to(url, RT.HYPERLINK, is_external=True)
            for hlink in link._element.iter(qn("a:hlinkClick")):
                old = hlink.get(qn("r:id"))
                hlink.set(qn("r:id"), rid)
                if old != rid and f'"{old}"' not in slide._element.xml:
                    slide.part.drop_rel(old)
        if i < len(REFERENCES) - 1:
            clone(base, "Ref 7 Rule", slide, f"Ref {number} Rule", y=2.87 + dy)
    notes(slide, "Reference only", 0, "References are provided for completeness; no narration required.",
          [u for *_, u in REFERENCES if u])
    return slide


def error_numbers():
    import pandas as pd

    v = ROOT / "runs" / "person_c" / "compression" / "bench_pruned30_fp16" / "validation"
    summary = json.loads((v / "summary.json").read_text(encoding="utf8"))
    f = pd.read_csv(v / "filaments.csv")
    pairs = pd.read_csv(v / "overlap_pairs.csv")
    sets = pd.read_csv(v / "annotation_sets.csv").set_index("name")
    tp_pred = pairs[pairs.iou > 0.5].groupby("annotation_set").pred_id.nunique().reindex(sets.index, fill_value=0)
    any_pred = pairs[pairs.iou > 0].groupby("annotation_set").pred_id.nunique().reindex(sets.index, fill_value=0)
    found, none = f[f.iou > 0.5], f[f.iou == 0]
    fp_overlap = int((any_pred - tp_pred).sum())
    result = dict(tp=summary["tp"], fp=summary["fp"], fn=summary["fn"], n_gt=len(f), n_pred=int(sets.n_pred.sum()),
                  gt_partial=int(((f.iou > 0) & (f.iou <= 0.5)).sum()), gt_none=len(none),
                  fp_overlap=fp_overlap, fp_pure=int(summary["fp"] - fp_overlap),
                  median_found=int(found.gt_area.median()), median_none=int(round(none.gt_area.median())),
                  found_lt500=float((found.gt_area < 500).mean()), none_lt500=float((none.gt_area < 500).mean()),
                  one_to_many=summary["custom_one_to_many"], many_to_one=summary["custom_many_to_one"],
                  pq=summary["PQ_dataset"], sq=summary["SQ_dataset"], rq=summary["RQ_dataset"])
    if result["tp"] + result["fn"] != result["n_gt"] or result["tp"] + result["fp"] != result["n_pred"]:
        raise ValueError(f"error counts do not reconcile: {result}")
    return result


def case_numbers(info):
    import pandas as pd

    after = pd.read_csv(ROOT / "runs/person_c/compression/bench_pruned30_fp16/validation/observations.csv")
    before = pd.read_csv(ROOT / "runs/improvements/deck/final_model_before_cleanup/observations.csv")
    pq = lambda df, obs: float(df.loc[df.observation == obs, "PQ"].iloc[0])
    return {role: dict(observation=c["observation"], before=pq(before, c["observation"]),
                       after=pq(after, c["observation"])) for role, c in info["cases"].items()}


def build(source, output):
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new --output name")
    info = json.loads((FIG / "figures.json").read_text(encoding="utf8"))
    prs = Presentation(source)
    if len(prs.slides) != 18:
        raise ValueError("expected the 18-slide final presentation")
    st = Styles(prs)
    original = list(prs.slides)
    src = {i + 1: slide for i, slide in enumerate(original)}
    changed = fix_student_number(prs)
    our_choices(src[8], st)
    setup(src[9], st)
    best_worst(src[12], st, case_numbers(info))
    conclusions(src[16], st)
    shorten_existing_notes(src[14], src[15])
    n1 = noisy_labels(prs, src[8], src, st, info["noisy"])
    n2 = tuning_results(prs, src[10], src, st)
    n3 = kinds_of_errors(prs, src[11], src, st, error_numbers())
    n5 = what_else(prs, src[14], src, st)
    n6 = bibliography(prs, src[18], st)
    order = original[:8] + [n1] + original[8:10] + [n2, original[10], n3] + original[11:14] + [n5] + original[14:] + [n6]
    entries = {slide.part: entry for slide, entry in zip(prs.slides, list(prs.slides._sldIdLst))}
    for entry in list(prs.slides._sldIdLst):
        prs.slides._sldIdLst.remove(entry)
    for slide in order:
        prs.slides._sldIdLst.append(entries[slide.part])
    prs.save(output)
    print(f"saved {output}: {len(order)} slides; student-number runs fixed: {changed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    build(args.input, args.output)


if __name__ == "__main__":
    main()
