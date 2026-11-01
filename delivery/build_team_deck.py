"""Build Person C's slides and assemble the team presentation (python-pptx).

Inputs (all recorded artifacts, nothing is retrained or re-scored):
  * Person B's finished deck: delivery/person_b_results.pptx (left unchanged)
  * Person A's EDA: results/eda_summary.json and figures/eda_*.png
  * Person C's results: results/person_c/*.json|csv and figures/person_c/*.png

Outputs:
  delivery/person_c_enhancement.pptx          Person C section only
  delivery/COS711_A2_team_presentation.pptx   title/roles + A + B + C + references
  delivery/person_c_speaker_notes.md          narration for the new slides

New slides copy Person B's style (16:9, Arial, navy titles, "Person X" footer)
and keep the upper-right corner empty for the speaker's camera. Run from the
repository root:  python delivery/build_team_deck.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, GREY = RGBColor(0x14, 0x2C, 0x45), RGBColor(0x17, 0x6C, 0x83), RGBColor(0x4C, 0x5B, 0x66)
LIGHT, WHITE, RED = RGBColor(0xEE, 0xF3, 0xF5), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xB5, 0x3A, 0x2A)
FONT = "Arial"


# --------------------------------------------------------------------------- #
# Small drawing helpers
# --------------------------------------------------------------------------- #
def _style(run, size, color=GREY, bold=False):
    run.font.name, run.font.size, run.font.bold = FONT, Pt(size), bold
    run.font.color.rgb = color


def text(slide, x, y, w, h, paragraphs, size=18, color=GREY, bold=False, spacing=6):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    for i, p in enumerate(paragraphs):
        p = dict(text=p) if isinstance(p, str) else p
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        run = para.add_run()
        run.text = ("\u2022 " if p.get("bullet") else "") + p["text"]
        _style(run, p.get("size", size), p.get("color", color), p.get("bold", bold))
        para.space_after = Pt(p.get("spacing", spacing))
    return box


def table(slide, x, y, w, rows, widths, row_h=0.42, size=14, header=True):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    grid = shape.table
    for j, width in enumerate(widths):
        grid.columns[j].width = Inches(width)
    for i, row in enumerate(rows):
        grid.rows[i].height = Inches(row_h)
        for j, value in enumerate(row):
            cell = grid.cell(i, j)
            cell.text = ""
            run = cell.text_frame.paragraphs[0].add_run()
            run.text = str(value)
            strong = (header and i == 0) or j == 0
            _style(run, size, NAVY if strong else GREY, strong)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if strong else WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
    return shape


def picture(slide, path, x, y, w=None, h=None, crop_bottom=0.0):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                   Inches(w) if w else None, Inches(h) if h else None)
    if crop_bottom:
        pic.crop_bottom = crop_bottom
        pic.height = int(pic.height * (1 - crop_bottom))
    return pic


def trimmed(path, pad=6) -> Path:
    """Copy of a figure without its white margins (build-time only, under delivery/.build)."""
    from PIL import Image, ImageChops

    image = Image.open(path).convert("RGB")
    box = ImageChops.difference(image, Image.new("RGB", image.size, (255, 255, 255))).getbbox()
    if box:
        box = (max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, image.width), min(box[3] + pad, image.height))
        image = image.crop(box)
    out = ROOT / "delivery" / ".build" / "trimmed" / Path(path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def aspect(path) -> float:
    from PIL import Image
    with Image.open(path) as image:
        return image.width / image.height


def new_slide(prs, title, owner, notes):
    layout = next(l for l in prs.slide_layouts if l.name == "Blank")
    slide = prs.slides.add_slide(layout)
    # Title stops at 9.6 in: the upper-right corner is reserved for the camera.
    text(slide, 0.75, 0.48, 8.85, 0.79, [dict(text=title, size=32, color=NAVY, bold=True)])
    text(slide, 0.75, 7.02, 9.58, 0.3, [dict(text=owner, size=16)])
    text(slide, 12.08, 7.02, 0.5, 0.3, [dict(text="0", size=15)])
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def renumber(prs):
    for number, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if (shape.has_text_frame and shape.text_frame.text.strip().isdigit()
                    and abs(shape.left - Inches(12.08)) < Inches(0.15)):
                runs = shape.text_frame.paragraphs[0].runs
                if runs:
                    runs[0].text = str(number)
                    for extra in runs[1:]:
                        extra.text = ""


def delete_all_slides(prs):
    ids = prs.slides._sldIdLst
    for entry in list(ids):
        prs.part.drop_rel(entry.rId)
        ids.remove(entry)


def reorder(prs, order):
    ids = prs.slides._sldIdLst
    entries = list(ids)
    for entry in entries:
        ids.remove(entry)
    for index in order:
        ids.append(entries[index])


# --------------------------------------------------------------------------- #
# Recorded evidence
# --------------------------------------------------------------------------- #
def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


class Evidence:
    def __init__(self, root: Path):
        self.c = load(root / "results/person_c/summary.json")
        self.eda = load(root / "results/eda_summary.json")
        self.b_run = load(root / "runs/person_b_final/summary.json") if (root / "runs/person_b_final/summary.json").is_file() else None
        final = root / "results/person_c/final_submission.json"
        self.final = load(final) if final.is_file() else None
        self.stage1 = list(csv.DictReader((root / "results/person_c/postprocess_stage1.csv").open(encoding="utf8")))
        self.compression = self.c.get("compression") or []
        self.fig = root / "figures/person_c"
        self.eda_fig = root / "figures"

    def family_best(self, family):
        rows = [r for r in self.stage1 if r["family"] == family]
        return max(rows, key=lambda r: float(r["PQ_dataset"]))

    def family_worst(self, family):
        rows = [r for r in self.stage1 if r["family"] == family]
        return min(rows, key=lambda r: float(r["PQ_dataset"]))


# --------------------------------------------------------------------------- #
# Slides
# --------------------------------------------------------------------------- #
def title_slide(prs, ev):
    s = new_slide(prs, "Solar filament segmentation", "COS711 Assignment 2",
                  "Each member introduces themselves and states their responsibility in one sentence. "
                  "Replace 'Person A/B/C' with names before recording.")
    text(s, 0.75, 1.35, 8.8, 0.9, [dict(text="Segmenting solar filaments with deep learning: Kaggle Solar Filament "
                                             "Segmentation Challenge 2026 (MAGFiLO, GONG H-alpha images)",
                                        size=19, color=TEAL)])
    table(s, 0.75, 2.55, 11.8, [
        ["Team member", "Responsibility", "Code"],
        ["Person A", "Data and evaluation: EDA, cleaning, observation-grouped time-block split, augmentation, local PQ metric",
         "data.py, metrics.py, eda.ipynb"],
        ["Person B", "Modelling: U-Net/ResNet34, losses, training loop, controlled comparisons, repeated seeds",
         "models.py, losses.py, train.py, experiments.py"],
        ["Person C", "Instances and delivery: post-processing, instance separation, compression, timing, submission, slides and video",
         "postprocess.py, offline_eval.py, compress.py, predict.py"],
    ], [1.9, 6.6, 3.3], row_h=0.78, size=15)
    text(s, 0.75, 5.95, 11.8, 0.8, [dict(text="Primary metric: Panoptic Quality (PQ), an instance-level score. Every number in "
                                             "this talk comes from the same frozen validation split (122 images, 200 annotation sets).",
                                        size=16)])


def dataset_slides(prs, ev):
    e = ev.eda
    agreement = e["annotator_agreement"]
    s = new_slide(prs, "Dataset: what we found", "Person A",
                  "We have 707 full-disc H-alpha images with one to three independent annotation sets each. "
                  "Filaments cover about 0.36 percent of the pixels, so background outnumbers filament roughly 278 to 1. "
                  "Annotators disagree a lot: scoring one annotator against another gives a PQ of only about 0.34 "
                  "on a sample of 100 images, which tells us the labels are noisy.")
    sets = e["sets_per_observation"]
    text(s, 0.75, 1.45, 6.2, 5.3, [
        dict(text=f"{e['n_observations']} images (2048 x 2048), {e['n_image_entries']:,} annotation sets, "
                  f"{e['n_annotations']:,} filaments", bullet=True),
        dict(text=f"1 / 2 / 3 annotation sets per image: {sets['1']} / {sets['2']} / {sets['3']}", bullet=True),
        dict(text=f"{e['filaments_per_set']['mean']:.1f} filaments per set on average (max {int(e['filaments_per_set']['max'])})",
             bullet=True),
        dict(text=f"Filaments cover {100 * e['fg_frac_mean']:.2f}% of pixels: background/filament about "
                  f"{e['bg_fg_ratio']:.0f} : 1", bullet=True),
        dict(text="6 GONG instruments, 2011 to 2022; test file names have a similar mix", bullet=True),
        dict(text=f"Annotator vs annotator PQ {agreement['PQ']:.2f} (sample of {e['annotator_agreement_n_observations']} "
                  "images): noisy labels, a reference not a ceiling", bullet=True),
        dict(text=f"{e['overlap_sets']} sets with overlapping polygons; original layers kept for scoring", bullet=True),
    ], size=17)
    picture(s, ev.eda_fig / "eda_imbalance.png", 7.1, 2.4, w=5.5)

    s = new_slide(prs, "Noisy, faint and disagreeing images", "Person A",
                  "Looking at images by eye showed three difficult groups: noisy observing conditions, very faint "
                  "filaments, and images where annotators drew different filaments. These motivated the use of "
                  "every annotation set as a separate training target and the robust evaluation over all sets.")
    y = 1.4
    for name, label in (("eda_samples_noisy.png", "Noisy observing conditions"),
                        ("eda_samples_faint.png", "Faint filaments"),
                        ("eda_samples_disagreeing.png", "Annotators disagree (colours = different annotators)")):
        strip = trimmed(ev.eda_fig / name)
        height = min(1.48, 11.8 / aspect(strip))
        text(s, 0.75, y, 9.0, 0.3, [dict(text=label, size=14, color=TEAL, bold=True)])
        picture(s, strip, 0.75, y + 0.33, h=height)
        y += 0.36 + height + 0.06

    split = e["split"]
    s = new_slide(prs, "Preparation and validation split", "Person A",
                  "All annotation sets of an image stay on the same side of the split. Because images close in time "
                  "are not independent, validation uses whole 14-day blocks with a one-day embargo, and the block "
                  "choice matches the instrument mix of the test file names. Preprocessing and augmentation options "
                  "were built here and screened by Person B.")
    text(s, 0.75, 1.45, 8.8, 2.6, [
        dict(text=f"Split by observation in 14-day time blocks, 1-day embargo: {split['n_train']} train / "
                  f"{split['n_val']} validation / {split['n_dropped']} dropped", bullet=True),
        dict(text="Validation blocks chosen to match the instrument mix of the test file names", bullet=True),
        dict(text="Cached instance masks; deterministic seeded crops (70% centred on filaments)", bullet=True),
        dict(text="Preprocessing built: disc mask, CLAHE, local normalisation; augmentation: flips, rotations, "
                  "affine, brightness/contrast, noise, blur (screened in Person B's study)", bullet=True),
    ], size=17)
    prep = trimmed(ev.eda_fig / "eda_preprocessing.png")
    timeline = trimmed(ev.eda_fig / "eda_split_timeline.png")
    prep_h = min(2.05, 11.8 / aspect(prep))
    picture(s, prep, 0.75, 3.55, h=prep_h)
    picture(s, timeline, 0.75, 3.65 + prep_h, h=min(1.25, 6.9 - 3.65 - prep_h))


def person_c_slides(prs, ev):
    c, base, sel = ev.c, ev.c["baseline"], ev.c["selected"]
    areas = c["component_areas"]
    boot = c["selected_bootstrap"]
    halves = c["split_half_selection"]
    gains = sorted(v["held_out_gain"] for v in halves.values())
    notes = {}

    title = "Where the errors were"
    notes[title] = (
        "I took the seed-45 probabilities from Person B and first asked which errors cost PQ. Pooled over the 200 "
        f"validation annotation sets there were {base['tp']} matched filaments, {base['fp']:,} extra predictions and "
        f"{base['fn']} misses, so extra detections outnumber misses two to one. The histogram shows why: "
        f"{areas['unmatched_below_min_area']:.0%} of the unmatched predicted regions are smaller than "
        f"{areas['min_area']} pixels, while only {areas['matched_below_min_area']:.0%} of matched regions and "
        f"{areas['reference_below_min_area']:.0%} of human-drawn filaments are that small. Fragmentation was "
        f"{base['custom_one_to_many']} cases against {base['custom_many_to_one']} merges.")
    s = new_slide(prs, title, "Person C", notes[title])
    text(s, 0.75, 1.4, 8.8, 1.9, [
        dict(text=f"Baseline conversion (cutoff 0.5, 8-connected regions): PQ {base['PQ_dataset']:.3f} = "
                  f"{base['tp']} matched, {base['fp']:,} extra, {base['fn']} missed", bullet=True),
        dict(text=f"Extra detections outnumber misses 2 : 1; fragmentation ({base['custom_one_to_many']}) "
                  f"is 4x more common than merging ({base['custom_many_to_one']})", bullet=True),
        dict(text=f"{areas['unmatched_below_min_area']:.0%} of unmatched regions are < {areas['min_area']} px, "
                  f"vs {areas['matched_below_min_area']:.0%} of matched and {areas['reference_below_min_area']:.0%} "
                  "of human filaments", bullet=True, color=TEAL, bold=True),
    ], size=17)
    picture(s, ev.fig / "pp_component_areas.png", 1.2, 3.35, h=3.6)

    title = "Post-processing study"
    notes[title] = (
        "Because the probabilities were saved, I built an offline runner that re-scores them with exactly the same "
        "evaluator. It first reproduced Person B's PQ of 0.3448101728 bit for bit. I then declared a grid of "
        f"{c['stage1_candidates'] - 1} settings, changing one thing at a time. Removing small regions and bridging "
        "nearby fragments helped most; hysteresis and closing helped less. Changing the pixel cutoff alone did "
        "nothing, which tells us the misses are not just faint pixels below 0.5. Splitting regions at confident "
        "cores with a watershed made things worse. The error bars are 95 percent bootstrap intervals over images.")
    s = new_slide(prs, title, "Person C", notes[title])
    picture(s, ev.fig / "pp_families.png", 0.6, 1.35, h=5.55)
    text(s, 10.55, 2.45, 2.1, 4.5, [
        dict(text="Same probabilities, same scorer", size=14, color=TEAL, bold=True),
        dict(text="Offline runner reproduces Person B's PQ exactly (0.3448101728)", size=13),
        dict(text=f"Declared grids: {c['stage1_candidates'] - 1} + {c['stage2_candidates'] - 1} settings", size=13),
        dict(text="Error bars: 95% bootstrap over validation images", size=13),
    ])

    title = "Selected post-processing"
    notes[title] = (
        "The selected rule has two steps. First, fragments separated by at most 16 pixels get the same instance "
        "ID, without adding any pixels. Then regions smaller than 320 pixels are removed. PQ rises from "
        f"{base['PQ_dataset']:.3f} to {sel['PQ_dataset']:.3f}. To check that we did not just fit the validation "
        "set, I tuned on one half of the validation time blocks and tested on the other half: the gain held at "
        f"{gains[0]:+.3f} and {gains[1]:+.3f}. Extra predictions halve and fragmentation drops, at a cost of "
        f"{base['tp'] - sel['tp']} matched small filaments and a few more merges.")
    s = new_slide(prs, title, "Person C", notes[title])
    text(s, 0.75, 1.35, 8.8, 1.95, [
        dict(text="Bridge fragments up to 16 px apart (one ID, no pixels added), then drop regions < 320 px",
             size=16),
        dict(text=f"PQ {base['PQ_dataset']:.3f} \u2192 {sel['PQ_dataset']:.3f}  "
                  f"(+{boot['delta_PQ']:.3f}; 95% CI +{boot['delta_ci95_low']:.3f} to +{boot['delta_ci95_high']:.3f})",
             size=22, color=TEAL, bold=True),
        dict(text=f"Tuned on half of the validation blocks, tested on the other half: {gains[0]:+.3f} and "
                  f"{gains[1]:+.3f}", size=16),
    ])
    picture(s, ev.fig / "pp_error_counts.png", 0.75, 3.45, h=3.45)
    text(s, 9.55, 3.7, 3.1, 3.0, [
        dict(text="Trade-off", size=15, color=TEAL, bold=True),
        dict(text=f"Extra regions {base['fp']:,} \u2192 {sel['fp']}", size=14),
        dict(text=f"Fragmented {base['custom_one_to_many']} \u2192 {sel['custom_one_to_many']}", size=14),
        dict(text=f"Lost matches: {base['tp'] - sel['tp']} small filaments", size=14, color=RED),
        dict(text=f"Merges {base['custom_many_to_one']} \u2192 {sel['custom_many_to_one']}", size=14, color=RED),
    ])

    title = "Best and worst cases"
    ex = {}
    for example in c["examples"]:  # ranked: keep the top example of each role, as in the figure
        ex.setdefault(example["role"], example)
    gain, loss = ex.get("largest gain"), ex.get("largest loss")
    notes[title] = (
        "These are the validation images that changed most. In the best case, a tiny partial detection of a real "
        "filament was removed: it never reached the IoU of 0.5 needed for a match, so it only counted as an extra "
        "object. In the worst case, a small real filament below 320 pixels was removed, which is the price of the "
        "size filter. Bridged fragments appear as one colour.")
    s = new_slide(prs, title, "Person C", notes[title])
    picture(s, ev.fig / "pp_examples_slide.png", 0.6, 1.3, h=5.6)
    text(s, 9.25, 2.5, 3.4, 4.4, [
        dict(text="Best case", size=15, color=TEAL, bold=True),
        dict(text=(f"PQ {gain['baseline_PQ']:.2f} \u2192 {gain['selected_PQ']:.2f}: a tiny partial detection "
                   "(IoU < 0.5) only counted as an extra object") if gain else "", size=13),
        dict(text="Worst case", size=15, color=RED, bold=True),
        dict(text=(f"PQ {loss['baseline_PQ']:.2f} \u2192 {loss['selected_PQ']:.2f}: a real filament "
                   "smaller than 320 px was removed") if loss else "", size=13),
        dict(text="Red = removed region; colours = instance IDs", size=12),
    ])

    title = "What did not help"
    t = ev.family_best("threshold")
    split_best, split_worst = ev.family_best("core_split"), ev.family_worst("core_split")
    tta = {r["name"]: r for r in c.get("tta_checks", {}).get("pp_check_tta", [])}
    notta = {r["name"]: r for r in c.get("tta_checks", {}).get("pp_check_no_tta_same_gpu", [])}
    tta_delta = (float(tta["selected"]["PQ_dataset"]) - float(notta["selected"]["PQ_dataset"])
                 if "selected" in tta and "selected" in notta else None)
    tta_best = max((float(r["PQ_dataset"]) for r in tta.values()), default=None)
    notta_best = max((float(r["PQ_dataset"]) for r in notta.values()), default=None)
    notes[title] = (
        "Negative results matter too. Moving the pixel cutoff between 0.3 and 0.7 never beat 0.5. Watershed "
        "splitting at confident cores made PQ worse because single filaments have uneven confidence and get cut "
        "into pieces, and merges were rare anyway. Flip test-time augmentation lowered PQ at our selected setting "
        "and, even after re-tuning the cutoff, gained almost nothing while costing nearly four times the model time. "
        "Filling holes and four-connectivity did not help either.")
    s = new_slide(prs, title, "Person C", notes[title])
    rows = [["Idea", "Change in PQ", "Why"],
            ["Pixel cutoff 0.3 to 0.7 (alone)", f"best {float(t['delta_PQ_dataset']):+.3f}",
             "Misses are not just faint pixels below 0.5"],
            ["Watershed split at confident cores", f"{float(split_best['delta_PQ_dataset']):+.3f} to "
                                                   f"{float(split_worst['delta_PQ_dataset']):+.3f}",
             f"Cuts single filaments; merges were rare ({base['custom_many_to_one']})"],
            ["Flip test-time augmentation", f"{tta_delta:+.3f}" if tta_delta is not None else "n/a",
             (f"3.8x model time; best re-tuned cutoff only {tta_best - notta_best:+.4f}"
              if tta_best is not None and notta_best is not None else "3.8x model time")],
            ["Fill holes", f"{float(ev.family_best('fill_holes')['delta_PQ_dataset']):+.3f}", "Predicted regions have no holes"],
            ["4-connectivity", f"{float(ev.family_best('connectivity')['delta_PQ_dataset']):+.3f}",
             "Breaks thin diagonal filaments"]]
    table(s, 0.75, 1.75, 11.8, rows, [4.0, 2.1, 5.7], row_h=0.62, size=15)
    text(s, 0.75, 5.75, 11.8, 0.8, [dict(text="Hysteresis (+0.023) and closing (+0.014) helped alone but were "
                                              "subsumed by the selected pair.", size=16, color=TEAL)])

    comp = [r for r in ev.compression if r.get("selected_PQ") is not None and r.get("device") != "CPU"]
    cpu = [r for r in ev.compression if r.get("device") == "CPU" and r.get("ms_per_image")]
    title = "Compression"
    by = {r["label"]: r for r in ev.compression}
    teacher = by.get("Teacher mixed precision (B default)")
    p30, p50 = by.get("Pruned 30% channels + fine-tune"), by.get("Pruned 50% channels + fine-tune")
    kd, ctrl = by.get("KD student ResNet18"), by.get("Label-only ResNet18 (control)")
    int8 = by.get("CPU: ONNX Runtime INT8")
    sentences = ["For compression I compared every variant on the same validation split, the deployable file size, "
                 "and the complete test pipeline time from reading the image to writing the CSV, on the same RTX 3070."]
    if teacher and by.get("Teacher FP16 weights"):
        sentences.append("Storing the weights in half precision halves the file with no measurable loss in PQ.")
    if teacher and p30 and p50:
        sentences.append(f"Removing 30 percent of the channels in every layer and fine-tuning for ten epochs kept PQ at "
                         f"{p30['selected_PQ']:.3f} against {teacher['selected_PQ']:.3f} with half the parameters; "
                         f"removing 50 percent was too destructive for the same short fine-tune "
                         f"({p50['selected_PQ']:.3f}).")
    if kd and ctrl:
        sentences.append(f"The distilled ResNet18 student scored {kd['selected_PQ']:.3f} and the label-only control "
                         f"{ctrl['selected_PQ']:.3f}; before post-processing the order was reversed, so with one run "
                         "each, distillation's effect is within run-to-run noise.")
    if teacher and teacher.get("validation_model_ms_per_image"):
        sentences.append(f"The network itself takes only about {teacher['validation_model_ms_per_image']:.0f} of the "
                         f"{teacher['ms_per_image']:.0f} milliseconds per image: reading the image, post-processing "
                         "and checking the encoding dominate, so a smaller network barely changes total time.")
    if int8:
        sentences.append("For CPU-only deployment, ONNX Runtime with INT8 weights is four times smaller and faster "
                         "than PyTorch, without losing PQ.")
    notes[title] = " ".join(sentences)
    s = new_slide(prs, title, "Person C", notes[title])
    rows = [["Variant", "Params", "File", "Val PQ", "ms / image", "Training"]]
    for r in comp:
        train = f"{r['training_seconds'] / 60:.0f} min" if r.get("training_seconds") else "-"
        rows.append([r["label"], f"{r['parameters'] / 1e6:.1f} M", f"{r['file_mb']:.0f} MB",
                     f"{r['selected_PQ']:.4f}", f"{r['ms_per_image']:.0f}" if r.get("ms_per_image") else "-", train])
    if len(rows) > 1:
        table(s, 0.75, 2.15, 11.8, rows, [4.3, 1.3, 1.3, 1.5, 1.7, 1.7], row_h=0.37, size=12)
    text(s, 0.75, 1.4, 8.8, 0.7, [dict(text="Same validation split, selected post-processing, complete image-to-CSV "
                                             "timing on one RTX 3070 (batch 1, model loading excluded)", size=14)])
    below = 2.3 + 0.37 * len(rows)
    if cpu:
        parts = [f"{r['label'].replace('CPU: ', '')} {r['ms_per_image'] / 1000:.1f} s"
                 + (f", PQ {r['selected_PQ']:.3f}" if r.get("selected_PQ") is not None else "")
                 + f", {r['file_mb']:.0f} MB" for r in cpu]
        text(s, 0.75, below, 11.8, 0.6, [dict(text=f"CPU deployment (first {cpu[0]['test_images_timed']} test images, "
                                                  "per image): " + "; ".join(parts), size=13)])
        below += 0.6
    if ev.final:
        text(s, 0.75, below, 11.8, 0.9, [dict(text=ev.final["compression_conclusion"], size=13,
                                               color=TEAL, bold=True)])

    title = "Final pipeline and submission"
    f = ev.final or {}
    notes[title] = (
        "The final pipeline runs whole-image inference, the selected post-processing, and checked RLE encoding. "
        + (f"It produced {f.get('n_filaments')} filaments on the {f.get('n_images')} test images. " if f else "")
        + "Model size, training time and end-to-end time are on the slide. The images are unlabelled test "
        "predictions, shown as illustrations only. The Kaggle score is added after the upload.")
    s = new_slide(prs, title, "Person C", notes[title])
    b = ev.b_run or {}
    facts = [["Item", "Value"],
             ["Final model", f.get("model_label", "U-Net / ResNet34, seed 45")],
             ["Size", f.get("size_text", f"{b.get('parameters', 0) / 1e6:.1f} M parameters")],
             ["Training time", f.get("training_text", f"{b.get('training_seconds', 0) / 60:.1f} min training loop, "
                                                       "20 epochs, RTX 5060 Ti")],
             ["End-to-end inference", f.get("timing_text", "see compression table")],
             ["Validation PQ", f.get("pq_text", f"{sel['PQ_dataset']:.3f} (selected post-processing)")],
             ["Test output", f"{f.get('n_filaments', '?')} filaments, {f.get('n_images', '?')} images, "
                             f"{len(f.get('zero_detection_images', []))} empty"],
             ["Kaggle PQ", "add after upload (not yet submitted)"]]
    table(s, 0.75, 1.45, 5.5, facts, [1.75, 3.75], row_h=0.66, size=13)
    picture(s, ev.fig / "test_examples.png", 6.45, 2.3, w=6.2)

    title = "Conclusions"
    final_size = (ev.final or {}).get("size_text", "")
    notes[title] = (
        "Our main take-away is that turning probabilities into separate objects mattered as much as the network: "
        "post-processing alone added more PQ than the whole training-configuration study. Errors were dominated by "
        "small extra detections, and a data-driven size filter plus fragment bridging targeted them. For efficiency, "
        "pruning and half-precision storage shrank the model a lot at the same validation PQ, but the network is only "
        "a small part of the end-to-end time. Our biggest roadblocks were tuning many settings on one validation split, "
        "which we handled with declared grids, split-half and bootstrap checks, and limited GPU time, so compression "
        "runs used one seed and aggressive pruning needed more fine-tuning than we had. Next time we would train for "
        "instances directly, for example with a boundary or skeleton target, prune gradually, repeat compression with "
        "several seeds, and submit to Kaggle early to compare local and leaderboard scores.")
    s = new_slide(prs, title, "Person C, then all", notes[title])
    text(s, 0.75, 1.4, 8.8, 5.5, [
        dict(text="Take-aways", size=19, color=TEAL, bold=True),
        dict(text=f"Instance conversion mattered as much as the network: +{boot['delta_PQ']:.3f} PQ from "
                  "post-processing alone, more than the training-configuration study", bullet=True, size=16),
        dict(text="Small extra detections, not misses, dominated the errors; let the data choose the filter",
             bullet=True, size=16),
        dict(text="Pruning + FP16 storage kept validation PQ at a fraction of the size"
                  + (f" ({final_size})" if final_size else "") + "; the network is a small part of end-to-end time",
             bullet=True, size=16),
        dict(text="Labels are noisy: annotator vs annotator PQ is about 0.34; validation is a reference, not a "
                  "leaderboard guarantee", bullet=True, size=16),
        dict(text="Roadblocks", size=19, color=TEAL, bold=True),
        dict(text="Many settings, one validation split: declared grids, split-half and bootstrap checks", bullet=True, size=16),
        dict(text="Limited GPU time: single-seed compression runs; 50% pruning needed longer fine-tuning", bullet=True, size=16),
        dict(text="Next time", size=19, color=TEAL, bold=True),
        dict(text="Train for instances (boundary or skeleton targets) instead of repairing them afterwards; prune "
                  "gradually; repeat compression with more seeds; submit to Kaggle early", bullet=True, size=16),
    ], spacing=4)
    return notes


def person_c_references(prs):
    s = new_slide(prs, "Enhancement references", "Person C",
                  "Bibliography slide; not spoken.")
    refs = [
        "Kirillov et al. (2019). Panoptic Segmentation. CVPR. https://doi.org/10.1109/CVPR.2019.00963",
        "Canny (1986). A Computational Approach to Edge Detection (hysteresis thresholding). IEEE TPAMI.",
        "Beucher & Meyer (1993). The morphological approach to segmentation: the watershed transformation.",
        "van der Walt et al. (2014). scikit-image: image processing in Python. PeerJ. https://doi.org/10.7717/peerj.453",
        "Hinton, Vinyals & Dean (2015). Distilling the Knowledge in a Neural Network. https://arxiv.org/abs/1503.02531",
        "Fang et al. (2023). DepGraph: Towards Any Structural Pruning. CVPR. https://arxiv.org/abs/2301.12900",
        "Li et al. (2017). Pruning Filters for Efficient ConvNets (L1 channel importance). https://arxiv.org/abs/1608.08710",
        "Micikevicius et al. (2018). Mixed Precision Training. ICLR. https://arxiv.org/abs/1710.03740",
        "Efron & Tibshirani (1993). An Introduction to the Bootstrap. Chapman & Hall.",
        "Iakubovskii. segmentation_models_pytorch. https://github.com/qubvel-org/segmentation_models.pytorch",
    ]
    text(s, 0.75, 1.45, 11.8, 5.4, [dict(text=r, size=14) for r in refs], spacing=8)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-deck", default=str(ROOT / "delivery/person_b_results.pptx"))
    parser.add_argument("--out-dir", default=str(ROOT / "delivery"))
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    ev = Evidence(ROOT)

    # Person C section on its own (Person B's deck is only used as a style template).
    prs = Presentation(args.b_deck)
    delete_all_slides(prs)
    notes = person_c_slides(prs, ev)
    person_c_references(prs)
    renumber(prs)
    prs.save(out / "person_c_enhancement.pptx")

    # Team deck: title + A + B (spoken) + C + references (B then C).
    prs = Presentation(args.b_deck)
    n_b = len(prs.slides)
    title_slide(prs, ev)
    dataset_slides(prs, ev)
    person_c_slides(prs, ev)
    person_c_references(prs)
    n_total = len(prs.slides)
    b_spoken, b_refs = list(range(0, n_b - 2)), [n_b - 2, n_b - 1]
    title_index, a_slides = n_b, list(range(n_b + 1, n_b + 4))
    c_slides = list(range(n_b + 4, n_total - 1))
    order = [title_index] + a_slides + b_spoken + c_slides + b_refs + [n_total - 1]
    reorder(prs, order)
    renumber(prs)
    prs.save(out / "COS711_A2_team_presentation.pptx")

    lines = ["# Person C speaker notes", "",
             "Narration for the Person C slides (also embedded in the PowerPoint notes). About 2.5 minutes for "
             "enhancement and compression, then 30 seconds of conclusions shared by the team.", ""]
    for title, note in notes.items():
        lines += [f"## {title}", "", note, ""]
    (out / "person_c_speaker_notes.md").write_text("\n".join(lines), encoding="utf8")
    print(f"wrote {out / 'person_c_enhancement.pptx'} and {out / 'COS711_A2_team_presentation.pptx'} "
          f"({n_total} slides)")


if __name__ == "__main__":
    main()
