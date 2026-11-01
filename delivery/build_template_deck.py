"""Populate the supplied university template using recorded team evidence."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
BLUE = RGBColor(0x00, 0x5B, 0xAA)
LIGHT = RGBColor(0xEA, 0xF2, 0xFA)
WHITE = RGBColor(255, 255, 255)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def clone_slide(prs, source):
    slide = prs.slides.add_slide(source.slide_layout)
    for shape in list(slide.shapes):
        slide.shapes._spTree.remove(shape._element)
    for shape in source.shapes:
        element = deepcopy(shape._element)
        for node in element.iter():
            for attribute, value in list(node.attrib.items()):
                if attribute.startswith("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"):
                    relation = source.part.rels[value]
                    node.set(attribute, slide.part.relate_to(
                        relation.target_ref if relation.is_external else relation.target_part,
                        relation.reltype, relation.is_external))
        slide.shapes._spTree.insert_element_before(element, "p:extLst")
    return slide


def body(slide):
    return slide.placeholders[1]


def fill(shape, paragraphs, size=21, bullet=True):
    frame = shape.text_frame
    original = frame.paragraphs
    example = original[min(1, len(original) - 1)] if bullet else original[0]
    properties = deepcopy(example._p.pPr)
    frame.clear()
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    for index, text in enumerate(paragraphs):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        if para._p.pPr is not None:
            para._p.remove(para._p.pPr)
        if properties is not None:
            para._p.insert(0, deepcopy(properties))
        if not bullet:
            for node in list(para._p.get_or_add_pPr()):
                if node.tag.rsplit("}", 1)[-1].startswith("bu"):
                    para._p.pPr.remove(node)
            para._p.pPr.append(OxmlElement("a:buNone"))
        para.space_before = Pt(0)
        para.space_after = Pt(12)
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.color.rgb = BLUE


def box(slide, x, y, w, h, lines, size=18, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    fill(shape, lines, size, bullet=False)
    for para in shape.text_frame.paragraphs:
        para.space_after = Pt(6)
        for run in para.runs:
            run.font.bold = bold
    return shape


def title(slide, text):
    shape = slide.shapes.title
    fill(shape, [text], 32, bullet=False)
    shape.width = Inches(9.35 if shape.left > Inches(0.8) else 6.95)
    for run in shape.text_frame.paragraphs[0].runs:
        run.font.bold = True


def footer(slide, owner, number):
    box(slide, 0.55, 7.1, 8.55, 0.27,
        [f"{owner}  |  COS711 Assignment 2  |  {number}"], size=10)


def picture(slide, path, x, y, w, h):
    from PIL import Image
    with Image.open(path) as image:
        ratio = image.width / image.height
    width, height = min(w, h * ratio), min(h, w / ratio)
    slide.shapes.add_picture(str(path), Inches(x + (w - width) / 2),
                             Inches(y + (h - height) / 2), Inches(width), Inches(height))


def table(slide, x, y, widths, contents, size=18, row_height=0.48):
    shape = slide.shapes.add_table(len(contents), len(widths), Inches(x), Inches(y),
                                  Inches(sum(widths)), Inches(row_height * len(contents)))
    for col, width in zip(shape.table.columns, widths):
        col.width = Inches(width)
    for i, row in enumerate(contents):
        for j, value in enumerate(row):
            cell = shape.table.cell(i, j)
            fill(cell, [str(value)], size, bullet=False)
            cell.margin_top = cell.margin_bottom = Inches(0.05)
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE if i == 0 else LIGHT if i % 2 else WHITE
            for run in cell.text_frame.paragraphs[0].runs:
                run.font.color.rgb = WHITE if i == 0 else BLUE
                run.font.bold = i == 0
                cell.text_frame.paragraphs[0].space_after = Pt(0)


def remove_body(slide):
    shape = body(slide)
    shape._element.getparent().remove(shape._element)


def test_figure(root, output, final):
    """Draw input / prediction pairs decoded from the exact final CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from pycocotools import mask as mu

    csv_path = root / "runs" / "person_c" / "final" / "submission.csv"
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != final["csv_sha256"]:
        raise ValueError("Final CSV has changed; reconcile its recorded receipt before building slides.")
    selected = ("20120622063154Uh", "20131027185354Bh")
    predictions = {name: [] for name in selected}
    for row in rows(csv_path):
        name = row["filament_id"].rsplit("_", 1)[0]
        if name in predictions:
            predictions[name].append(row["segmentation_rle"])
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for pair, name in zip(axes, selected):
        files = list((root / "data" / "test" / "test_images").glob(f"{name}.*"))
        if len(files) != 1:
            raise ValueError(f"Expected exactly one test image for {name}, found {len(files)}.")
        with Image.open(files[0]) as image:
            grey = np.asarray(image.convert("L"), dtype=np.float32) / 255
        rgb = np.repeat(grey[..., None], 3, axis=2)
        for counts in predictions[name]:
            mask = mu.decode({"size": list(grey.shape), "counts": counts.encode("ascii")}).astype(bool)
            rgb[mask] = 0.25 * rgb[mask] + 0.75 * rng.uniform(0.3, 1, 3)
        pair[0].imshow(grey, cmap="gray", vmin=0, vmax=1)
        pair[1].imshow(rgb)
        pair[0].set_title(f"Input: {name}", fontsize=12)
        pair[1].set_title(f"Final model: {len(predictions[name])} filaments", fontsize=12)
        for ax in pair:
            ax.axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return [{"observation": name, "filaments": len(predictions[name])} for name in selected]


def build(template, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new --output filename.")
    prs = Presentation(template)
    if len(prs.slides) != 9:
        raise ValueError("Expected the supplied nine-slide COS711 university template.")
    original = list(prs.slides)
    # Clone before replacing the template prompts, retaining masters and branding.
    extra_data = clone_slide(prs, original[2])
    extra_methods = clone_slide(prs, original[3])
    extra_examples = clone_slide(prs, original[5])
    extra_enhance = clone_slide(prs, original[6])
    extra_refs = clone_slide(prs, original[8])
    order = [original[0], original[1], original[2], extra_data, original[3], extra_methods,
             original[4], original[5], extra_examples, original[6], extra_enhance,
             original[7], original[8], extra_refs]
    ids = {slide.part: entry for slide, entry in zip(prs.slides, list(prs.slides._sldIdLst))}
    for entry in list(prs.slides._sldIdLst):
        prs.slides._sldIdLst.remove(entry)
    for slide in order:
        prs.slides._sldIdLst.append(ids[slide.part])

    eda = load(ROOT / "results" / "eda_summary.json")
    c = load(ROOT / "results" / "person_c" / "summary.json")
    final = load(ROOT / "results" / "person_c" / "final_submission.json")
    confirmation = {r["variant"]: float(r["PQ_dataset"]) for r in rows(
        ROOT / "results" / "person_b_study" / "confirmation.csv")}
    repeat = rows(ROOT / "results" / "person_b_study" / "fresh_seed_summary.csv")[0]
    compression = rows(ROOT / "results" / "person_c" / "compression.csv")
    base, selected = c["baseline"], c["selected"]
    fig = ROOT / "figures" / "person_c"
    scratch = ROOT / "delivery" / ".build" / "template_deck"
    scratch.mkdir(parents=True, exist_ok=True)
    test_cases = test_figure(ROOT, scratch / "final_test_pairs.png", final)
    script = []

    def notes(slide, owner, seconds, text, sources):
        slide.notes_slide.notes_text_frame.text = (
            f"{owner} | Target: {seconds} seconds\n\n{text}\n\nEvidence:\n" + "\n".join(sources))
        script.append({"slide": len(script) + 1, "owner": owner, "seconds": seconds,
                       "title": slide.shapes.title.text if slide.shapes.title else "Title",
                       "narration": text, "sources": sources})

    s = order[0]
    heading = next(shape for shape in s.shapes if shape.has_text_frame and "Assignment 3" in shape.text)
    for para in heading.text_frame.paragraphs:
        for run in para.runs:
            run.text = run.text.replace("Assignment 3", "Assignment 2").replace(
                "[Team member names and student numbers]", "Person A  |  Person B  |  Person C\n[Add names and student numbers]")
    heading.left, heading.top, heading.width, heading.height = (
        Inches(0.25), Inches(0.55), Inches(10.15), Inches(3.0))
    notes(s, "All members", 15,
          "We built a deep-learning system that identifies the pixels belonging to each individual solar filament. "
          "Filaments are clouds of solar material that appear as dark structures in these telescope images. "
          "Replace the names and student numbers before recording.",
          ["COS711-A2.pdf"])

    s = order[1]
    title(s, "Team and Responsibilities")
    body(s).width = Inches(8.2)
    fill(body(s), [
        "Person A [name / student number]: prepared and analysed the data; built image, annotation and validation tools.",
        "Person B [name / student number]: trained the segmentation network; compared methods and repeated the selected recipe.",
        "Person C [name / student number]: improved individual-filament predictions; compared compression and checked the final CSV.",
        "Goal: give every filament its own pixel mask, not just detect that filaments are present.",
    ], 23)
    notes(s, "All members", 25,
          "Each member introduces themselves and their responsibility. A mask is a pixel-level marking of an object. "
          "Instance segmentation means producing a separate mask for each individual object. "
          "Each presenter must be visible, head and shoulders, in the upper-right corner.",
          ["HANDOFF.md", "COS711-A2.pdf"])

    s = order[2]
    title(s, "Dataset Preparation")
    body(s).height = Inches(4.7)
    fill(body(s), [
        f"{eda['n_observations']} labelled observations; {eda['n_image_entries']:,} annotation sets; "
        f"{eda['n_annotations']:,} marked filaments.",
        f"{eda['n_test_images']} unlabelled test images; six GONG instruments; 2011-2022.",
        f"Filaments occupy only {100 * eda['fg_frac_mean']:.2f}% of pixels: about 278 background pixels per filament pixel.",
        "Challenges: faint thin structures, image noise and disagreement between annotators.",
    ], 21)
    s.placeholders[2].text = ""
    picture(s, ROOT / "figures" / "eda_imbalance.png", 6.8, 2.05, 5.45, 3.05)
    notes(s, "Person A", 40,
          "We analysed the supplied Kaggle images and labels. An annotation is a human marking. Multiple annotation "
          "sets can describe the same observation. Filaments cover less than one percent of pixels; this is class "
          "imbalance, meaning one category is much more common than the other. A model predicting only background "
          "could get most pixels right while finding no filaments. We therefore evaluate individual objects and shapes. "
          "We did not obtain external labels for the test images.",
          ["results/eda_summary.json", "figures/eda_imbalance.png"])

    s = order[3]
    title(s, "Dataset Preparation: Our Choices")
    body(s).width, body(s).height = Inches(8.2), Inches(3.4)
    fill(body(s), [
        "584 training / 122 validation / 1 excluded observation; all labels for one image stay together.",
        "14-day time blocks and a 1-day gap reduce near-duplicate leakage; instrument mix follows the test set.",
        "Train on random 512 x 512 patches; 70% of crop attempts target filament pixels when present.",
        "Screened contrast, disc masking, local normalisation and added transformations. Selected recipe keeps random crops, without added transformations.",
    ], 20)
    s.placeholders[2].text = ""
    picture(s, ROOT / "figures" / "eda_split_timeline.png", 0.95, 5.65, 8.0, 1.05)
    notes(s, "Person A", 45,
          "The training set teaches the model; validation is held back for comparing choices. Leakage means allowing "
          "information from evaluation examples to influence learning in an unfair way. We group related observations "
          "and keep all annotation sets together. Instruments still occur on both sides, so this is not an unseen-instrument "
          "test. Normalisation rescales pixel brightness using training-only statistics. Preprocessing changes inputs "
          "before prediction; augmentation creates varied training examples. The selected no-augmentation recipe "
          "disables added flips, rotations, brightness, noise and blur, but still samples random patches.",
          ["splits/split_v1.json", "results/person_b_study/methodology.md", "results/eda_summary.json"])

    s = order[4]
    title(s, "Experimental Setup and Methodology")
    body(s).width = Inches(8.2)
    fill(body(s), [
        "U-Net with a ResNet34 encoder, initialised from ImageNet: grayscale image in, filament score per pixel out.",
        "Training loss: binary cross entropy + Dice; rewards correct pixel labels and foreground overlap.",
        "AdamW; learning rate 0.0003; cosine schedule; batch 4; mixed precision; 20-epoch full runs.",
        "Python / PyTorch / segmentation_models_pytorch; OpenCV, Albumentations and pycocotools.",
        "Person B: RTX 5060 Ti. Person C: RTX 3070 (8 GB); ONNX Runtime and torch-pruning for compression.",
    ], 21)
    notes(s, "Person B", 45,
          "U-Net is a network that combines broad image patterns with fine spatial detail. The ResNet34 encoder is "
          "its feature-extracting component, starting from ImageNet pretrained weights. Loss is a number measuring "
          "prediction error. Binary cross entropy penalises wrong pixel labels; Dice rewards overlap. AdamW adjusts "
          "the learned weights, and the learning rate controls the update size. The cosine schedule gradually reduces "
          "that rate. An epoch here contains 960 sampled examples, not a pass over every unique image. Mixed precision "
          "uses smaller numerical formats for some calculations to reduce memory. GPUs accelerate this computation.",
          ["results/person_b_study/methodology.md", "results/person_c/person_c_results.md"])

    s = order[5]
    title(s, "Methodology: Tuning and Evaluation")
    body(s).width = Inches(8.2)
    fill(body(s), [
        "11 recipes screened for 6 epochs; strongest two retrained for 20 epochs alongside the baseline.",
        "Compared learning rates, image strategy, preprocessing, augmentation, voted labels, loss and U-Net++.",
        "After selection, three fresh random seeds measured run-to-run variation; 17 runs / 186 epochs in Person B's study.",
        "Panoptic Quality (PQ): rewards finding individual filaments and matching their shapes; not pixel accuracy.",
        "Match only if intersection-over-union (IoU) > 0.5; pool counts across all 200 validation annotation sets.",
    ], 21)
    notes(s, "Person B", 45,
          "A random seed controls random choices during training. We separated recipe selection from three fresh runs. "
          "IoU is shared mask area divided by the area covered by either mask. PQ equals the sum of matched IoUs divided "
          "by matched objects plus half the extra and missed objects. Our evaluator follows the organiser's reference "
          "and keeps overlapping annotation layers. The fixed sample budgets improve comparability, but the 1024 "
          "whole-image recipe also changes field of view and computing cost. Voted labels change observation weighting. "
          "A short screen may favour fast learners; it does not prove every rejected recipe would lose after longer training.",
          ["results/person_b_study/protocol.json", "results/person_b_study/methodology.md",
           "results/evaluation_reference.json"])

    s = order[6]
    title(s, "Results")
    remove_body(s)
    table(s, 0.95, 1.95, [5.8, 2.25], [
        ["Approach", "Validation PQ"],
        ["Original baseline (20 epochs)", f"{confirmation['baseline']:.4f}"],
        ["1024 whole images (selection seed 42)", f"{confirmation['resolution_1024']:.4f}"],
        ["No added transforms (selection seed 42)", f"{confirmation['no_augmentation']:.4f}"],
        ["Selected recipe: 3 fresh runs, mean +/- SD", f"{float(repeat['mean']):.4f} +/- {float(repeat['std_across_runs']):.4f}"],
        ["Representative seed 45", f"{base['PQ_dataset']:.4f}"],
        ["Seed 45 + output cleanup", f"{selected['PQ_dataset']:.4f}"],
        ["Final compressed model + cleanup", f"{final['validation_PQ']:.4f}"],
    ], size=17, row_height=0.46)
    box(s, 0.95, 5.82, 8.0, 0.85,
        ["The two full-budget alternatives nearly tied (+0.00063 PQ). Final compressed result is one run, not a three-run mean."],
        17)
    table(s, 9.45, 2.0, [2.85], [["Kaggle: add result"], ["Public PQ: [ADD]"], ["Rank / date: [ADD]"]], 17, 0.5)
    box(s, 9.45, 3.75, 2.85, 1.25, ["[Insert leaderboard screenshot here]"], 18)
    notes(s, "Person B / C", 55,
          "The original full-budget baseline scored 0.3201. The two alternatives nearly tied; we do not claim statistical "
          "superiority from that difference. Across fresh seeds 43, 44 and 45, the selected recipe averaged 0.3370 "
          "with sample standard deviation 0.0076. Standard deviation describes variation across runs. Seed 45 "
          "is the representative checkpoint, not the mean. Person C's cleanup raises its score to 0.4034, "
          "and the final smaller model scores 0.4031. Fill the Kaggle score, rank, date and screenshot after an actual "
          "submission; they are not measured here. Local validation and Kaggle evaluate different images.",
          ["results/person_b_study/confirmation.csv", "results/person_b_study/fresh_seed_summary.csv",
           "results/person_c/summary.json", "results/person_c/final_submission.json"])

    s = order[7]
    title(s, "Results: Example Predictions")
    remove_body(s)
    picture(s, fig / "pp_examples_slide.png", 0.92, 1.55, 8.25, 5.35)
    box(s, 9.45, 2.0, 2.85, 3.2, [
        "Labelled validation crops: teacher before / after cleanup.",
        "Largest gain: 0.34 to 0.63. Removing extra regions helps.",
        "Largest loss: 0.67 to 0.46. Small real structures can be removed.",
        "Colours = object IDs; red = removed pixels.",
    ], 16)
    notes(s, "Person C", 45,
          "These are actual labelled validation crops, selected by the largest improvement and largest decrease in "
          "observation PQ after cleanup. They are not rankings of hidden test accuracy. The score is averaged over "
          "all annotation sets for the complete observation; the pictured reference is the first set and the view "
          "is a crop. In the favourable case, removing extra regions reduces penalties. In the unfavourable case, "
          "the size rule also removes small structures marked by a human. Red highlights removed pixels; colours "
          "identify separate objects and have no physical meaning. These illustrate the teacher's cleanup, not final-model masks.",
          ["results/person_c/summary.json: examples", "figures/person_c/pp_examples_slide.png"])

    s = order[8]
    title(s, "Results: Final Test Predictions")
    remove_body(s)
    picture(s, scratch / "final_test_pairs.png", 0.92, 1.55, 8.25, 5.35)
    box(s, 9.45, 2.0, 2.85, 3.2, [
        "Original inputs beside masks decoded from the final CSV.",
        "Two fixed scenes; not best / worst accuracy rankings.",
        "Dark structures look plausible; faint edges and nearby filaments remain difficult.",
        "Unlabelled: Kaggle score pending.",
    ], 16)
    notes(s, "Person C", 40,
          "Left: original grayscale input. Right: the final pruned FP16 model's submitted masks overlaid in colour. "
          "The two fixed images are " + ", ".join(case["observation"] for case in test_cases) + ". "
          "They were chosen to illustrate different scenes, not ranked by test accuracy. Masks are decoded directly "
          "from the hash-checked final CSV, so these are the submitted predictions rather than the teacher's predictions. "
          "Some elongated dark structures are captured, but visual plausibility cannot establish correctness. "
          "Faint edges, background texture and adjacent structures remain risks; no test labels are used.",
          ["runs/person_c/final/submission.csv", "results/person_c/final_submission.json"])

    s = order[9]
    title(s, "Segmentation Enhancement")
    remove_body(s)
    table(s, 0.95, 1.95, [5.8, 2.25], [
        ["Post-processing on the same saved scores", "Validation PQ"],
        ["Baseline: cutoff 0.5, connected regions", f"{base['PQ_dataset']:.4f}"],
        ["Remove regions smaller than 320 pixels", "0.3948"],
        ["Group nearby fragments under one ID", "0.3782"],
        ["Both rules together (selected)", f"{selected['PQ_dataset']:.4f}"],
    ], 18)
    box(s, 0.95, 4.5, 8.1, 2.4, [
        "Group fragments about <= 16 pixels apart (bridge radius 8); do not add foreground pixels. Then remove groups < 320 pixels.",
        f"Extra predictions: {base['fp']:,} to {selected['fp']}. Splitting cases: 84 to 50. Cost: 17 fewer matches; merging cases 20 to 25.",
        "PQ gain +0.0585; 95% bootstrap interval +0.0486 to +0.0679. Held-out half gains: +0.0592 and +0.0557.",
    ], 18)
    box(s, 9.45, 2.0, 2.85, 3.35, [
        "Offline scorer first reproduced B's result exactly.",
        "55 + 50 settings tested.",
        "More complex rule: only +0.0005 above the selected pair.",
        "Flip averaging was slower and did not justify adoption.",
    ], 18)
    notes(s, "Person C", 55,
          "Post-processing changes the output after the network predicts. We investigated extra blobs and fragmented "
          "filaments using saved scores, first exactly reproducing Person B's results. Of unmatched regions, 48 percent "
          "were below 320 pixels, compared with about 4 percent of matched regions. We group nearby fragments under "
          "one ID without filling their gap, then discard small groups. This roughly halves extra detections. Counts "
          "are pooled over annotation sets, and fragmentation/merging are our recorded overlap diagnostics. "
          "Bootstrap intervals come from repeatedly resampling observations; they support the gain, but do not "
          "remove every risk from validation tuning. Held-out-half checks also improved. Watershed splitting, hole "
          "filling and changing the cutoff alone offered no meaningful benefit. Flip averaging means predicting "
          "flipped copies too; it cost about 3.8 times model inference and was not adopted.",
          ["results/person_c/postprocess_stage1.csv", "results/person_c/postprocess_stage2.csv",
           "results/person_c/summary.json", "results/person_c/person_c_results.md"])

    s = order[10]
    title(s, "Segmentation Enhancement: Compression")
    remove_body(s)
    labels = [
        ("Teacher mixed precision (B default)", "Teacher, mixed precision"),
        ("Teacher FP16 weights", "Teacher, FP16 storage"),
        ("KD student ResNet18", "Distilled ResNet18"),
        ("Label-only ResNet18 (control)", "Label-only control"),
        ("Pruned 50% channels + fine-tune", "50% channel pruning"),
        ("Pruned 30% + FP16 weights", "30% pruning + FP16 (FINAL)"),
    ]
    by_label = {row["label"]: row for row in compression}
    contents = [["GPU variant", "PQ", "MiB", "ms/image"]]
    for key, label in labels:
        row = by_label[key]
        contents.append([label, f"{float(row['selected_PQ']):.4f}",
                         f"{float(row['file_mb']):.1f}", f"{float(row['ms_per_image']):.0f}"])
    table(s, 0.95, 1.9, [4.5, 1.05, 1.05, 1.45], contents, 17, 0.43)
    box(s, 0.95, 5.02, 8.1, 1.95, [
        "Final: 11.93 M parameters; 22.9 MiB (~75% smaller). Training: 12.4 min teacher (B GPU) + 4.0 min pruning fine-tune (C GPU).",
        "475 ms/image over 180 images: read-to-CSV, model loading excluded. Smaller did not mean faster.",
        "CPU INT8: PQ 0.4068; 23.6 MiB; 1,158 ms/image (20 images). Separate CPU deployment.",
    ], 16)
    box(s, 9.45, 2.0, 2.85, 3.3, [
        "Pruning removes internal feature channels.",
        "FP16 stores learned numbers with fewer bits.",
        "Distillation teaches a smaller model using the teacher's outputs.",
        "Choose smallest GPU file within 0.002 PQ of teacher; speed breaks ties.",
    ], 16)
    notes(s, "Person C", 60,
          "Parameters are the model's learned numbers. Pruning removes selected internal feature channels and "
          "fine-tuning trains the remaining network again. FP16 stores weights using 16-bit numbers. "
          "Distillation trains a smaller student from both labels and teacher predictions; our matched label-only "
          "control isolates that teaching effect for this run. Fifty percent channel pruning lost quality, so we "
          "transparently added a milder thirty percent follow-up and its FP16 composition. The unchanged selection "
          "rule chooses the smallest eligible GPU model within 0.002 PQ of the teacher. The final file is 22.9 MiB "
          "(one MiB is 1,048,576 bytes), compared with 93.4 MiB, and takes 475 milliseconds per image end-to-end "
          "on the RTX 3070. This includes reading, preparation, model execution, cleanup, mask encoding checks and CSV. "
          "Training times use different GPUs and are not a hardware-controlled comparison. INT8 stores quantised "
          "values in eight bits; its CPU deployment is a separate timing comparison on twenty test images, with "
          "calibration using training images only. The final CSV contains 1,421 filaments for 180 images, including "
          "three without detections; every encoded mask was independently checked.",
          ["results/person_c/compression.csv", "results/person_c/final_selection_rule.json",
           "results/person_c/final_submission.json", "results/person_c/final_submission_check.json"])

    s = order[11]
    title(s, "Conclusions")
    fill(body(s), [
        "Main gain: better individual-object cleanup, not a more complicated network. Final local PQ: 0.4031.",
        "Compression reduced the file by about 75% while retaining nearly the same PQ; not the fastest GPU option.",
        "Roadblocks: faint thin structures, noisy / disagreeing labels, and cleanup removing small real filaments.",
        "Next: repeat final-model runs, test boundary-aware learning and consistent mild contrast enhancement; obtain Kaggle evidence.",
    ], 22)
    notes(s, "Person C / all", 40,
          "The main demonstrated improvement came from understanding object-level errors. File-size reduction did "
          "not automatically make the entire pipeline faster, because processing outside the network also matters. "
          "The final compressed result is one run and the hidden-test score still needs to be added. "
          "If repeating the project, we would allocate time for repeated final-model runs and for methods that "
          "learn thin boundaries and continuity. We would test mild contrast enhancement during both training and "
          "prediction rather than assuming prettier images improve scores. Future ideas are proposals, not measured "
          "results. Before submission add names, Kaggle evidence, and record all presenters within ten minutes and fifty MB.",
          ["results/person_c/person_c_results.md", "COS711-A2.pdf"])

    references = load(ROOT / "delivery" / "references.json")
    groups = [
        ("Bibliography", ["magfilo", "pq", "official", "unet", "resnet", "adamw"]),
        ("Bibliography: Methods and Tools", ["dice", "tversky", "unetplusplus", "smp", "cosine"]),
    ]
    for s, (heading, keys) in zip(order[12:], groups):
        title(s, heading)
        fill(body(s), [references[key]["short"] for key in keys], 17)
        if s is order[13]:
            box(s, 0.55, 5.65, 6.65, 1.1, [
                "Compression: Hinton et al. (2015), Distilling the Knowledge in a Neural Network; "
                "Fang et al. (2023), DepGraph; ONNX Runtime quantisation documentation.",
            ], 16)
        links = [references[key]["url"] for key in keys]
        links.extend([
            "https://www.kaggle.com/competitions/filament-segmentation-2026",
            "https://arxiv.org/abs/1503.02531",
            "https://arxiv.org/abs/2301.12900",
            "https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html",
        ])
        notes(s, "Reference only", 0, "References are provided for completeness; no narration required.", links)
        for para, key in zip(body(s).text_frame.paragraphs, keys):
            para.runs[0].hyperlink.address = references[key]["url"]

    for index, (slide, record) in enumerate(zip(order, script), 1):
        footer(slide, record["owner"], index)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output)
    receipt = {
        "template": template.name,
        "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "output": output.name,
        "slides": len(order),
        "target_narration_seconds": sum(record["seconds"] for record in script),
        "kaggle_result": "PLACEHOLDER: team must add measured result and screenshot",
        "final_validation_PQ": final["validation_PQ"],
        "final_csv_sha256": final["csv_sha256"],
        "test_illustrations": test_cases,
        "script": script,
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Created {output}: {len(order)} slides; target narration {receipt['target_narration_seconds']} seconds.")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "delivery" / "templates" / "COS711_A2_Presentation_Template.pptx")
    parser.add_argument("--output", type=Path, default=ROOT / "delivery" / "COS711_A2_team_template.pptx")
    args = parser.parse_args()
    build(args.template, args.output)


if __name__ == "__main__":
    main()
