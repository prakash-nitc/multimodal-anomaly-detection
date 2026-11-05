# -*- coding: utf-8 -*-
"""Extend the Phase 3 deck with a Phase 4 section, in the Phase 3 deck's own style.

The presented Phase 3 deck (docs/06_presentations/DA-ZVAD_Phase3_Review.pptx, including
any hand edits made to it) is read, never written. A copy gains a Phase 4 section after
its summary slide: what broke on XD-Violence, how the round was run, the three
ingredients of the frozen configuration, progress on every benchmark, the comparison
with published methods and why the strongest of them score higher. The reference
slides move to the end and a fourth reference page carries the new citations, numbered
after the Phase 3 list.

Helpers are copied from scripts/build_phase3_deck.py rather than imported, because
importing that file rebuilds and saves the Phase 3 deck.

Usage:  python scripts/build_phase3_phase4_deck.py
"""
import os

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(ROOT, "docs", "06_presentations", "DA-ZVAD_Phase3_Review.pptx")
OUT = os.path.join(ROOT, "docs", "06_presentations", "DA-ZVAD_Phase3_Phase4_Review.pptx")
ARCH = os.path.join(ROOT, "docs", "09_paper", "dazvad_architecture_v3_paper.png")

# ---------------------------------------------------------------- palette (Phase 3)
INK        = RGBColor(0x14, 0x1C, 0x1E)
INK_2      = RGBColor(0x3A, 0x4A, 0x4C)
MUTED      = RGBColor(0x61, 0x72, 0x74)
ACCENT     = RGBColor(0x0D, 0x6B, 0x67)
ACCENT_LT  = RGBColor(0xDD, 0xED, 0xEB)
CAUTION    = RGBColor(0x8A, 0x57, 0x15)
CAUTION_LT = RGBColor(0xF6, 0xEB, 0xD8)
FAIL       = RGBColor(0x8C, 0x3A, 0x31)
FAIL_LT    = RGBColor(0xF6, 0xE4, 0xE1)
SURF       = RGBColor(0xF5, 0xF7, 0xF7)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
FONT, MONO = "Segoe UI", "Consolas"

W, H = 13.333, 7.5
L, R = 0.85, 0.85
CW = W - L - R

prs = Presentation(BASE)
BLANK = prs.slide_layouts[6]
_n = {"i": 30}                 # the Phase 3 summary is numbered 30


# ---------------------------------------------------------------- helpers (Phase 3)
def _tf(shape, text, size, bold=False, color=INK, font=FONT, align=PP_ALIGN.LEFT,
        spacing=1.0, italic=False):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = font
    return tf


def box(slide, x, y, w, h, text, size, **kw):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    _tf(s, text, size, **kw)
    return s


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(0.75)
    s.shadow.inherit = False
    return s


def slide(title, eyebrow=None, number=True, notes=None):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, W, H, WHITE)
    y = 0.52
    if eyebrow:
        box(s, L, y, CW, 0.25, eyebrow.upper(), 11, bold=True, color=ACCENT)
        y += 0.30
    box(s, L, y, CW, 0.6, title, 27, bold=True, color=INK)
    rect(s, L, y + 0.72, 1.5, 0.035, ACCENT)
    if number:
        _n["i"] += 1
        box(s, W - R - 0.6, H - 0.52, 0.6, 0.25, str(_n["i"]), 10, color=MUTED, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def bullets(slide_, items, top, left=L, width=CW, size=16, gap=0.46, color=INK_2, marker=True):
    y = top
    for it in items:
        head, rest = it if isinstance(it, tuple) else (None, it)
        if marker:
            rect(slide_, left, y + 0.10, 0.075, 0.075, ACCENT)
        s = slide_.shapes.add_textbox(Inches(left + (0.26 if marker else 0)), Inches(y - 0.04),
                                      Inches(width - 0.26), Inches(0.4))
        tf = s.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.line_spacing = 1.25
        if head:
            r = p.add_run(); r.text = head + "  "
            r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = INK; r.font.name = FONT
        r = p.add_run(); r.text = rest
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
        y += gap
    return y


def table(slide_, data, x, y, w, col_w=None, size=13, header=True, hi_rows=(), row_h=0.42,
          num_cols=(), fail_rows=()):
    rows, cols = len(data), len(data[0])
    shp = slide_.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(row_h * rows))
    tbl = shp.table
    tbl.first_row = header
    tbl.horz_banding = False
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Inches(cw)
    for ri, row in enumerate(data):
        tbl.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = cell.margin_right = Inches(0.14)
            cell.margin_top = cell.margin_bottom = Inches(0.05)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0 and header:
                cell.fill.fore_color.rgb = INK
            elif ri in hi_rows:
                cell.fill.fore_color.rgb = ACCENT_LT
            elif ri in fail_rows:
                cell.fill.fore_color.rgb = FAIL_LT
            else:
                cell.fill.fore_color.rgb = WHITE if ri % 2 else SURF
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if ci in num_cols else PP_ALIGN.LEFT
            r = p.add_run(); r.text = str(val)
            r.font.name = MONO if (ci in num_cols and ri > 0) else FONT
            r.font.size = Pt(size)
            r.font.bold = (ri == 0 and header) or ri in hi_rows
            r.font.color.rgb = WHITE if (ri == 0 and header) else INK
    return shp


def callout(slide_, x, y, w, h, tag, text, tone="accent", size=15):
    bar, bg, tc = {"accent": (ACCENT, ACCENT_LT, ACCENT), "caution": (CAUTION, CAUTION_LT, CAUTION),
                   "fail": (FAIL, FAIL_LT, FAIL)}[tone]
    rect(slide_, x, y, w, h, bg)
    rect(slide_, x, y, 0.05, h, bar)
    box(slide_, x + 0.28, y + 0.16, w - 0.55, 0.24, tag.upper(), 10.5, bold=True, color=tc)
    box(slide_, x + 0.28, y + 0.48, w - 0.55, h - 0.62, text, size, color=INK)


def bignum(slide_, x, y, value, label, color=ACCENT, vsize=72, w=3.2):
    box(slide_, x, y, w, 1.1, value, vsize, bold=True, color=color, font=FONT)
    box(slide_, x, y + 1.08, w, 0.5, label, 13, color=MUTED)


def refs_slide(title, eyebrow, entries):
    s = slide(title, eyebrow)
    y = 1.92
    for n, text in entries:
        sh = s.shapes.add_textbox(Inches(L), Inches(y), Inches(CW), Inches(0.30))
        tf = sh.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.line_spacing = 1.02
        r = p.add_run(); r.text = "[%d]  " % n
        r.font.size = Pt(9); r.font.bold = True; r.font.color.rgb = INK; r.font.name = FONT
        r = p.add_run(); r.text = text
        r.font.size = Pt(9); r.font.color.rgb = INK; r.font.name = FONT
        y += 0.335
    return s


# ---------------------------------------------------------------- new references
# Numbered after the 37 of the Phase 3 deck; authors and venues read from each PDF.
NEW_REFS = [
    ("wu2020xdviolence", "P. Wu, J. Liu, Y. Shi, Y. Sun, F. Shao, Z. Wu, Z. Yang, Not only look, but also listen: Learning multimodal violence detection under weak supervision, in: Proceedings of the European Conference on Computer Vision (ECCV), 2020."),
    ("cao2023nwpu", "C. Cao, Y. Lu, P. Wang, Y. Zhang, A new comprehensive benchmark for semi-supervised video anomaly detection and anticipation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023."),
    ("yin2026nova", "W.-C. Yin, Y.-C. Kao, C.-K. Lin, Y.-C. Tseng, NOVA: Normal-side modeling for training-free zero-shot video anomaly detection, arXiv preprint arXiv:2609.06360 (2026)."),
    ("zou2026askhint", "S. Zou, X. Tian, L. Wesemann, F. Waschkowski, Z. Yang, J. Zhang, Unlocking vision-language models for video anomaly detection via fine-grained prompting, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026."),
    ("bolya2025pe", "D. Bolya, P.-Y. Huang, P. Sun, J. H. Cho, A. Madotto, C. Wei, T. Ma, J. Zhi, J. Rajasegaran, H. Rasheed, J. Wang, M. Monteiro, H. Xu, S. Dong, N. Ravi, D. Li, P. Dollár, C. Feichtenhofer, Perception Encoder: The best visual embeddings are not at the output of the network, in: Advances in Neural Information Processing Systems (NeurIPS), 2025."),
    ("lee2025flashback", "H. Lee, H. Kim, I.-J. Kim, Y. Choi, Flashback: Memory-driven zero-shot, real-time video anomaly detection, arXiv preprint arXiv:2505.15205 (2025)."),
    ("song2026vlmanswer", "I. Song, J. Lee, A VLM answer is not an anomaly score: Rank compression across image and video anomaly detection, arXiv preprint arXiv:2608.21244 (2026)."),
    ("yang2024anomalyruler", "Y. Yang, K. Lee, B. Dariush, Y. Cao, S.-Y. Lo, Follow the rules: Reasoning for video anomaly detection with large language models, arXiv preprint arXiv:2407.10299 (2024)."),
    ("acsintoae2022ubnormal", "A. Acsintoae, A. Florescu, M.-I. Georgescu, T. Mare, P. Sumedrea, R. T. Ionescu, F. S. Khan, M. Shah, UBnormal: New benchmark for supervised open-set video anomaly detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022."),
]
OLD_NO = {"anyanomaly2025": 2, "zanella2024lavad": 3, "radford2021clip": 4, "ye2025vera": 9,
          "liu2023llava": 24, "sultani2018ucfcrime": 27, "liu2018shanghaitech": 21,
          "lv2021mpn": 33, "yan2023fpdm": 34, "micorek2024mulde": 36}
REF_NO = dict(OLD_NO)
REF_NO.update({k: 38 + i for i, (k, _t) in enumerate(NEW_REFS)})


def cite(*keys):
    return "[" + ", ".join(str(n) for n in sorted({REF_NO[k] for k in keys})) + "]"


# ================================================================ PHASE 4 SECTION
s = slide("What happened after Phase 3", "Phase 4  ·  November 2026", notes=(
    "This section is what I did after the last review. Read the timeline left to right. Phase 2 showed "
    "one shared sentence helps on campus video. Phase 3 had the system write a sentence per camera, and "
    "UCF-Crime reached 0.824. In Phase 4 I added two benchmarks, and on XD-Violence the sentence broke. "
    "The rest of this section is how I fixed that, without any training, and what the new model scores."))
stages = [
    ("PHASE 2", "One sentence for the scene", "ShanghaiTech 0.734", ACCENT_LT, ACCENT),
    ("PHASE 3", "A sentence per camera, written by the system", "ShanghaiTech 0.751 · UCF-Crime 0.824", ACCENT_LT, ACCENT),
    ("PHASE 4a", "Two new benchmarks — the sentence breaks", "XD-Violence 0.624 → 0.417", FAIL_LT, FAIL),
    ("PHASE 4b", "Describe normal calmly, list concrete actions", "XD-Violence 0.722 · frozen", ACCENT_LT, ACCENT),
]
for i, (tag, what, num, bg, tc) in enumerate(stages):
    x = L + i * 2.95
    rect(s, x, 2.1, 2.75, 2.6, bg)
    rect(s, x, 2.1, 0.05, 2.6, tc)
    box(s, x + 0.25, 2.3, 2.4, 0.3, tag, 11, bold=True, color=tc)
    box(s, x + 0.25, 2.7, 2.35, 1.1, what, 16, bold=True, color=INK)
    box(s, x + 0.25, 3.95, 2.35, 0.6, num, 13, color=INK_2, font=MONO)
callout(s, L, 5.2, CW, 1.25, "The point of this section",
        "Every step was decided by a rule written before the result existed, and the failures are reported. "
        "No model was trained at any point.", size=15)

s = slide("Two new benchmarks — and the sentence broke", "Phase 4  ·  What broke", notes=(
    "Two new benchmarks. XD-Violence: films, sports, games, news and CCTV, scored by average precision, its "
    "official measure. NWPU Campus: 43 campus scenes, some anomalies depend on the scene. With no sentence XD "
    "scores 0.624; with the LLaVA sentence it falls to 0.417. That is the failure this section answers."))
bignum(s, L, 2.0, "0.624", "XD-Violence AP — no sentence", color=ACCENT, w=3.4)
bignum(s, L, 3.9, "0.417", "XD-Violence AP — with the LLaVA sentence", color=FAIL, w=3.6)
table(s, [["Benchmark", "What it contains", "Measure"],
          ["XD-Violence " + cite("wu2020xdviolence"), "800 test videos: films, sports, games, news, CCTV; 6 violence classes", "AP"],
          ["NWPU Campus " + cite("cao2023nwpu"), "242 test videos, 43 campus scenes, scene-dependent anomalies", "AUROC"]],
      L + 4.2, 2.0, CW - 4.2, col_w=[2.3, 3.98, 1.15], size=13, row_h=0.62)
callout(s, L + 4.2, 4.25, CW - 4.2, 1.05, "NWPU Campus",
        "The sentence neither helped nor hurt there: 0.659 without, 0.667 with — within noise.", size=14)

s = slide("Why it broke: the sentence described the violence as normal", "Phase 4  ·  Diagnosis", notes=(
    "The cause is simple. LLaVA writes the sentence from a video's first three frames. In 127 of the 500 "
    "violent XD videos, the violence had already started in those frames, so the sentence called the fight "
    "normal. Dropping the sentence on just those videos recovers two thirds of the loss. The lesson: the fix "
    "must not depend on what the first frames happen to show."))
bullets(s, [("How the sentence is made.", "LLaVA captions the first 3 sampled frames of each video."),
            ("What went wrong.", "In 127 of 500 violent XD videos, the violence had already begun in those frames."),
            ("The effect.", "The normal-side sentence named the event, so the detector treated it as normal."),
            ("The check.", "Dropping the sentence on those 127 videos recovers 66% of the loss (0.417 → 0.554).")],
        2.05, size=17, gap=0.62)
callout(s, L, 4.85, CW, 1.4, "Lesson",
        "Whatever replaces the sentence must not depend on what a camera's first frames happen to show.",
        tone="caution", size=16)

s = slide("Fifteen new papers — and a mistake in our own design", "Phase 4  ·  What the literature taught us", notes=(
    "I read fifteen recent papers. The key lesson came from NOVA: the normal side must never contain actions "
    "that look like anomalies. Our earlier near-miss list did exactly the opposite - it added hugging as normal "
    "next to fighting. I disclosed that. ASK-Hint showed concrete actions beat abstract labels. And the "
    "strongest methods use a newer encoder, PE-Core."))
bullets(s, [("NOVA " + cite("yin2026nova") + ".", "The normal side must never contain look-alike actions."),
            ("Our mistake.", "Our ‘near-miss’ list added look-alikes (‘hugging’ beside fighting) — the opposite. Disclosed."),
            ("ASK-Hint " + cite("zou2026askhint") + ".", "Concrete actions (‘punching’) work better than labels (‘fighting’)."),
            ("Encoders.", "The strongest recent methods use PE-Core " + cite("bolya2025pe") + ", a newer frozen encoder.")],
        2.05, size=17, gap=0.62)
callout(s, L, 4.85, CW, 1.4, "What this changed",
        "Phase 4 tests these ideas one at a time — and corrects the near-miss design instead of hiding it.", size=16)

s = slide("How Phase 4 was run: one idea at a time, rules first", "Phase 4  ·  Method", notes=(
    "Before each idea I wrote down the rule and my predictions and committed them to Git, so the date proves "
    "the order. Every run first had to reproduce the old scores exactly. Then the rule decided, not me. "
    "An idea is kept only if it is better on more than half the datasets, significantly better on at least "
    "two, and significantly worse on none. Significance is a paired bootstrap over videos."))
steps = [("1  Write the rule", "and predictions"), ("2  Commit to Git", "before any score"),
         ("3  Reproduce", "old scores exactly"), ("4  Rule decides", "keep or drop")]
for i, (h1, h2) in enumerate(steps):
    x = L + i * 2.95
    rect(s, x, 2.05, 2.75, 1.25, SURF)
    box(s, x + 0.22, 2.25, 2.4, 0.35, h1, 16, bold=True, color=INK)
    box(s, x + 0.22, 2.7, 2.4, 0.35, h2, 14, color=INK_2)
callout(s, L, 3.75, CW, 1.55, "The keep rule, across the four benchmarks",
        "Higher on more than half of them  ·  significantly better on at least two (95% interval above zero)  ·  "
        "significantly worse on none.\nSignificance: 1,000 paired re-draws of the test videos.", size=16)

s = slide("Ingredient 1 — a stronger frozen encoder (PE-Core)", "Phase 4  ·  What we kept", notes=(
    "First ingredient: PE-Core replaces CLIP ViT-L. Same job, newer model. But be clear: on its own it is better "
    "only on UCF-Crime and clearly worse on XD. So the encoder alone is not the improvement; the gains came "
    "from the text, which the next two slides show."))
table(s, [["No text changes", "ShanghaiTech", "UCF-Crime", "XD (AP)", "NWPU"],
          ["CLIP ViT-L/14 (Phases 1–3)", "0.707", "0.756", "0.624", "0.659"],
          ["PE-Core " + cite("bolya2025pe"), "0.666", "0.847", "0.494", "0.618"]],
      L, 2.1, CW, col_w=[4.03, 1.9, 1.9, 1.9, 1.9], size=15, row_h=0.6, num_cols=(1, 2, 3, 4))
callout(s, L, 4.3, CW, 1.4, "Reading it",
        "Better on UCF-Crime only; significantly worse on XD-Violence. The encoder alone is not the fix — "
        "the gains came from the text.", tone="caution", size=16)

s = slide("Ingredient 2 — describe normal calmly (C)", "Phase 4  ·  What we kept", notes=(
    "Second ingredient, the main idea. Eight fixed phrases added to the normal side: four quiet places, four "
    "calm people. Calm normality keeps the two sides far apart, so action moves clearly toward abnormal. "
    "On PE-Core: plus 9 on ShanghaiTech, 14 on XD, 5 on NWPU, all significant; UCF minus 3, within noise. "
    "A second, completely different wording also passed, so it is not one lucky set of words."))
table(s, [["Quiet places", "Calm people"],
          ["an empty walkway", "a person standing still"],
          ["an empty street", "a person sitting quietly"],
          ["an empty room", "two people talking calmly"],
          ["a quiet open area with nobody around", "a person walking slowly"]],
      L, 2.05, 6.4, col_w=[3.4, 3.0], size=13, row_h=0.45)
table(s, [["Effect on PE-Core", "Change"],
          ["ShanghaiTech", "+0.091 *"], ["XD-Violence (AP)", "+0.141 *"],
          ["NWPU Campus", "+0.051 *"], ["UCF-Crime", "−0.030"]],
      L + 6.8, 2.05, CW - 6.8, col_w=[2.9, 1.93], size=13, row_h=0.45, num_cols=(1,))
callout(s, L, 4.6, CW, 1.2, "Not luck",
        "* significant (95% interval above zero). A second set of eight different calm phrases also passed the rule. "
        "Rule from NOVA " + cite("yin2026nova") + ": no look-alike actions on the normal side.", size=14)

s = slide("Ingredient 3 — a frozen bank of 48 concrete actions (B)", "Phase 4  ·  What we kept", notes=(
    "Third ingredient: the abnormal side becomes 48 concrete actions, twelve types of four. Written once by an "
    "LLM before it was scored, then frozen, and the same list for every dataset. On top of the calm phrases "
    "it adds about 9 points on XD and 4 on UCF, both significant; ShanghaiTech and NWPU move within noise. "
    "Be open that the LLM knew these datasets' anomaly types - that is why the sealed test matters."))
types = ["fighting", "weapon", "theft", "vandalism", "road accident", "fire or explosion", "riot or arrest",
         "abuse", "falling", "vehicle among pedestrians", "sudden running", "trespass"]
rows = [types[i:i + 3] for i in range(0, 12, 3)]
t3 = table(s, [["12 types  ×  4 phrases each", "", ""]] + rows, L, 2.05, 7.0, col_w=[2.2, 2.4, 2.4], size=13, row_h=0.45)
t3.table.cell(0, 0).merge(t3.table.cell(0, 2))
table(s, [["Effect on top of C", "Change"],
          ["XD-Violence (AP)", "+0.088 *"], ["UCF-Crime", "+0.039 *"],
          ["ShanghaiTech", "+0.026"], ["NWPU Campus", "−0.030"]],
      L + 7.4, 2.05, CW - 7.4, col_w=[2.6, 1.63], size=13, row_h=0.45, num_cols=(1,))
callout(s, L, 4.6, CW, 1.2, "How it was made",
        "Examples: ‘two people punching each other’, ‘a cyclist riding through a crowd of pedestrians’. "
        "Written once by an LLM before scoring, then frozen — the same list for every dataset. "
        "Concrete actions follow ASK-Hint " + cite("zou2026askhint") + ". * significant.", size=14)

# the new framework, drawn
s = slide("The new framework: no sentence, 59 fixed phrases", "Phase 4  ·  Proposed framework", notes=(
    "This is the new architecture. Compare it with the Phase 3 one: M1 is PE-Core instead of CLIP; M3 no longer "
    "holds a sentence per camera - it holds the calm normal phrases and the action bank, 59 phrases fixed in "
    "advance. Scoring and smoothing are unchanged. LLaVA is gone. A new camera needs no new data."))
with Image.open(ARCH) as im:
    pw, ph = im.size
pic_w = 8.55
pic_h = pic_w * ph / pw
s.shapes.add_picture(ARCH, Inches(L), Inches(1.85), Inches(pic_w))
x2 = L + pic_w + 0.25
rect(s, x2, 1.85, CW - pic_w - 0.25, pic_h, ACCENT_LT)
box(s, x2 + 0.2, 2.0, CW - pic_w - 0.6, 0.3, "WHAT CHANGED", 10.5, bold=True, color=ACCENT)
bullets(s, [("M1", "PE-Core replaces CLIP"), ("M3", "calm normal + action bank, not a sentence"),
            ("Same", "scoring and smoothing"), ("Gone", "LLaVA and the sentence")],
        2.45, left=x2 + 0.2, width=CW - pic_w - 0.55, size=13, gap=0.78)

# progress
s = slide("Progress on every benchmark", "Phase 4  ·  Results", notes=(
    "This is the progress slide. Each row is a stage of the project, on the same test videos. The shared sentence "
    "of Phase 2, the per-camera and per-video sentences of Phase 3, then Phase 4. The final row is the frozen "
    "model. XD went from 0.417 to 0.722; UCF from 0.824 to 0.856; ShanghaiTech from 0.751 to 0.783. NWPU is the "
    "honest exception: 0.639, about three points below the calm phrases alone, within noise."))
table(s, [["Stage", "ShanghaiTech", "UCF-Crime", "XD (AP)", "NWPU"],
          ["Phase 1–2 · CLIP, no sentence", "0.707", "0.756", "0.624", "0.659"],
          ["Phase 2 · one shared sentence", "0.734", "0.813", "0.582", "0.648"],
          ["Phase 3 · sentence per camera / video (LLaVA)", "0.751", "0.824", "0.417", "0.667"],
          ["Phase 4 · PE-Core + calm phrases", "0.757", "0.817", "0.635", "0.669"],
          ["Phase 4 · PE-Core + calm phrases + action bank (frozen)", "0.783", "0.856", "0.722", "0.639"]],
      L, 2.0, CW, col_w=[5.23, 1.6, 1.6, 1.6, 1.6], size=14, row_h=0.52, num_cols=(1, 2, 3, 4),
      hi_rows=(5,), fail_rows=())
callout(s, L, 5.35, CW, 1.15, "Against the Phase 3 model, same videos",
        "No worse on any benchmark, significantly better on XD-Violence (+0.305) — and no captioning model.", size=15)

s = slide("Progress, drawn", "Phase 4  ·  Results", notes=(
    "The same numbers as bars. Look at XD: the Phase 3 bar collapses and the Phase 4 bars recover and pass where "
    "we started. On the other three, each stage is level or higher, except the small NWPU dip at the end."))
cd = CategoryChartData()
cd.categories = ["ShanghaiTech", "UCF-Crime", "XD-Violence (AP)", "NWPU Campus"]
series = [("Phase 2 · shared sentence", (73.4, 81.3, 58.2, 64.8), RGBColor(0xC9, 0xD3, 0xD2)),
          ("Phase 3 · LLaVA sentence", (75.1, 82.4, 41.7, 66.7), RGBColor(0x8F, 0xA4, 0xA6)),
          ("Phase 4 · final (frozen)", (78.3, 85.6, 72.2, 63.9), ACCENT)]
for name, vals, _c in series:
    cd.add_series(name, vals)
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(L), Inches(1.9), Inches(CW), Inches(4.75), cd)
ch = gf.chart
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(12); ch.legend.font.name = FONT
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 30, 100
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xE8, 0xE8)
va.tick_labels.font.size = Pt(11); va.tick_labels.font.name = FONT
ch.category_axis.tick_labels.font.size = Pt(12); ch.category_axis.tick_labels.font.name = FONT
plot = ch.plots[0]
plot.gap_width = 70
plot.has_data_labels = True
plot.data_labels.font.size = Pt(10); plot.data_labels.font.name = FONT
plot.data_labels.number_format = "0.0"; plot.data_labels.number_format_is_linked = False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
for ser, (_n2, _v, col) in zip(plot.series, series):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col

s = slide("What we tried and dropped", "Phase 4  ·  Negative findings", notes=(
    "Briefly, what did not survive the rule. Labels gained on three datasets but hurt NWPU. Two encoders only "
    "helped XD at double the cost. NOVA's scoring by type raised XD to 0.769 but cost 8 points on ShanghaiTech. "
    "Flashback's templates hurt NWPU. Choosing phrases by scene recognised scenes well but added nothing. "
    "Camera memory hurt ShanghaiTech and XD. All of these had rules written first."))
table(s, [["Idea", "Why it was dropped"],
          ["‘Normal scene:’ / ‘Anomalous scene:’ labels", "Three significant gains, but significantly worse on NWPU"],
          ["Both encoders together", "Only XD improved, at double the cost"],
          ["Scoring by anomaly type (NOVA)", "XD rose to 0.769, but ShanghaiTech fell 0.082"],
          ["Different templates per side (Flashback)", "Significantly worse on NWPU"],
          ["Phrases chosen by scene", "No gain; scene recognition itself worked (11 of 12 cameras)"],
          ["Memory of the camera's own past", "Significantly worse on ShanghaiTech and XD"]],
      L, 2.0, CW, col_w=[4.9, 6.73], size=14, row_h=0.5)
callout(s, L, 5.75, CW, 0.95, "Every failure is reported", "All six had rules written before scoring; none is hidden.", size=14)

s = slide("Against published methods", "Phase 4  ·  Assessment", notes=(
    "Here is where the frozen model stands, using only figures I checked in each paper's PDF. On UCF-Crime and XD "
    "we are behind only the top few methods. On ShanghaiTech we are level with trained methods like FPDM. "
    "On NWPU we sit inside the range of trained methods. Our intervals are several points wide, so I do not "
    "claim to beat any one method - and these numbers are exploratory until the sealed test."))
table(s, [["Method", "UCF-Crime", "XD (AP)", "ShanghaiTech", "NWPU"],
          ["NOVA " + cite("yin2026nova"), "89.86", "84.82", "—", "—"],
          ["ASK-Hint " + cite("zou2026askhint"), "89.83", "—", "—", "—"],
          ["Flashback-PE " + cite("lee2025flashback"), "87.29", "75.13", "—", "—"],
          ["DA-ZVAD, final (ours)", "85.6", "72.2", "78.3", "63.9"],
          ["Frozen VLM " + cite("song2026vlmanswer"), "85.32", "68.98", "—", "—"],
          ["AnyAnomaly " + cite("anyanomaly2025"), "80.7", "—", "79.7", "—"],
          ["LAVAD " + cite("zanella2024lavad"), "80.3", "62.01", "—", "—"],
          ["VERA " + cite("ye2025vera") + " — uses training data", "—", "70.54", "—", "—"],
          ["FPDM / MULDE " + cite("yan2023fpdm", "micorek2024mulde") + " — trained", "74.7 / 78.5", "—", "78.6 / 81.3", "—"],
          ["Cao et al. " + cite("cao2023nwpu") + " — trained, scene-conditioned", "—", "—", "—", "68.2"]],
      L, 1.95, CW, col_w=[5.03, 1.65, 1.65, 1.65, 1.65], size=12, row_h=0.39, num_cols=(1, 2, 3, 4), hi_rows=(4,))
box(s, L, 6.35, CW, 0.5, "VERA as quoted by Flashback; others from each paper's own tables. Point estimates — "
    "gaps of 1–2 points are within our uncertainty.", 11, color=MUTED)

s = slide("Why the strongest methods score higher", "Phase 4  ·  Assessment", notes=(
    "Why can't we match NOVA or Flashback? Look at what they use. NOVA writes twenty normal and twenty abnormal "
    "phrases for every anomaly class, with guidance for each dataset, and only tests on crime datasets. Flashback "
    "builds a memory of a million generated captions - 76 hours, 181 dollars - and uses a video encoder. ASK-Hint "
    "runs a large video-language model and chose its prompt count from its own results table. We use one frozen "
    "encoder and 59 fixed phrases, the same for crime and campus cameras."))
table(s, [["Method", "What gives it the edge", "What it costs"],
          ["NOVA", "20 normal + 20 abnormal phrases per anomaly class, guidance for each dataset", "Tailored to crime datasets; tested on UCF and XD only"],
          ["Flashback", "Memory of 1 million generated captions; a video encoder", "GPT-4o, 76 hours, $181 to build"],
          ["ASK-Hint", "A large video-language model asked about each video", "Heavy; prompt count chosen on its own results"],
          ["Frozen VLM", "A large model asked a question about every frame", "Heavy at run time"],
          ["Ours", "One frozen encoder, 59 fixed phrases", "The same setup for crime and campus"]],
      L, 2.0, CW, col_w=[1.9, 5.13, 4.6], size=13, row_h=0.6, hi_rows=(5,))

s = slide("Why we did not reach their numbers", "Phase 4  ·  Assessment", notes=(
    "Three reasons we are below them. One frame at a time: running looks like walking in a single frame. "
    "The whole frame at once: a small bicycle in a crowd barely changes the frame's vector. And tailoring has a "
    "price: when I tried NOVA's scoring, XD rose to 0.769 but ShanghaiTech fell 8 points. Matching them would "
    "need a heavy model, training, or tailoring per dataset - each breaks our goal. The table is what could "
    "close part of the gap without breaking it."))
bullets(s, [("One frame at a time.", "Running looks like walking in a single frame."),
            ("The whole frame at once.", "A small bicycle in a crowd barely changes the frame's vector."),
            ("Tailoring has a price.", "NOVA-style scoring raised our XD to 0.769 but cost 0.082 on ShanghaiTech.")],
        2.05, size=16, gap=0.5)
table(s, [["What could close part of the gap", "What it fixes", "Status"],
          ["Patch-level scoring", "Small objects (ShanghaiTech, NWPU)", "Planned; needs new encoding"],
          ["A frozen video encoder", "Anomalies defined by motion", "Future work"],
          ["Heavy video-language model or training", "Everything", "Breaks the training-free goal"]],
      L, 3.6, CW, col_w=[4.4, 4.0, 3.23], size=13, row_h=0.48)
callout(s, L, 5.75, CW, 0.95, "Our goal", "Stay light and training-free, and work on any camera.", size=14)

s = slide("Honest status: frozen, and one sealed test to come", "Phase 4  ·  Status", notes=(
    "Be upfront. These numbers are exploratory: the model was chosen on these four datasets, and the bank was "
    "written by an LLM that knew their anomaly types. That is why on the sixth of November I froze the model, "
    "before downloading any UBnormal data. UBnormal will be scored once, and the result reported whatever it is."))
bullets(s, [("Exploratory.", "The model was chosen on these four datasets; many ideas were scored."),
            ("Not independent.", "The LLM that wrote the bank knew these datasets' anomaly types."),
            ("NWPU dipped.", "0.669 → 0.639 when the bank was added, within noise."),
            ("Still behind the leaders on XD.", "0.722 against Flashback's 0.751 and NOVA's 0.848.")],
        2.05, size=16, gap=0.56)
callout(s, L, 4.55, CW, 1.75, "Frozen on 6 November 2026",
        "Every setting fixed by hash before any UBnormal " + cite("acsintoae2022ubnormal") + " video or label was seen. "
        "29 unseen scenes, official test split, scored once — and reported whatever it shows.", size=15)

s = slide("Summary", "Phase 4", notes=(
    "Close on progress and honesty together. The sentence broke on XD for a reason we found. Describing normal "
    "calmly and listing concrete actions fixed it, without training and without a captioning model. The model "
    "is frozen and one sealed test remains."))
points = [("What broke", "A sentence written from first frames called the violence normal on XD (0.417)."),
          ("What fixed it", "Calm normal phrases and a frozen bank of 48 concrete actions, on PE-Core."),
          ("The progress", "XD 0.417 → 0.722 · UCF 0.824 → 0.856 · ShanghaiTech 0.751 → 0.783 — no training."),
          ("The method", "One idea at a time, rules written first, six failures reported."),
          ("What remains", "One sealed test on unseen UBnormal; patch scoring; a video encoder.")]
for i, (h1, txt) in enumerate(points):
    y = 2.0 + i * 0.86
    box(s, L, y, 0.5, 0.5, str(i + 1), 26, bold=True, color=ACCENT)
    box(s, L + 0.6, y, 2.6, 0.4, h1, 16, bold=True, color=INK)
    box(s, L + 3.3, y + 0.02, CW - 3.3, 0.6, txt, 15, color=INK_2)

# new reference page
refs_slide("References", "4 of 4", [(REF_NO[k], t) for k, t in NEW_REFS])


# ================================================================ assemble
# move the three Phase 3 reference slides behind the new section, before page 4
ids = prs.slides._sldIdLst
items = list(ids)
refs = items[31:34]                 # slides 32-34: References 1-3 of 3
for el in refs:
    ids.remove(el)
for el in refs:
    ids.insert(len(ids) - 1, el)    # before the new "4 of 4" page

# renumber the reference pages and their slide numbers
order = list(prs.slides)
for page_no, sl in enumerate(order[-4:-1], 1):
    for sh in sl.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text.strip()
        if t.upper() == "%d OF 3" % page_no:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.text = r.text.replace(" OF 3", " OF 4").replace(" of 3", " of 4")
for idx, sl in enumerate(order, 0):
    if idx < 31:
        continue
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().isdigit() and sh.left > Inches(11) and sh.top > Inches(6.8):
            sh.text_frame.paragraphs[0].runs[0].text = str(idx)

# The Phase 3 summary claimed parity with the best training-free methods, and three of
# its lines wrapped under the next row's band. In this copy the claim is corrected and the
# lines are shortened to one line each, keeping their content.
SUMMARY_LINES = {
    "Domain adaptation research targets": "DA research scopes out concept shift [14, 15]; anomaly detection is dominated by it.",
    "0.751 on ShanghaiTech": "0.751 ShanghaiTech, 0.824 UCF-Crime (comparable to LAVAD, AnyAnomaly), no training; mostly calibration.",
    "It needs several scenes": "Needs several scenes: flat on Avenue (one camera), strong on ShanghaiTech and UCF-Crime, not growing beyond.",
}
fixed = 0
for sh in order[30].shapes:
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                for start, new in SUMMARY_LINES.items():
                    if r.text.startswith(start):
                        r.text = new
                        fixed += 1
print("summary lines replaced:", fixed, "of", len(SUMMARY_LINES))

prs.save(OUT)
print("saved:", OUT, "slides:", len(prs.slides))
