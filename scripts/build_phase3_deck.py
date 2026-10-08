# -*- coding: utf-8 -*-
"""Build the Phase 3 progress deck for DA-ZVAD (copied from the Phase 2 builder,
which is kept unchanged so the presented deck stays reproducible).

Design intent: a panel review, not a conference talk. Slides are read while
being spoken over, so each carries one claim, states its evidence, and stops.
Numbers are large where they are the point. The narrative deliberately includes
the failed first run -- for a panel questioning whether the work is genuinely
the student's, a debugging story is stronger evidence than a clean table.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
import os

# ---------------------------------------------------------------- palette
INK        = RGBColor(0x14, 0x1C, 0x1E)
INK_2      = RGBColor(0x3A, 0x4A, 0x4C)
MUTED      = RGBColor(0x61, 0x72, 0x74)
ACCENT     = RGBColor(0x0D, 0x6B, 0x67)
ACCENT_LT  = RGBColor(0xDD, 0xED, 0xEB)
CAUTION    = RGBColor(0x8A, 0x57, 0x15)
CAUTION_LT = RGBColor(0xF6, 0xEB, 0xD8)
FAIL       = RGBColor(0x8C, 0x3A, 0x31)
FAIL_LT    = RGBColor(0xF6, 0xE4, 0xE1)
RULE       = RGBColor(0xD8, 0xE0, 0xDF)
SURF       = RGBColor(0xF5, 0xF7, 0xF7)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Segoe UI"
MONO = "Consolas"

W, H = 13.333, 7.5
L, R = 0.85, 0.85
CW = W - L - R          # content width

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]

_n = {"i": 0}


# ---------------------------------------------------------------- helpers
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
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                               Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(0.75)
    s.shadow.inherit = False
    return s


def slide(title, eyebrow=None, number=True):
    """Standard content slide: eyebrow, title, accent rule."""
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
        box(s, W - R - 0.6, H - 0.52, 0.6, 0.25, str(_n["i"]), 10,
            color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def bullets(slide_, items, top, left=L, width=CW, size=16, gap=0.46,
            color=INK_2, marker=True):
    y = top
    for it in items:
        if isinstance(it, tuple):
            head, rest = it
        else:
            head, rest = None, it
        if marker:
            rect(slide_, left, y + 0.10, 0.075, 0.075, ACCENT)
        s = slide_.shapes.add_textbox(Inches(left + (0.26 if marker else 0)),
                                      Inches(y - 0.04), Inches(width - 0.26),
                                      Inches(0.4))
        tf = s.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.line_spacing = 1.25
        if head:
            r = p.add_run(); r.text = head + "  "
            r.font.size = Pt(size); r.font.bold = True
            r.font.color.rgb = INK; r.font.name = FONT
        r = p.add_run(); r.text = rest
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
        y += gap
    return y


def table(slide_, data, x, y, w, col_w=None, size=13, header=True,
          hi_rows=(), row_h=0.42, num_cols=()):
    rows, cols = len(data), len(data[0])
    shp = slide_.shapes.add_table(rows, cols, Inches(x), Inches(y),
                                  Inches(w), Inches(row_h * rows))
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
            cell.margin_left = Inches(0.14)
            cell.margin_right = Inches(0.14)
            cell.margin_top = Inches(0.05)
            cell.margin_bottom = Inches(0.05)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0 and header:
                cell.fill.fore_color.rgb = INK
            elif ri in hi_rows:
                cell.fill.fore_color.rgb = ACCENT_LT
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
    bar, bg, tc = {
        "accent":  (ACCENT, ACCENT_LT, ACCENT),
        "caution": (CAUTION, CAUTION_LT, CAUTION),
        "fail":    (FAIL, FAIL_LT, FAIL),
    }[tone]
    rect(slide_, x, y, w, h, bg)
    rect(slide_, x, y, 0.05, h, bar)
    box(slide_, x + 0.28, y + 0.16, w - 0.55, 0.24, tag.upper(), 10.5,
        bold=True, color=tc)
    box(slide_, x + 0.28, y + 0.48, w - 0.55, h - 0.62, text, size, color=INK)


def bignum(slide_, x, y, value, label, color=ACCENT, vsize=72, w=3.2):
    box(slide_, x, y, w, 1.1, value, vsize, bold=True, color=color, font=FONT)
    box(slide_, x, y + 1.08, w, 0.5, label, 13, color=MUTED)


_here = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(_here, "..", "docs", "09_paper", "figures")


def figure_slide(title, eyebrow, png, read, tone="accent"):
    """A slide that is one figure, a line telling the audience how to read it,
    and nothing else. The figures are the paper's own, not redrawn for the
    deck, so what is projected is what the report contains."""
    path = os.path.join(FIGDIR, png)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{png} not found in {FIGDIR}. Run scripts/make_report_figures.py "
            f"and scripts/make_result_charts.py first.")
    s = slide(title, eyebrow)
    # The caption band is anchored, not pushed down by the picture: a tall
    # figure would otherwise squeeze it to nothing. The picture is then fitted
    # into whatever remains above it and centred there, so the six figure
    # slides all place their caption on the same line.
    TOP, CAP_H, CAP_Y = 1.95, 1.42, 5.72
    with Image.open(path) as im:
        pw, ph = im.size
    avail_w, avail_h = CW, CAP_Y - 0.25 - TOP
    w_in = min(avail_w, avail_h * pw / ph)
    h_in = w_in * ph / pw
    s.shapes.add_picture(path, Inches(L + (CW - w_in) / 2),
                         Inches(TOP + (avail_h - h_in) / 2), Inches(w_in))
    callout(s, L, CAP_Y, CW, CAP_H, "How to read it", read, tone=tone, size=14)
    return s


def refs_slide(title, eyebrow, entries):
    """A plain numbered reference list, set the way the report sets it.

    Full Elsevier-style entries run 200-290 characters. Two columns would wrap
    almost every one onto four short lines, so this is a single column across
    the full text width. No group headings and no accent colour: on a reference
    slide the number is the only thing anyone needs to find.
    """
    s = slide(title, eyebrow)
    y = 1.92
    for n, text in entries:
        sh = s.shapes.add_textbox(Inches(L), Inches(y), Inches(CW), Inches(0.30))
        tf = sh.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = 0
        tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.line_spacing = 1.02
        r = p.add_run(); r.text = "[%d]  " % n
        r.font.size = Pt(9); r.font.bold = True
        r.font.color.rgb = INK; r.font.name = FONT
        r = p.add_run(); r.text = text
        r.font.size = Pt(9); r.font.color.rgb = INK; r.font.name = FONT
        y += 0.335
    return s



# ================================================================ REFERENCES
# Generated from docs/09_paper/references.bib by scripts/make_refs.py, in the
# paper's own first-citation order -- so [n] on a slide is [n] in the printed
# report and the two cannot drift apart. Regenerate after editing the .bib:
#     PYTHONIOENCODING=utf-8 python scripts/make_refs.py > refs_generated.py
REFS = [
    ('wilkinghoff2026context',
     'K. Wilkinghoff, N. Madan, J. M. Valverde, K. Nasrollahi, R. T. Ionescu, R. Wisniewski, T. B. Moeslund, W. Wang, Z.-H. Tan, Out of context: Reliability in multimodal anomaly detection requires contextual inference, arXiv preprint arXiv:2604.13252 (2026).'),
    ('anyanomaly2025',
     'S. Ahn, Y. Jo, K. Lee, S. Kwon, I. Hong, S. Park, AnyAnomaly: Zero-shot customizable video anomaly detection with LVLM, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026, pp. 3026–3035.'),
    ('zanella2024lavad',
     'L. Zanella, W. Menapace, M. Mancini, Y. Wang, E. Ricci, Harnessing large language models for training-free video anomaly detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18527–18536.'),
    ('radford2021clip',
     'A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever, Learning transferable visual models from natural language supervision, in: Proceedings of the International Conference on Machine Learning (ICML), 2021, pp. 8748–8763.'),
    ('jeong2023winclip',
     'J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, O. Dabeer, WinCLIP: Zero-/few-shot anomaly classification and segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 19606–19616.'),
    ('zhou2024anomalyclip',
     'Q. Zhou, G. Pang, Y. Tian, S. He, J. Chen, AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection, in: Proceedings of the International Conference on Learning Representations (ICLR), 2024.'),
    ('bergmann2019mvtec',
     'P. Bergmann, M. Fauser, D. Sattlegger, C. Steger, MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9592–9600.'),
    ('roth2022patchcore',
     'K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler, Towards total recall in industrial anomaly detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 14318–14328.'),
    ('ye2025vera',
     'M. Ye, W. Liu, P. He, VERA: Explainable video anomaly detection via verbalized learning of vision-language models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 8679–8688.'),
    ('wu2024ovvad',
     'P. Wu, X. Zhou, G. Pang, Y. Sun, J. Liu, P. Wang, Y. Zhang, Open-vocabulary video anomaly detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18297–18307.'),
    ('patel2015visual',
     'V. M. Patel, R. Gopalan, R. Li, R. Chellappa, Visual domain adaptation: A survey of recent advances, IEEE Signal Processing Magazine 32 (2015) 53–69.'),
    ('wang2018deep',
     'M. Wang, W. Deng, Deep visual domain adaptation: A survey, Neurocomputing 312 (2018) 135–153.'),
    ('wilson2020survey',
     'G. Wilson, D. J. Cook, A survey of unsupervised deep domain adaptation, ACM Computing Surveys 51 (2020) 1–46.'),
    ('liu2022deep',
     'X. Liu, C. Yoo, F. Xing, H. Oh, G. El Fakhri, J.-W. Kang, J. Woo, Deep unsupervised domain adaptation: A review of recent advances and perspectives, APSIPA Transactions on Signal and Information Processing 11 (2022). ArXiv:2208.07422.'),
    ('singhal2023domain',
     'P. Singhal, R. Walambe, S. Ramanna, K. Kotecha, Domain adaptation: Challenges, methods, datasets, and applications, IEEE Access 11 (2023) 6973–7020.'),
    ('fan2026llm',
     'L. Fan, F. Liu, C. Chen, Domain adaptation of large language models for geotechnical applications, Solid Earth Sciences (2026). Art. no. 100285.'),
    ('adavad2024',
     'D. Guo, Y. Fu, S. Li, Ada-VAD: Domain adaptable video anomaly detection, in: Proceedings of the SIAM International Conference on Data Mining (SDM), 2024, pp. 634–642.'),
    ('lu2020fewshot',
     'Y. Lu, F. Yu, M. K. K. Reddy, Y. Wang, Few-shot scene-adaptive anomaly detection, in: Proceedings of the European Conference on Computer Vision (ECCV), 2020, pp. 125–141.'),
    ('aich2023zxvad',
     'A. Aich, K.-C. Peng, A. K. Roy-Chowdhury, Cross-domain video anomaly detection without target domain adaptation, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023, pp. 2578–2590.'),
    ('kouw2018introduction',
     'W. M. Kouw, M. Loog, An introduction to domain adaptation and transfer learning, arXiv preprint arXiv:1812.11806 (2018).'),
    ('liu2018shanghaitech',
     'W. Liu, W. Luo, D. Lian, S. Gao, Future frame prediction for anomaly detection — A new baseline, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 6536–6545.'),
    ('dosovitskiy2021vit',
     'A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An image is worth 16x16 words: Transformers for image recognition at scale, in: Proceedings of the International Conference on Learning Representations (ICLR), 2021.'),
    ('vaswani2017attention',
     'A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need, in: Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5998–6008.'),
    ('liu2023llava',
     'H. Liu, C. Li, Q. Wu, Y. J. Lee, Visual instruction tuning, in: Advances in Neural Information Processing Systems (NeurIPS), 2023, pp. 34892–34916.'),
    ('dettmers2023qlora',
     'T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, QLoRA: Efficient finetuning of quantized LLMs, in: Advances in Neural Information Processing Systems (NeurIPS), 2023, pp. 10088–10115.'),
    ('lu2013avenue',
     'C. Lu, J. Shi, J. Jia, Abnormal event detection at 150 FPS in MATLAB, in: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2013, pp. 2720–2727.'),
    ('sultani2018ucfcrime',
     'W. Sultani, C. Chen, M. Shah, Real-world anomaly detection in surveillance videos, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 6479–6488.'),
    ('fawcett2006roc',
     'T. Fawcett, An introduction to ROC analysis, Pattern Recognition Letters 27 (2006) 861–874.'),
    ('bradley1997auc',
     'A. P. Bradley, The use of the area under the ROC curve in the evaluation of machine learning algorithms, Pattern Recognition 30 (1997) 1145–1159.'),
    ('davis2006pr',
     'J. Davis, M. Goadrich, The relationship between precision-recall and ROC curves, in: Proceedings of the International Conference on Machine Learning (ICML), 2006, pp. 233–240.'),
    ('scholkopf2001oneclass',
     'B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, R. C. Williamson, Estimating the support of a high-dimensional distribution, Neural Computation 13 (2001) 1443–1471.'),
    ('lv2021mpn',
     'H. Lv, C. Chen, Z. Cui, C. Xu, Y. Li, J. Yang, Learning normal dynamics in videos with meta prototype network, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 15425–15434.'),
    ('yan2023fpdm',
     'C. Yan, S. Zhang, Y. Liu, G. Pang, W. Wang, Feature prediction diffusion model for video anomaly detection, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 5527–5537.'),
    ('zhou2024mapdm',
     'H. Zhou, J. Cai, Y. Ye, Y. Feng, C. Gao, J. Yu, Z. Song, W. Yang, Video anomaly detection with motion and appearance guided patch diffusion model, arXiv preprint arXiv:2412.09026 (2024).'),
    ('micorek2024mulde',
     'J. Micorek, H. Possegger, D. Narnhofer, H. Bischof, M. Kampel, MULDE: Multiscale log-density estimation via denoising score matching for video anomaly detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18868–18877.'),
    ('girdhar2023imagebind',
     'R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, I. Misra, ImageBind: One embedding space to bind them all, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 15180–15190.'),
    ('maaz2024videochatgpt',
     'M. Maaz, H. Rasheed, S. Khan, F. Khan, Video-ChatGPT: Towards detailed video understanding via large vision and language models, in: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024, pp. 12585–12602.'),
]

REF_NO = {k: i for i, (k, _t) in enumerate(REFS, 1)}


def cite(*keys):
    """Inline marker, set the way the report sets one.

    Numbers are sorted and runs of three or more collapse to a range, so
    citing all six surveys reads [11-15, 20] rather than a bare list in
    whatever order the argument happened to name them.
    """
    ns = sorted({REF_NO[k] for k in keys})
    parts, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append("%d–%d" % (ns[i], ns[j]))
        else:
            parts.extend(str(n) for n in ns[i:j + 1])
        i = j + 1
    return "[" + ", ".join(parts) + "]"


# ================================================================ TITLE
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, WHITE)
rect(s, 0, 0, 0.32, H, ACCENT)
box(s, 1.5, 1.75, 10.5, 0.3, "PHASE 3 PROGRESS  ·  SEMESTER 3",
    12, bold=True, color=ACCENT)
box(s, 1.5, 2.25, 10.6, 1.5,
    "Domain-Adaptive Zero-Shot Anomaly Detection Using Vision-Language Models",
    38, bold=True, color=INK, spacing=1.06)
rect(s, 1.5, 3.95, 2.0, 0.04, ACCENT)
box(s, 1.5, 4.25, 10.0, 0.5,
    "Adapting a video anomaly detector to a new camera with one sentence the "
    "system writes itself — no training, no labels, every model frozen.",
    16, color=INK_2, spacing=1.3)
box(s, 1.5, 5.55, 6.0, 0.35, "Prakash Kumar Sarangi", 17, bold=True, color=INK)
box(s, 1.5, 5.92, 7.0, 0.6,
    "M.Tech Computer Science and Engineering\n"
    "Supervisors: Dr. Pranesh Das  ·  Dr. Raju Hazari", 12.5, color=MUTED,
    spacing=1.35)
box(s, 1.5, 6.75, 7.0, 0.3, "National Institute of Technology Calicut", 12.5,
    bold=True, color=ACCENT)
box(s, W - R - 2.2, 6.75, 2.2, 0.3, "October 2026", 12.5, color=MUTED,
    align=PP_ALIGN.RIGHT)

# ================================================================ 0b INTRO
# Added Oct 2026 at the supervisor's request: one slide on what anomaly
# detection is, with a picture, before the research gap. The two photographs
# are Wikimedia Commons images (CC BY-SA 2.0, credited on the slide); the score
# panel is a labelled illustration. See scripts/make_intro_figure.py.
s = slide("What is video anomaly detection?", "Introduction")
_intro = os.path.join(_here, "..", "docs", "06_presentations",
                      "fig_intro_anomaly_detection.png")
if not os.path.isfile(_intro):
    raise FileNotFoundError("run scripts/make_intro_figure.py first")
with Image.open(_intro) as _im:
    _pw, _ph = _im.size
_iw = 11.2
s.shapes.add_picture(_intro, Inches(L + (CW - _iw) / 2), Inches(1.8), Inches(_iw))
_ih = _iw * _ph / _pw
_cw, _gap = (CW - 2 * 0.25) / 3, 0.25
for _k, (_head, _body, _bg, _col) in enumerate([
        ("WHAT", "Find moments in video that do not fit what usually happens at "
                 "that place — a collision, a fight, a fall.",
         ACCENT_LT, ACCENT),
        ("HOW", "Score every frame. When the score crosses a threshold, flag "
                "that moment for a human to check.", SURF, INK),
        ("WHY IT IS HARD", "Anomalies are rare and varied, so no one can collect "
                           "examples of them all. Systems learn “normal” instead.",
         CAUTION_LT, CAUTION),
]):
    _x = L + _k * (_cw + _gap)
    _y = 1.8 + _ih + 0.12
    rect(s, _x, _y, _cw, 1.2, _bg)
    box(s, _x + 0.2, _y + 0.1, _cw - 0.4, 0.25, _head, 11, bold=True, color=_col)
    box(s, _x + 0.2, _y + 0.36, _cw - 0.4, 0.8, _body, 12.5, color=INK, spacing=1.15)
box(s, L, 1.8 + _ih + 1.36, CW - 0.8, 0.22,
    "Photos: “Scramble from above, SHIBUYA SKY” by Sei F; “Japanese car accident” "
    "by Shuets Udono — both CC BY-SA 2.0, via Wikimedia Commons, cropped. "
    "Right panel: illustration.", 8.5, italic=True, color=MUTED)

# ================================================================ 1 PROBLEM
s = slide("A detector is tied to the place it learned", "The problem")
bullets(s, [
    ("Standard approach.", "Show the system weeks of ordinary footage from one "
     "camera until it learns what normal looks like there."),
    ("The cost.", "Move it anywhere else and it fails — a new site means new "
     "footage collection and a full retraining cycle."),
], 1.95, width=CW, size=16.5, gap=0.76)

rect(s, L, 3.55, 5.55, 2.4, SURF)
box(s, L + 0.35, 3.80, 4.9, 0.3, "SHOPPING MALL", 12, bold=True, color=ACCENT)
box(s, L + 0.35, 4.20, 4.9, 1.5,
    "A forklift moving through the aisles is\nan immediate alarm.",
    17, color=INK, spacing=1.35)

rect(s, L + 5.95, 3.55, 5.55, 2.4, SURF)
box(s, L + 6.3, 3.80, 4.9, 0.3, "FACTORY FLOOR", 12, bold=True, color=CAUTION)
box(s, L + 6.3, 4.20, 4.9, 1.5,
    "The identical forklift is completely\nroutine.",
    17, color=INK, spacing=1.35)

box(s, L, 6.30, CW, 0.5,
    "Same footage. Opposite answers. The picture did not change — the rule did.",
    17, bold=True, color=INK, align=PP_ALIGN.CENTER)

# ================================================================ 2 SHIFT
s = slide("The literature solves the adjacent problem", "Research gap  ·  1 of 2")
box(s, L, 1.95, CW, 0.4,
    "Domain adaptation distinguishes between kinds of difference between "
    "places. " + cite("liu2022deep", "wang2018deep", "wilson2020survey", "singhal2023domain",
                      "patel2015visual", "kouw2018introduction"),
    16, color=INK_2)

table(s, [
    ["Type of shift", "What moves", "Example"],
    ["Covariate shift", "Appearance", "A stop sign in fog versus sunshine. Still a stop sign."],
    ["Concept shift", "The rule itself", "A bicycle on a road versus on a footpath. Identical image, opposite label."],
], L, 2.55, CW, col_w=[2.9, 2.2, 6.53], size=14, hi_rows=(2,), row_h=0.72)

callout(s, L, 4.85, CW, 1.35, "What the surveys say",
        "“Concept shift is usually not a common problem in popular object "
        "classification… this review mainly focuses on covariate shift.”   "
        "— Liu et al., 2022 " + cite("liu2022deep") + ".   Singhal et al. "
        + cite("singhal2023domain") + " list stable p(y|x) as the first condition "
        "under which domain adaptation is justified.",
        tone="caution", size=14.5)

box(s, L, 6.45, CW, 0.4,
    "Fair for object recognition — a cat is a cat everywhere. "
    "But anomaly detection is built on concept shift.",
    16, bold=True, color=INK)

# ================================================================ 3 GAP
s = slide("Why that matters for anomaly detection", "Research gap  ·  2 of 2")
box(s, L, 1.9, CW, 0.4,
    "In anomaly detection, “normal” is defined by the deployment context — "
    "not by the object. That is the definition of the task.",
    16.5, color=INK_2, spacing=1.3)

table(s, [
    ["Input", "Domain A", "Domain B"],
    ["A person running", "Park — normal", "Bank vault — anomalous"],
    ["A person lying down", "Beach — normal", "Factory aisle — anomalous"],
    ["A vehicle", "Road — normal", "Walkway — anomalous"],
], L, 2.7, CW, col_w=[3.6, 4.0, 4.03], size=14.5, row_h=0.5)

callout(s, L, 5.0, CW, 1.35, "The gap we target",
        "Methods that align appearance cannot help here — the appearance is "
        "already identical. Two domains can share p(x) exactly while the "
        "labelling function differs. A 2026 position paper " + cite("wilkinghoff2026context")
        + " argues the same premise independently; what is missing is a test "
        "of whether supplied context does any work.", size=14.5)

# ================================================================ 4 APPROACH
s = slide("DA-ZVAD: one sentence per camera, written by the system", "Proposed framework")
_here = os.path.dirname(os.path.abspath(__file__))
img = next((c for c in (
    os.path.join(_here, "dazvad_architecture.png"),
    os.path.join(_here, "..", "docs", "09_paper", "dazvad_architecture.png"),
    os.path.join(_here, "..", "docs", "06_presentations", "dazvad_architecture.png"),
) if os.path.isfile(c)), "")
if img:
    # The supervisor's review was that the figure did not read at projection
    # size. The figure itself was redrawn darker and larger (make_architecture.py);
    # the other half of that fix is here -- giving it 8.85in instead of 7.5in by
    # narrowing the module list beside it, which the figure now restates in
    # more detail anyway.
    s.shapes.add_picture(img, Inches(L), Inches(1.80), width=Inches(8.85))
box(s, L + 9.15, 1.80, 2.90, 0.3, "EVERY MODEL FROZEN", 11.5, bold=True,
    color=ACCENT)
bullets(s, [
    ("M1", "Frozen CLIP " + cite("radford2021clip") + " scores each frame: "
           "normal vs abnormal text."),
    ("M2", "Moving average over time — no parameters."),
    ("M3", "One scene sentence per camera, in the normal prompts only."),
    ("M4", "Frozen LLaVA " + cite("liu2023llava") + " writes each camera's "
           "sentence, and explains events."),
], 2.30, left=L + 9.15, width=2.90, size=11.5, gap=0.82)

callout(s, L + 9.15, 5.75, 2.90, 1.15, "New camera",
        "Caption its first frames. No labels, no gradients.", size=13)

# ================================================================ 5 FROZEN
s = slide("Why freezing everything is the point", "Method  ·  design rationale")
bullets(s, [
    ("The risk.", "If the system learned even a little from the new site, and "
     "performance improved, we could not say what caused it — the sentence, or "
     "the learning."),
    ("Our design.", "No parameter anywhere is allowed to change. Exactly one "
     "thing in the system can vary: the text."),
    ("The consequence.", "Any measured difference is attributable to the "
     "sentence. There is no other candidate."),
], 2.0, size=16.5, gap=0.95)

callout(s, L, 4.75, CW, 1.5, "Identifiability",
        "This is what makes the adaptation claim testable rather than asserted. "
        "You cannot run our central experiment on the competing systems — "
        "their text is learned on training data " + cite("ye2025vera") + ", or "
        "entangled in an LLM prior " + cite("zanella2024lavad") + ", or paired with "
        "trained detection heads " + cite("wu2024ovvad") + ".", size=16)

box(s, L, 6.5, CW, 0.4,
    "It also means the claim can be falsified — which is the next slide.",
    15, italic=True, color=MUTED)

# ================================================================ 6 PROTOCOL
s = slide("A protocol that can prove us wrong", "Evaluation design")
box(s, L, 1.95, CW, 0.4,
    "Run the identical pipeline four times. Change only the sentence.",
    16.5, color=INK_2)

table(s, [
    ["Condition", "Sentence supplied", "Purpose"],
    ["none", "M3 disabled", "Lower reference"],
    ["generic", "“a generic scene”", "Controls for merely having context"],
    ["matched", "Correct description of the scene", "The proposed operating condition"],
    ["mismatched", "Description of the wrong domain", "FALSIFYING CONTROL"],
], L, 2.6, CW, col_w=[2.4, 4.6, 4.63], size=14, hi_rows=(4,), row_h=0.52)

callout(s, L, 5.15, CW, 1.35, "Predicted signature, fixed before measurement",
        "matched ≥ generic ≥ none, with mismatched measurably WORSE. "
        "If a deliberately wrong description costs nothing, the method ignores "
        "its context and our claim is refuted.", size=15.5)

# ================================================================ 7 SETUP
s = slide("What we ran", "Experiments  ·  setup")
table(s, [
    ["Video", "ShanghaiTech " + cite("liu2018shanghaitech") + " (12 views) · CUHK Avenue "
              + cite("lu2013avenue") + " (1 view) · UCF-Crime " + cite("sultani2018ucfcrime")
              + " (~290 scenes)"],
    ["Image", "MVTec AD " + cite("bergmann2019mvtec") + " — detection baseline only"],
    ["Scale", "418 test videos · 97,752 sampled frames · frame-level ground truth"],
    ["Models", "CLIP ViT-L/14 " + cite("radford2021clip", "dosovitskiy2021vit") + " (LAION-2B) and "
               "LLaVA-1.5-7B " + cite("liu2023llava") + " 4-bit — all frozen"],
    ["Hardware", "NVIDIA A40, college GPU server · PyTorch 2.3.1 / CUDA 12.1"],
    ["Runs", "5 full runs, cached-embedding analyses, LLaVA captioning, UCF-Crime pre-registered run"],
], L, 1.95, CW, col_w=[1.6, 10.03], size=14, header=False, row_h=0.5)

callout(s, L, 5.25, CW, 1.6, "Every run records its own conditions",
        "Each run writes a manifest: the exact code commit, whether the working "
        "tree was clean, host, GPU, driver and library versions, full "
        "configuration, and per-dataset frame and label counts. Committed "
        "alongside the results — every figure in this deck is traceable to the "
        "state that produced it.", size=14)

# ================================================================ 7b DATA
figure_slide(
    "What the two campus benchmarks look like", "Experiments  ·  the data",
    "fig_dataset_samples.png",
    "Each column is one fixed camera: the scene stays constant and only the "
    "event changes — an empty walkway and the same walkway with a cyclist look "
    "alike and are labelled opposite. UCF-Crime, added in Phase 3, is the "
    "opposite case: almost every one of its 290 videos is a different place.")

# ================================================================ 8 FAILURE
s = slide("The first result was a failure", "Experiments  ·  what happened")
rect(s, L, 1.95, CW, 1.9, FAIL_LT)
rect(s, L, 1.95, 0.05, 1.9, FAIL)
bignum(s, L + 0.5, 2.15, "0.49", "frame-level AUROC — 0.50 is random guessing",
       color=FAIL, vsize=64, w=3.0)
box(s, L + 4.2, 2.35, 7.2, 1.2,
    "Fifty minutes of GPU time produced a detector performing exactly as well "
    "as a coin flip.", 17, color=INK, spacing=1.3)

box(s, L, 4.2, CW, 0.4, "And the central experiment came out backwards:",
    16.5, bold=True, color=INK)

table(s, [
    ["Condition", "Predicted", "Observed"],
    ["matched (correct description)", "Best", "WORST — 0.666"],
    ["mismatched (wrong description)", "Worst", "Among the best — 0.695"],
], L, 4.8, CW, col_w=[5.03, 3.3, 3.3], size=14.5, row_h=0.5)

box(s, L, 6.55, CW, 0.4,
    "This is the point at which the idea looks broken.",
    16, italic=True, color=FAIL)

# ================================================================ 9 FIX 1
s = slide("Diagnosis 1 — we were measuring it wrong", "Experiments  ·  diagnosis")
bullets(s, [
    ("12 camera views.", "CLIP sits at a different baseline score under each one "
     "— different lighting, different angle."),
    ("Our error.", "We pooled every frame from all 12 cameras into one ranking."),
    ("The analogy.", "Ranking students from different schools by raw marks when "
     "the schools grade differently. The comparison destroys the ordering."),
], 1.95, size=16, gap=0.88)

box(s, L, 4.5, CW, 0.35,
    "The published protocol for this benchmark " + cite("liu2018shanghaitech") + " "
    "normalises each clip first. "
    "We were not doing it.", 15.5, color=MUTED)

rect(s, L, 5.05, CW, 1.35, ACCENT_LT)
bignum(s, L + 0.6, 5.12, "0.49", "as reported", color=MUTED, vsize=42, w=2.2)
box(s, L + 3.0, 5.42, 0.9, 0.6, "→", 34, bold=True, color=ACCENT)
bignum(s, L + 4.2, 5.12, "0.67", "correctly pooled", color=ACCENT, vsize=42, w=2.4)
box(s, L + 7.3, 5.25, 4.1, 1.1,
    "Identical scores, no smoothing. Uses no labels.\n"
    "With smoothing (w=31): 0.52 → 0.71.\nBoth figures are in the paper.",
    13.5, color=INK, spacing=1.25)

# ================================================================ 9b EVIDENCE
figure_slide(
    "The evidence for that diagnosis", "Experiments  ·  diagnosis",
    "fig_camera_baselines.png",
    "Left: every frame projected to two dimensions and coloured by camera — the "
    "views sit in separate regions. Right: how similar each view is to the "
    "average frame, differing by more than 0.13 between views. Neither panel "
    "uses labels, so this is not hindsight.")

# ================================================================ 10 FIX 2
s = slide("Diagnosis 2 — the description cancelled itself out",
          "Experiments  ·  the finding")
box(s, L, 1.9, CW, 0.35,
    "We were appending the scene sentence to BOTH prompt sets:", 16, color=INK_2)

rect(s, L, 2.45, 5.55, 1.15, SURF)
box(s, L + 0.3, 2.62, 5.0, 0.25, "NORMAL PROMPT", 10.5, bold=True, color=ACCENT)
box(s, L + 0.3, 2.9, 5.0, 0.65,
    "“a university campus walkway with\n pedestrians, everything is normal”",
    12, color=INK, font=MONO, spacing=1.2)

rect(s, L + 5.95, 2.45, 5.55, 1.15, SURF)
box(s, L + 6.25, 2.62, 5.0, 0.25, "ABNORMAL PROMPT", 10.5, bold=True, color=CAUTION)
box(s, L + 6.25, 2.9, 5.0, 0.65,
    "“a university campus walkway with pedestrians,\n but something abnormal is happening”",
    12, color=INK, font=MONO, spacing=1.2)

bullets(s, [
    ("Shared words.", "Each prompt set is averaged into one summary vector. "
     "Shared text enters both — so the two summaries move toward each other."),
    ("The method depends on them being different.", "We were erasing the very "
     "contrast the decision rests on."),
    ("Which explains the inversion.", "An accurate description matches every "
     "frame strongly, so it absorbs the most contrast. A wrong one matches "
     "nothing, so it does no damage."),
], 3.85, size=15.5, gap=0.78)

callout(s, L, 5.88, CW, 1.05, "The fix",
        "Attach the description to the normal prompts only — the scene defines "
        "what normal looks like here; an anomaly is a departure from it.",
        size=15)

# ================================================================ 11 SWEEP
s = slide("After the fix: the predicted signature", "Results  ·  central experiment")
box(s, L, 1.9, CW, 0.35,
    "ShanghaiTech, all 107 clips, per-clip normalised. Every model frozen; "
    "only the sentence and its injection point vary.", 14.5, color=MUTED)

table(s, [
    ["Injection point", "none", "generic", "matched", "mismatched", "gap"],
    ["Both prompt sets", "0.707", "0.670", "0.666", "0.695", "−0.029"],
    ["Normal set only", "0.707", "0.691", "0.734", "0.628", "+0.105"],
], L, 2.5, CW, col_w=[3.23, 1.68, 1.68, 1.68, 1.68, 1.68], size=15,
    hi_rows=(2,), row_h=0.58, num_cols=(1, 2, 3, 4, 5))

box(s, L, 4.35, CW, 0.35,
    "The “none” column is identical in both rows — confirming nothing but the "
    "injection point changed.", 14, italic=True, color=MUTED)

rect(s, L, 4.95, 5.55, 1.75, ACCENT_LT)
bignum(s, L + 0.45, 5.1, "0.734", "correct description", color=ACCENT,
       vsize=44, w=3.5)

rect(s, L + 5.95, 4.95, 5.55, 1.75, FAIL_LT)
bignum(s, L + 6.4, 5.1, "0.628", "wrong description  —  a 10-point penalty",
       color=FAIL, vsize=44, w=4.9)

box(s, L, 6.9, CW, 0.35,
    "Nothing else was permitted to change, so the text caused it — page 17 says how.",
    15.5, bold=True, color=INK, align=PP_ALIGN.CENTER)

# ================================================================ 11b SWEEP CHART
figure_slide(
    "The same experiment, drawn", "Results  ·  central experiment",
    "fig_chart_sweep.png",
    "The number the experiment exists to produce is the arrow: the distance "
    "between the matched and mismatched bars. Grey bars put the sentence in "
    "both prompt sets and that distance is negative. Orange bars put it in the "
    "normal set only and it becomes +0.105.")

# ================================================================ 11c PER FRAME
figure_slide(
    "One clip, two sentences", "Results  ·  what the gap looks like",
    "fig_context_effect.png",
    "Same frozen models, same frames, same smoothing — the only difference is "
    "whether the sentence describes a campus walkway or an industrial site. "
    "The curves track each other outside the event and separate inside it. "
    "Nothing else was free to vary, so the text caused the separation.")

# ================================================================ 12 PRECISION
s = slide("Stating the claim precisely", "Results  ·  interpretation")
rect(s, L, 1.95, 5.55, 2.15, SURF)
box(s, L + 0.4, 2.2, 4.8, 0.3, "THE WEAKER HALF", 11.5, bold=True, color=MUTED)
box(s, L + 0.4, 2.6, 4.8, 1.3,
    "A correct description beats none by +0.027.\n\n"
    "Positive at every window, but inside the ±0.036 split-to-split spread.",
    15, color=INK, spacing=1.3)

rect(s, L + 5.95, 1.95, 5.55, 2.15, ACCENT_LT)
box(s, L + 6.35, 2.2, 4.8, 0.3, "THE CLAIM WE MAKE", 11.5, bold=True, color=ACCENT)
box(s, L + 6.35, 2.6, 4.8, 1.3,
    "A WRONG description lowers the benchmark score by 0.105.\n\n"
    "Nothing else could have caused it.",
    15, color=INK, spacing=1.3)

callout(s, L, 4.4, CW, 1.55, "What a second metric showed (Phase 3)",
        "Judged only by which frames rank highest INSIDE each video, the gap is "
        "+0.006 (pooled: +0.100, same w=5). The sentence mostly shifts each "
        "video's overall score level — "
        "telling cameras apart — and the pooled metric rewards that. It barely "
        "re-ranks frames within a video. A full audit is the next experiment.",
        tone="caution", size=15)

box(s, L, 6.15, CW, 0.7,
    "A secondary finding, not present in the literature: WHERE the description "
    "is injected dominates WHAT it says — to the point of reversing the "
    "direction of the effect.",
    16, bold=True, color=INK, spacing=1.3)

# ================================================================ 13 ABLATION
s = slide("Which components earn their place", "Results  ·  component ablation")
table(s, [
    ["Scoring signal", "Held-out AUROC", "Full test set"],
    ["Scene-centre normality only", "0.585 ± 0.025", "0.585"],
    ["Semantic + scene-centre", "0.645 ± 0.034", "0.640"],
    ["Kinematic — motion only", "0.685 ± 0.015", "0.686"],
    ["Semantic + kinematic", "0.711 ± 0.034", "0.706"],
    ["Semantic — language only", "0.718 ± 0.036", "0.707"],
], L, 2.0, CW, col_w=[5.2, 3.25, 3.18], size=14.5, hi_rows=(5,), row_h=0.47,
    num_cols=(1, 2))

box(s, L, 4.65, CW, 0.35,
    "Held-out figures are means over five clip-level partitions, with the "
    "spread across them.", 13.5, italic=True, color=MUTED)

bullets(s, [
    ("The language pathway alone is best.", "Adding a motion signal costs 0.001 "
     "on the full set; adding scene-centre normality costs 0.067."),
    ("This was not the expected answer.", "ShanghaiTech's anomalies look "
     "kinematic — a bicycle at cycling speed — so appearance and motion should "
     "be complementary. Measured here, they are not."),
], 5.2, size=15, gap=1.0)

# ================================================================ 13b ABLATION CHART
figure_slide(
    "Window length, and what each component adds", "Results  ·  ablation",
    "fig_chart_ablation.png",
    "Left: smoothing helps up to w=31 and then hurts, so the window has a real "
    "optimum rather than the metric rewarding more blur; the grey curve is the "
    "wrong pooling and never leaves chance. Right: the error bars overlap, so "
    "the honest claim is that nothing added beats language alone — not that "
    "language alone wins.", tone="caution")

# ================================================================ 14 AVENUE
s = slide("A second domain — the replication test", "Results  ·  cross-domain")
box(s, L, 1.9, CW, 0.35,
    "Identical frozen configuration applied to CUHK Avenue. Nothing retuned. "
    "Only the scene sentence changed.", 15, color=INK_2)

table(s, [
    ["Dataset", "none", "generic", "matched", "mismatched", "gap"],
    ["ShanghaiTech  (12 views)", "0.707", "0.691", "0.734", "0.628", "+0.105"],
    ["CUHK Avenue  (1 view)", "0.706", "0.729", "0.677", "0.657", "+0.020"],
], L, 2.5, CW, col_w=[3.23, 1.68, 1.68, 1.68, 1.68, 1.68], size=15,
    hi_rows=(2,), row_h=0.58, num_cols=(1, 2, 3, 4, 5))

rect(s, L, 4.35, 5.55, 1.5, ACCENT_LT)
box(s, L + 0.35, 4.55, 4.9, 0.3, "DETECTION TRANSFERS", 11.5, bold=True, color=ACCENT)
box(s, L + 0.35, 4.9, 4.9, 0.8,
    "0.706 vs 0.707 with no descriptor.\nThe detector works equally well.",
    15, color=INK, spacing=1.3)

rect(s, L + 5.95, 4.35, 5.55, 1.5, FAIL_LT)
box(s, L + 6.3, 4.55, 4.9, 0.3, "ADAPTATION DOES NOT", 11.5, bold=True, color=FAIL)
box(s, L + 6.3, 4.9, 4.9, 0.8,
    "Gap 5× smaller, and the correct\ndescription scores BELOW none.",
    15, color=INK, spacing=1.3)

callout(s, L, 6.0, CW, 1.15, "Our explanation — stated as a conjecture",
        "ShanghaiTech has 12 camera views; Avenue has one, so a scene description "
        "has nothing to tell apart. Tested next inside single ShanghaiTech views, "
        "and later on UCF-Crime's ~290 scenes.",
        tone="caution", size=14.5)

# ================================================================ 15 WITHIN-VIEW
s = slide("We tested that explanation, and it held", "Results  ·  the control")
box(s, L, 1.9, CW, 0.7,
    "ShanghaiTech is twelve single-view datasets stacked together. If the "
    "descriptor works by telling the model WHICH scene it is in, then confining "
    "the sweep to one camera view should reproduce Avenue's flat result.",
    15.5, color=INK_2, spacing=1.3)

table(s, [
    ["Evaluation", "Scenes", "Clips", "Gap"],
    ["CUHK Avenue (single view)", "1", "21", "+0.020"],
    ["ShanghaiTech, within a single view (mean)", "1 each", "5–34 each", "+0.033"],
    ["ShanghaiTech, pooled across views", "12", "107", "+0.105"],
    ["UCF-Crime, own sentence per video (raw, w=5)", "~290", "290", "+0.096"],
], L, 2.7, CW, col_w=[5.6, 1.9, 2.3, 1.83], size=14, hi_rows=(2,), row_h=0.47,
    num_cols=(3,))

callout(s, L, 5.25, CW, 1.05, "The prediction could have failed",
        "A within-view gap near +0.105 would have refuted it; it came back at a "
        "third. But ~290 scenes gave no bigger gap than 12: it needs scenes, "
        "it does not keep growing.", size=14)

box(s, L, 6.45, CW, 0.7,
    "What the sentence mainly supplies is WHICH scene you are in — not what "
    "counts as normal within it. Three of nine views show a negative gap, so "
    "inside one scene the effect is not reliable.",
    15.5, bold=True, color=INK, spacing=1.3)

# ================================================================ 15b WITHIN-VIEW CHART
figure_slide(
    "Every camera view, one at a time", "Results  ·  the control",
    "fig_chart_within_view.png",
    "The orange line is the pooled result; each bar is one view on its own. "
    "Most fall well short of it and three are negative — but views 03 and 07 "
    "nearly reach it, so this is a shift in the average, not a clean collapse. "
    "The bars rest on between five and thirty-four clips each.",
    tone="caution")

# ================================================================ P3-1 PER-CAMERA
figure_slide(
    "A sentence per camera — written by the system", "Phase 3  ·  Result 6",
    "fig_chart_per_camera.png",
    "Each camera gets its own sentence instead of one for all twelve. Written by "
    "hand: 0.749. Written by LLaVA from the camera's first three frames, with no "
    "human and no labels: 0.751. Give each camera ANOTHER camera's sentence and "
    "it drops — even the best of 11 swaps (0.737) loses. The sentence works by "
    "matching its camera.")

# ================================================================ P3-2 UCF-CRIME
s = slide("UCF-Crime: predictions written down first", "Phase 3  ·  Result 7")
box(s, L, 1.9, CW, 0.72,
    "290 real CCTV videos " + cite("sultani2018ucfcrime") + " from ~290 "
    "different places, 140 with a crime. Four predictions were committed to "
    "git before any score was computed.",
    15, color=INK_2, spacing=1.3)
bignum(s, L, 2.75, "0.824", "AUROC, no training — LAVAD 0.803, AnyAnomaly 0.807",
       w=3.6, vsize=60)
table(s, [
    ["Registered prediction", "Measured", "Outcome"],
    ["P1  gap grows beyond ShanghaiTech's +0.105", "+0.096", "Refuted"],
    ["P2  own sentence beats borrowed ones", "+0.110", "Holds"],
    ["P3  own sentence beats one shared sentence", "+0.011", "Holds (small)"],
    ["P4  correct beats wrong-domain description", "+0.086", "Holds"],
], L + 4.0, 2.75, CW - 4.0, col_w=[4.55, 1.45, 1.63], size=13, row_h=0.5,
    hi_rows=(1,), num_cols=(1,))
callout(s, L, 5.55, CW, 1.4, "What it shows",
        "Another video's sentence scores BELOW no sentence (0.713 vs 0.756): the "
        "wrong scene misleads. P1 failed, so the claim is narrowed from 'the "
        "benefit scales with scene diversity' to 'the benefit needs it'.",
        size=14.5)

figure_slide(
    "UCF-Crime, drawn", "Phase 3  ·  Result 7",
    "fig_chart_ucf_crime.png",
    "The video's own LLaVA sentence reaches 0.824; borrowed sentences average "
    "0.713, below having none. The dashed line is LAVAD's 0.803 (AnyAnomaly: "
    "0.807). Both are other papers' numbers, so read it as comparable, not as a "
    "margin.")

# ================================================================ 16 NEGATIVES
s = slide("Six things that did not work", "Results  ·  negative findings")
table(s, [
    ["Modification", "Outcome"],
    ["Quadrant scoring, to catch small objects", "No improvement in any configuration"],
    ["Prompts naming bicycles and vehicles", "Much worse alone — 0.486"],
    ["Clip's own average as the normality reference", "0.585, and it degrades the language signal"],
    ["Adding a motion signal", "Costs 0.001 — not complementary"],
    ["Local temporal deviation before projection", "Better scorer, but narrows the context gap"],
    ["Per-prompt max pooling", "0.678 against 0.707"],
], L, 2.0, CW, col_w=[6.6, 5.03], size=13.5, row_h=0.5)

callout(s, L, 5.3, CW, 1.55, "Why these are in the deck",
        "The claim is that the minimal configuration is the right one. That is "
        "only credible alongside the alternatives that were tried. A clean "
        "table of successes is the artefact that is easy to fabricate; six "
        "diagnosed failures are not.", size=15.5)

# ================================================================ 16 POSITION
s = slide("Where this sits against the literature", "Assessment")
# Rebuilt Sep 2026. The previous version compared our ShanghaiTech figure with
# LAVAD's, but LAVAD reports on UCF-Crime and XD-Violence and not on either
# benchmark used here -- a cross-dataset comparison presented as a same-dataset
# one. The comparison that isolates our claim is zero-shot CLIP: this pipeline
# with the sentence removed.
table(s, [
    ["Training-free method (no target data)", "Avenue", "ShT", "UCF"],
    ["Zero-shot CLIP " + cite("radford2021clip"), "62.3", "60.9", "53.2‡"],
    ["Zero-shot ImageBind " + cite("girdhar2023imagebind"), "64.5", "61.3", "53.7‡"],
    ["LLaVA-1.5 " + cite("liu2023llava"), "67.4", "59.6", "72.8‡"],
    ["Video-ChatGPT " + cite("maaz2024videochatgpt"), "76.9", "69.1", "—"],
    ["LAVAD " + cite("zanella2024lavad") + " — captioner + LLM + refiner", "—", "—", "80.3‡"],
    ["DA-ZVAD (ours) — one frozen encoder + a sentence", "67.7", "73.4", "82.4"],
    ["AnyAnomaly " + cite("anyanomaly2025") + " — 3 LVLM queries per segment",
     "87.3", "79.7", "80.7"],
], L, 1.92, CW, col_w=[6.6, 1.6, 1.6, 1.83], size=13.5, hi_rows=(6,), row_h=0.38)

box(s, L, 5.02, CW, 0.28,
    "‡ from LAVAD " + cite("zanella2024lavad") + ", Table 1. All other figures from "
    "AnyAnomaly " + cite("anyanomaly2025") + ", Tables 5–6.  Ours: Avenue/ShT one "
    "shared sentence; UCF one sentence per video.", 11, italic=True, color=MUTED)

bullets(s, [
    ("AnyAnomaly asks “is the thing I named happening?”  "
     "We ask “is anything happening that doesn’t belong here?”",
     "Theirs needs a list of what could go wrong — at evaluation it is handed "
     "the benchmark’s own anomaly classes. Ours needs a description of the ordinary."),
], 5.42, size=13.5, gap=0.9)

callout(s, L, 6.35, CW, 0.85, "Where we lead",
        "First of six training-free methods on UCF-Crime; behind AnyAnomaly on "
        "the campus benchmarks.", size=13.5)

# ================================================================ 17 LIMITS
s = slide("Limitations we are stating ourselves", "Assessment")
bullets(s, [
    ("The mechanism is bounded.", "It works on ShanghaiTech and UCF-Crime and "
     "nearly vanishes on single-camera Avenue. A pre-registered prediction that "
     "it grows with scene diversity was refuted."),
    ("Resolution ceiling.", "Whole-frame embeddings at 224×224 cannot resolve "
     "small objects. Quadrant scoring did not close it; patch-level scoring is "
     "untested."),
    ("What the sentence actually does is not settled.", "Inside each video it "
     "barely re-ranks frames (+0.006); most of the 0.105 is per-video score "
     "shifts that the pooled metric rewards. An audit is the next experiment."),
    ("The metrics are scale-sensitive.", "Softmax plus per-clip normalisation "
     "changes pooled figures without changing rankings; on UCF-Crime the effect "
     "shrinks under per-video normalisation."),
    ("Configuration selection.", "No validation split exists for these "
     "benchmarks, so we split clips and report the half never used to select."),
    ("No explanations yet.", "M4 now writes the camera sentences, but has not "
     "yet explained flagged events."),
], 1.95, size=14.5, gap=0.86)

# ================================================================ 18 NEXT
s = slide("What comes next", "Next")
table(s, [
    ["Priority", "Work", "Why it matters"],
    ["1", "Measurement audit: within-video ranking and raw scores, all benchmarks",
     "Decides what the sentence really does: re-ranks frames or calibrates cameras"],
    ["2", "XD-Violence (pre-registered) and UCF-Crime on every frame",
     "LAVAD's second benchmark; makes the comparison airtight"],
    ["3", "NWPU Campus — scene-dependent anomalies",
     "The only benchmark where the same event flips label by scene: a direct concept-shift test"],
    ["4", "Patch-level scoring; M4 explanations",
     "Higher figure on small anomalies; completes the framework for the thesis"],
], L, 2.0, CW, col_w=[1.2, 5.5, 4.93], size=13.5, row_h=0.62, num_cols=(0,))

callout(s, L, 5.2, CW, 1.5, "Done since Phase 2",
        "Within-view control with a confidence interval; prototype direction "
        "measured; per-camera sentences written by the system (0.751); "
        "UCF-Crime under pre-registered predictions (0.824); every citation "
        "checked against its source.", size=15)

# ================================================================ 19 SUMMARY
s = slide("Summary", "Phase 3")
items = [
    ("The gap", "Domain adaptation research targets covariate shift by "
     "explicit scoping " + cite("liu2022deep", "singhal2023domain") + ". Anomaly "
     "detection is dominated by concept shift."),
    ("The method", "Every model frozen; adaptation carried by one sentence per "
     "camera, which the system writes itself from a few frames."),
    ("The result", "0.751 on ShanghaiTech and 0.824 on UCF-Crime with no "
     "training. A wrong sentence costs 0.105 — mostly by shifting scores between videos."),
    ("The finding", "Where the description is injected dominates what it says, "
     "to the point of reversing the effect. Not in the literature."),
    ("The boundary", "It needs several scenes: flat on single-camera Avenue, "
     "strong on ShanghaiTech and UCF-Crime — but it does not keep growing with "
     "more scenes, as a pre-registered test showed."),
]
y = 1.95
for i, (head, body) in enumerate(items, 1):
    rect(s, L, y, CW, 0.86, SURF if i % 2 else WHITE)
    box(s, L + 0.3, y + 0.24, 0.5, 0.4, str(i), 22, bold=True, color=ACCENT)
    box(s, L + 0.95, y + 0.12, 2.5, 0.3, head, 14, bold=True, color=INK)
    box(s, L + 0.95, y + 0.42, 10.4, 0.4, body, 13.5, color=INK_2, spacing=1.2)
    y += 0.9

box(s, L, 6.75, CW, 0.35,
    "All results reproducible from committed code and run manifests.",
    13, italic=True, color=MUTED, align=PP_ALIGN.CENTER)

# ================================================================ 20 REFERENCES
# Every entry below is a real key in docs/09_paper/references.bib and is cited
# in the paper; the two lists are kept in step deliberately, so a question about
# any reference on screen can be answered from the document.
# Split across as many slides as the entries need rather than a fixed two:
# the list grew from 30 to 37 with the comparative analysis and ran off the
# bottom. 13 full Elsevier-style entries is what fits above the page number.
_numbered = [(i, t) for i, (_k, t) in enumerate(REFS, 1)]
_PER_SLIDE = 13
_pages = [_numbered[i:i + _PER_SLIDE]
          for i in range(0, len(_numbered), _PER_SLIDE)]
for _i, _page in enumerate(_pages, 1):
    refs_slide("References", "%d of %d" % (_i, len(_pages)), _page)

# ================================================================ NOTES
NOTES = [
 "Title. Introduce yourself and the one-line premise: adapting an anomaly "
 "detector to a new camera with one sentence the system writes itself, with nothing retrained.",

 "Keep it simple - this slide is for anyone on the panel who is not in this "
 "field. Left: a busy crossing, hundreds of people, all normal. Middle: a "
 "collision on a crossing - that is an anomaly. Right: what a detector "
 "produces, a score for every frame that rises at the unusual moment and "
 "crosses the alarm line; it is an illustration. Then the three boxes, one "
 "line each. These photos are not from our data - our data starts on page 9.",

 "Open with the example, not the definition. Mall versus factory - the same "
 "forklift, opposite answers. Then the cost: every new customer means new "
 "footage and a retraining cycle. That cost is what we are removing.",

 "This is the slide that shows you read the six surveys. Read the Liu quote "
 "aloud - the field scopes concept shift OUT by explicit decision. Make clear "
 "you are not criticising them: for object recognition it is the right call.",

 "Land the consequence. In anomaly detection normality IS the deployment "
 "context. Mention the 2026 position paper yourself - an independent group "
 "reached the same premise, which answers 'did you invent this problem?'",

 "Walk the four modules left to right in about thirty seconds. Point at the "
 "SETUP box: LLaVA captions a new camera's first frames and that caption is "
 "the camera's sentence - no human needed. Then stop on M3, the only place the "
 "domain enters, and its one rule: the sentence goes into the normal prompts only.",

 "The strongest methodological point in the deck. Say it slowly. And note that "
 "the competing systems CANNOT run our experiment - their text is learned on "
 "training data (VERA), or entangled in an LLM prior (LAVAD), or paired with "
 "trained detection heads (OVVAD).",

 "Emphasise that the interpretation was fixed BEFORE any measurement, and that "
 "the mismatched condition is designed to refute us. A panel will respect a "
 "test you could have failed - and later in the deck, one you partly did.",

 "Keep this brief - it is the credibility slide. If anyone doubts the work is "
 "yours, offer to open a manifest. Three video benchmarks now, UCF-Crime added "
 "in Phase 3; MVTec is only the image baseline.",

 "Do not rush past this and do not apologise for it. The first run was chance, "
 "and the key experiment came out backwards. Then pause. The next two slides "
 "are what makes this presentation worth listening to.",

 "The schools analogy works on everyone - use it. Stress twice that the fix "
 "uses no labels and is the benchmark's own published protocol, and that both "
 "figures appear in the paper.",

 "The most interesting slide. Point at the two prompts and let them see the "
 "shared words. Then the punchline: the more accurate the description, the "
 "more damage it does. That is why the result inverted.",

 "The headline. Point at the none column being identical in both rows - that "
 "is the control, and it proves only the injection point changed. Then point "
 "at 0.628 and say a wrong sentence costs ten points on the benchmark metric. "
 "Do not say 'the text redefines normal' - page 17 explains why.",

 "Be scrupulous. The +0.027 sits inside the spread, so we report direction "
 "only. Then the Phase 3 check, said plainly: judged only inside each video, "
 "the gap is +0.006. The sentence mostly shifts each video's score level - it "
 "tells cameras apart - rather than re-ranking frames. We found this ourselves "
 "and the audit is the next experiment. Saying it first is what protects you.",

 "Volunteer the surprise: we expected motion to help and it does not. The "
 "pooled embedding already registers enough of it. Reporting the prediction "
 "that failed is stronger than reporting only the ones that held.",

 "The most important slide in the deck. Detection transfers almost exactly; "
 "the adaptation does not. Give the honest reading - on Avenue a placeholder "
 "beats an accurate description, so that benefit cannot be domain adaptation. "
 "Then give the conjecture AND the experiment that would settle it - the next "
 "slide, and UCF-Crime later in the deck.",

 "This is the slide that shows a full cycle: we saw something odd, formed an "
 "explanation, designed a test that could have killed it, ran it, and it held. "
 "Say out loud that a gap near +0.105 here would have refuted us. Then give "
 "the caveat yourself - three of nine views are negative, so within one scene "
 "the effect is not reliable. The last row is Phase 3: ~290 scenes gave no "
 "bigger gap than 12, so the claim is 'needs scenes', not 'grows with scenes'.",

 "Say why you are showing failures: the claim is that the minimal "
 "configuration is right, and that is only credible next to what was tried. "
 "If the panel suspects generated work, this slide is your best answer.",

 "Anchor on the trained baseline: 73.4 matches the benchmark's own 2018 "
 "baseline using none of its training data. Then the reframe - AnyAnomaly is "
 "handed the class list, we are not. Then the UCF column: first of six "
 "training-free methods, above LAVAD and AnyAnomaly - say 'comparable or "
 "better', since these are other papers' numbers. The ‡ rows come from LAVAD's "
 "paper, the rest from AnyAnomaly's.",

 "Deliver these confidently rather than apologetically. Dwell on the first "
 "two: we found ourselves that the pooled metric rewards score shifts, and "
 "that applies to every CLIP-based zero-shot VAD paper, not only ours.",

 "The audit comes first because it decides how the result is described. Then "
 "XD-Violence, LAVAD's other benchmark. NWPU Campus is the one dataset where "
 "the same event is normal in one scene and abnormal in another - the direct "
 "test of concept shift. Point at the box at the bottom for what is done.",

 "Close on the boundary, not the number. Say plainly that one registered "
 "prediction failed and the claim was narrowed, and that the within-video "
 "check is being audited. Then hand over.",
]
# Notes for the figure and reference slides, spliced in at the deck positions
# those slides occupy. Keys are FINAL (0-based) deck positions, so insert in
# ASCENDING order: by the time we insert at key k, every lower-keyed slide is
# already in place and index k is the slot the note belongs in. Descending
# order silently shifts every note after the first splice onto the wrong slide.
EXTRA_NOTES = {
    9: "Thirty seconds, no more. Point down one column and say the scene never "
       "changes - only the event does. Then one line on UCF-Crime: there, almost "
       "every video is a different place, which is why each gets its own sentence.",

    12: "This is the proof that the diagnosis was not invented after the fact. "
        "Say clearly that neither panel uses labels. The cameras are separate "
        "regions of the space and sit at different similarity levels, which is "
        "exactly why pooling raw scores destroyed the ordering.",

    15: "The picture of the previous table. Trace the two arrows with a finger. "
        "The grey arrow points the wrong way and the orange one points the "
        "right way, and the only thing separating them is where the sentence "
        "was attached.",

    16: "The most persuasive slide in the deck for a sceptical panel, because "
        "there is no aggregation to argue with - two runs over identical "
        "frames with identical frozen weights. Say that the curves coincide "
        "outside the event on purpose: that is the control.",

    19: "Volunteer the weakness before anyone asks. The error bars overlap, so "
        "the top three are statistically tied. The claim is that nothing we "
        "added beat plain language, which is a negative result about our own "
        "elaborations, not a win over the alternatives.",

    22: "Show the spread yourself. Three views are negative and two nearly "
        "reach the pooled figure, so the result is a shift in the mean rather "
        "than a collapse everywhere. Saying this before the panel spots it is "
        "the difference between a caveat and a hole.",

    23: "Phase 3 starts here. Say what changed: each camera gets its own "
        "sentence, and LLaVA writes it from the first three frames - no human, "
        "no labels. Point at the swapped bars: even the best swap loses, so the "
        "sentence works by matching its camera. Do not claim LLaVA beats a "
        "human; 0.751 against 0.749 is noise.",

    24: "The strongest slide for credibility. The predictions were committed "
        "to git before any score existed. Say P1 failed before anyone asks, "
        "and that the claim was narrowed because of it. On LAVAD: 'comparable "
        "or better'. LAVAD samples every 16th frame too, but other papers' "
        "numbers can differ by small protocol details. AnyAnomaly gets 0.807 "
        "here - below us, unlike on the campus benchmarks.",

    25: "Point at the borrowed-sentence bar first: worse than no sentence at "
        "all. That is the clearest evidence that the sentence tells the model "
        "where it is. Then the dashed LAVAD line, with the protocol caveat.",

    31: "Do not read these out. They are here because they are the evidence "
        "base for the gap: the six surveys are what let us say the field "
        "scopes concept shift out. Be ready to say which one contains the "
        "quote - it is Liu et al. 2022.",

    32: "Also not read aloud. If asked how the work is positioned, this slide "
        "is the answer: the frozen models are all off-the-shelf, the "
        "benchmarks and metrics are the standard ones, and every method we "
        "compare against is here.",

    33: "The third reference slide exists because the comparative analysis "
        "added seven entries. If asked which of these you actually read "
        "rather than cited from a table: the six DA surveys, AnyAnomaly, "
        "LAVAD and WinCLIP. The one-class figures are as tabulated by "
        "AnyAnomaly, and the slide says so.",
}
for _idx in sorted(EXTRA_NOTES):
    NOTES.insert(_idx, EXTRA_NOTES[_idx])

# zip() truncates silently, so a mismatch would drop notes off the end of the
# deck without any error. Fail loudly instead.
assert len(NOTES) == len(prs.slides._sldIdLst), (
    f"{len(NOTES)} notes for {len(prs.slides._sldIdLst)} slides -- adding a "
    f"slide requires adding its note at the matching index.")

for sl, txt in zip(prs.slides, NOTES):
    sl.notes_slide.notes_text_frame.text = txt

# ================================================================ SAVE
# Written to the documented home for presentations rather than beside the
# build script, so there is only ever one current deck to find.
out = os.path.join(_here, "..", "docs", "06_presentations",
                   "DA-ZVAD_Phase3_Review.pptx")
prs.save(out)
print(f"saved: {out}")
print(f"slides: {len(prs.slides._sldIdLst)}   notes: {len(NOTES)}")
