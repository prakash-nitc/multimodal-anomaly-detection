# -*- coding: utf-8 -*-
"""Turn docs/09_paper/references.bib into the deck's reference list.

The slides used to carry shortened references ("Liu et al. -- Deep unsupervised
domain adaptation. APSIPA 2022."). The supervisor asked for the full entries as
they appear in the report, so the single source of truth is now the paper's own
.bib file: the deck and the paper cannot drift apart, and a number on a slide is
the same number in the report.

Numbering follows FIRST-CITATION ORDER IN main.tex, which is what the Elsevier
numeric style produces -- so [7] on a slide is [7] in the printed paper.

Usage:  python scripts/make_refs.py          # prints the Python literal
        python scripts/make_refs.py --check  # report entries never cited
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB = os.path.join(ROOT, "docs", "09_paper", "references.bib")
TEX = os.path.join(ROOT, "docs", "09_paper", "main.tex")


def parse_bib(path):
    """key -> {field: value}. Values are unwrapped to a single spaced line."""
    text = io.open(path, encoding="utf-8").read()
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        fields = {"__kind__": kind}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,|$)", body, re.S):
            fields[fm.group(1).strip().lower()] = " ".join(fm.group(2).split())
        entries[key] = fields
    return entries


def cite_order(path):
    """Keys in order of first \\cite in the tex, comments stripped."""
    lines = []
    for line in io.open(path, encoding="utf-8"):
        out, esc = [], False
        for ch in line:
            if esc:
                out.append(ch); esc = False; continue
            if ch == "\\":
                out.append(ch); esc = True; continue
            if ch == "%":
                break
            out.append(ch)
        lines.append("".join(out))
    text = "".join(lines)
    order, seen = [], set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", text):
        for key in m.group(1).split(","):
            key = key.strip()
            if key and key not in seen:
                seen.add(key)
                order.append(key)
    return order


_ACCENTS = {
    r'{\"o}': "ö", r'{\"u}': "ü", r'{\"a}': "ä",
    r"{\'e}": "é", r"{\'a}": "á", r"{\'o}": "ó",
    r"{\`e}": "è", r"{\c c}": "ç", r"{\ss}": "ß",
}


def delatex(s):
    """Bib source -> what the printed reference actually reads as.

    Brace groups exist in the .bib to protect capitalisation ({WinCLIP}) and to
    build accented letters (Sch{\\"o}lkopf); neither should reach a slide, and
    the dash forms have to become real dashes rather than '---'.
    """
    for k, v in _ACCENTS.items():
        s = s.replace(k, v)
    s = s.replace("---", "—").replace("--", "–")
    s = s.replace("{", "").replace("}", "")
    return s


def fmt_authors(raw):
    """'Patel, V. M. and Gopalan, R.' -> 'V. M. Patel, R. Gopalan'."""
    names = []
    for part in re.split(r"\s+and\s+", raw):
        part = part.strip()
        if not part:
            continue
        if "," in part:
            last, first = part.split(",", 1)
            names.append("%s %s" % (first.strip(), last.strip()))
        else:
            names.append(part)
    return ", ".join(names)


def fmt_entry(f):
    """One reference, Elsevier numeric style -- the format the report prints."""
    out = fmt_authors(f.get("author", "")) + ", " + f.get("title", "").rstrip(".")
    if f.get("booktitle"):
        out += ", in: " + f["booktitle"] + ", " + f.get("year", "")
        if f.get("pages"):
            out += ", pp. " + f["pages"].replace("--", "\u2013")
        out += "."
    elif f.get("journal"):
        j = f["journal"]
        if "arxiv" in j.lower():
            out += ", " + j + " (" + f.get("year", "") + ")."
        else:
            out += ", " + j
            if f.get("volume"):
                out += " " + f["volume"]
            out += " (" + f.get("year", "") + ")"
            if f.get("pages"):
                out += " " + f["pages"].replace("--", "\u2013")
            out += "."
    else:
        out += ", " + f.get("year", "") + "."
    if f.get("note"):
        note = f["note"]
        out += " " + (note[0].upper() + note[1:] if note else "") + "."
    return delatex(" ".join(out.split()))


def main():
    bib = parse_bib(BIB)
    order = cite_order(TEX)
    missing = [k for k in order if k not in bib]
    uncited = [k for k in bib if k not in order]
    if "--check" in sys.argv:
        print("bib entries : %d" % len(bib))
        print("cited in tex: %d" % len(order))
        if missing:
            print("CITED BUT NOT IN BIB: %s" % missing)
        if uncited:
            print("IN BIB, NEVER CITED: %s" % uncited)
        return
    # Cited entries first, in paper order; anything only the deck needs follows.
    keys = [k for k in order if k in bib] + sorted(uncited)
    print("REFS = [")
    for k in keys:
        print("    (%r,\n     %r)," % (k, fmt_entry(bib[k])))
    print("]")


if __name__ == "__main__":
    main()
