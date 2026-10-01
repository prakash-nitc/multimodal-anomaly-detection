# -*- coding: utf-8 -*-
"""Cross-check the comparison table in main.tex against the provenance ledger.

Two wrong figures reached the September 2026 draft, both because a number was
taken from a web search summary rather than read out of the paper it belongs to.
A comparison against LAVAD was made on a benchmark LAVAD does not evaluate on,
and four one-class rows came from a third paper's table via a search result.

The rule that prevents a repeat is in docs/07_reference_papers/PROVENANCE.md: a
figure from another paper enters the draft only once it is in the ledger with a
source PDF and a location inside it. This script enforces the mechanical half of
that -- it cannot tell whether someone actually opened the PDF, but it can tell
whether the number in the table is in the ledger, whether the ledger and the
table agree, and whether the source PDF is present.

Usage:  python scripts/check_provenance.py           # report
        python scripts/check_provenance.py --strict  # exit 1 on any problem
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "docs", "09_paper", "main.tex")
PAPERS = os.path.join(ROOT, "docs", "07_reference_papers")
LEDGER = os.path.join(PAPERS, "PROVENANCE.md")

OURS = "DA-ZVAD"          # our own rows are measurements, not external figures
TABLE_LABEL = "tbl:sota"


def ledger_rows():
    """(value, method, source, status) for every numeric ledger row."""
    rows, sources = [], {}
    for line in io.open(LEDGER, encoding="utf-8"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # source table: | Key | File | Supplies |
        if len(cells) == 3 and cells[0].startswith("`"):
            sources[cells[0].strip("`")] = cells[1].strip("`")
        # ledger table: | Value | Metric | Method | Source | Location | Status |
        if len(cells) == 6 and re.fullmatch(r"[0-9.]+", cells[0]):
            rows.append((cells[0], cells[2], cells[3].strip("`"), cells[5]))
    return rows, sources


def table_numbers():
    """(value, method) for every figure in the comparison table."""
    tex = io.open(TEX, encoding="utf-8").read()
    i = tex.index(TABLE_LABEL)
    block = tex[i:tex.index("end{tabularx}", i)]
    found = []
    for line in block.split("\n"):
        if "&" not in line or line.strip().startswith("\\multicolumn"):
            continue
        cells = [c.strip() for c in line.split("&")]
        method = re.sub(r"\\[a-zA-Z]+|[{}$^\\]", "", cells[0]).strip()
        for c in cells[1:]:
            for v in re.findall(r"\d+\.\d+", c):
                found.append((v, method))
    return found


def main() -> int:
    rows, sources = ledger_rows()
    known = {(v, m) for v, m, _s, _st in rows}
    status = {(v, m): st for v, m, _s, st in rows}
    problems = []

    for value, method in table_numbers():
        if OURS in method:
            continue                                   # our own measurement
        hit = next((k for k in known if k[0] == value and
                    (k[1] in method or method.startswith(k[1].split()[0]))), None)
        if hit is None:
            problems.append("NOT IN LEDGER   %-22s %s" % (method[:22], value))
        elif status[hit] != "verified":
            problems.append("UNVERIFIED      %-22s %s" % (method[:22], value))

    missing_pdf = [(k, f) for k, f in sources.items()
                   if not os.path.isfile(os.path.join(PAPERS, f))]

    print("comparison-table figures checked against the ledger")
    for p in problems:
        print("  " + p)
    if not problems:
        print("  all external figures present and marked verified")

    if missing_pdf:
        print("\nsource PDFs not in docs/07_reference_papers/")
        for k, f in missing_pdf:
            unver = sum(1 for v, m, s, st in rows if s == k and st != "verified")
            print("  %-14s %-46s (%d unverified figure(s) depend on it)"
                  % (k, f, unver))
    else:
        print("\nall source PDFs present")

    if "--strict" in sys.argv and (problems or missing_pdf):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
