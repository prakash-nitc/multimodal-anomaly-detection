# -*- coding: utf-8 -*-
"""Find sentences in main.tex that are hard to read aloud.

Length alone is a weak signal, so this also counts clause boundaries: commas,
semicolons, dashes and subordinating words. A long sentence built of short
coordinate clauses reads fine; a medium one with four levels of subordination
does not, and it is the second kind that cannot be explained on demand.
"""
import re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"p:\Research\multimodal-anomaly-detection\docs\09_paper\main.tex"
raw = open(P, encoding="utf-8").read().split("\n")

# body only: drop comments, tables, math blocks, algorithm, preamble
body, keep, skip_env = [], False, 0
SKIP = ("tabularx", "table", "algorithm", "algorithmic", "equation", "figure")
for i, line in enumerate(raw, 1):
    s = line.strip()
    if s.startswith(r"\section{Introduction}"):
        keep = True
    if not keep or s.startswith("%"):
        continue
    if any(s.startswith(rf"\begin{{{e}") for e in SKIP):
        skip_env += 1
    if skip_env:
        if any(s.startswith(rf"\end{{{e}") for e in SKIP):
            skip_env -= 1
        continue
    body.append((i, line))

# stitch into text, remembering the line each sentence starts on
text, marks = "", []
for ln, line in body:
    marks.append((len(text), ln))
    text += line + " "

def line_of(pos):
    out = marks[0][1]
    for p, ln in marks:
        if p <= pos:
            out = ln
        else:
            break
    return out

# strip LaTeX so word counts reflect prose
clean = re.sub(r"\\cite[pt]?\{[^}]*\}", "CITE", text)
clean = re.sub(r"\\(?:eq)?ref\{[^}]*\}", "REF", clean)
clean = re.sub(r"\$[^$]*\$", "MATH", clean)
clean = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{)?", " ", clean)
clean = re.sub(r"[{}]", " ", clean)
clean = re.sub(r"\s+", " ", clean)

SUB = re.compile(r"\b(which|that|because|although|whereas|while|since|so that|"
                 r"rather than|in which|where|when|if|unless|therefore|however|"
                 r"thereby|thus|hence)\b", re.I)

rows = []
for m in re.finditer(r"[^.!?]+[.!?]", clean):
    s = m.group().strip()
    words = len(s.split())
    if words < 12:
        continue
    clauses = s.count(",") + s.count(";") + s.count("---") + len(SUB.findall(s))
    # rough difficulty: long AND heavily subordinated
    score = words * (1 + 0.35 * clauses)
    rows.append((score, words, clauses, line_of(m.start()), s))

rows.sort(reverse=True)
print(f"{'score':>6}{'words':>7}{'clauses':>9}{'line':>7}   sentence")
print("-" * 100)
for score, w, c, ln, s in rows[:18]:
    print(f"{score:>6.0f}{w:>7}{c:>9}{ln:>7}   {s[:200]}")

print(f"\ntotal sentences >=12 words : {len(rows)}")
print(f"over 40 words              : {sum(1 for r in rows if r[1] > 40)}")
print(f"over 30 words              : {sum(1 for r in rows if r[1] > 30)}")
print(f"median length              : {sorted(r[1] for r in rows)[len(rows)//2]}")
