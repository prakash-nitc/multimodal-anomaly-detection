# 09_paper — the living research document

The single-column document maintained for supervisor review. It grows as findings
arrive; it is never rewritten from scratch.

## Files

| File | Purpose |
|---|---|
| `main.tex` | The document. Sections 1–10 in the agreed order. |
| `references.bib` | Bibliography. Add entries here, cite with `\cite{key}`. |
| `dazvad_architecture.png` | Figure 1. Regenerate via `scripts/make_architecture_diagram.py`. |

## Putting it on Overleaf (first time)

1. Overleaf → **New Project → Upload Project**.
2. Zip these three files together (no enclosing folder) and upload, **or** create a
   blank project and upload the three files individually.
3. Set the main document to `main.tex` (Menu → Main document) if not detected.
4. Compile. Run twice so citations resolve.
5. Menu → **Share** → *Turn on link sharing* → copy the **"Anyone with this link
   can view"** URL and send it to the supervisor.

> Give the supervisor the **read-only** link unless he asks to edit. He can then
> check progress at any time without needing an account.

## Maintaining it (every time you have a finding)

1. **Fill in the content** — replace a `\pending{...}` marker with the real result.
   Never delete a marker without filling it; never write a number that has not been
   measured.
2. **Update the status table** (§ Document Status) if a section's state changed.
3. **Add a revision-history row** — date + one line on what changed. This is what
   makes progress visible to the supervisor between meetings.
4. Recompile.

## Conventions used in the document

| Macro | Renders as | Use |
|---|---|---|
| `\pending{what is needed}` | red `[PENDING: ...]` | Content not yet produced |
| `\done` `\inprog` `\notstarted` | status keywords | The status table |
| `\note{...}` | grey italic | An aside to the supervisor or to yourself |

**The rule that matters:** results appear in this document only after they have been
measured. Everything unmeasured stays behind a `\pending` marker. That is what makes
the document trustworthy to a reader who is checking it periodically.
