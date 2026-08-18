# 📖 08_understanding/ — plain-language technical notes

One note per completed phase, written for **understanding**, not for show. Read the note
for a phase and you should be able to explain that phase's work to anyone — guide, panel,
or interviewer — without opening the code.

Every note follows the same structure:

1. **What we built** — the artifact, in one paragraph
2. **Why** — the reasoning behind the design choices
3. **How it works** — plain-language walkthrough (no jargon without explanation)
4. **Key terms** — the vocabulary you must be comfortable with
5. **What to say if asked** — likely questions with short, honest answers
6. **Self-quiz** — recall questions (answers at the end); do them without looking back
7. **What's next** — how this phase feeds the next one

Reading is not enough for a review: finish each note's self-quiz, and before any
panel/viva, run at least two mock-QA rounds against the collected questions.

**Read in order** — later notes assume the earlier ones:

| Note | Covers |
|---|---|
| [00_ml_basics.md](00_ml_basics.md) | ML from zero: training vs inference, embeddings, softmax, ViT, contrastive learning, metrics — each tied to where it appears in our project |
| [01_foundations.md](01_foundations.md) | The core idea, CLIP, the DA-ZVAD architecture, the framework code, the MVTec baseline |
| [02_experiments_and_gaps.md](02_experiments_and_gaps.md) | The DA landscape (four lanes), the three experiments (grid, context sweep with mismatched negative control, explanations under shift), micro vs macro AUROC |
| [03_domain_adaptation_deep_dive.md](03_domain_adaptation_deep_dive.md) | **Study guide for the guide's 6 survey papers**: formal DA theory, the four shift types, classical vs LLM-era adaptation, DA across fields, and why concept shift is our genuine gap. Includes oral-defense targets + self-quiz |

*(New notes are added here as phases complete.)*

**Concepts-first rule:** every phase note introduces any *new* concept in plain language
before showing the implementation that uses it — you should never meet an unexplained term
after reading in order.
