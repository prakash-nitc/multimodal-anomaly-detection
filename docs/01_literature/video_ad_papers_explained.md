# Post-Midsem Video AD Papers — Viva-Ready Explanations

---

# Paper 1: OVVAD
**Open-Vocabulary Video Anomaly Detection**
Wu et al. | CVPR 2024 | Weakly Supervised

---

## One-Liner
*"OVVAD keeps CLIP frozen and adds a tiny temporal adapter on top to detect anomalies in video — including anomaly types it has never seen during training."*

## The Problem It Solves
Traditional video AD methods can only detect anomaly categories they were trained on. If you train on "fighting" and "robbery", the system can't detect "arson" at test time. OVVAD solves this by making detection **open-vocabulary** — it can detect novel anomaly types never seen during training.

## How It Works (4 Steps)

```
Step 1: Freeze CLIP
   Video frames → Frozen CLIP ViT-B/16 → Frame embeddings
   (CLIP is NOT fine-tuned — preserves zero-shot ability)

Step 2: Temporal Adapter
   Frame embeddings → Lightweight adapter (<1M params) → Temporal features
   (Captures "what changed between frame t and t+1")

Step 3: Semantic Knowledge Injection (SKI)
   LLM generates text descriptions of anomaly categories
   → Injected into visual pipeline to bridge vision-language gap

Step 4: Anomaly Synthesis
   During training, creates FAKE anomaly videos for unseen categories
   → Model learns to generalise beyond training categories
```

## Key Architecture Details (for viva)

| Component | Detail |
|---|---|
| Visual backbone | Frozen CLIP ViT-B/16 (NOT fine-tuned) |
| Temporal adapter | <1M parameters, placed AFTER CLIP |
| Training | Weakly supervised (video-level labels only, not frame-level) |
| Innovation | Anomaly Synthesis creates pseudo-novel anomalies |
| Dataset | UCF-Crime (13 anomaly categories) |
| Result | **86.40% AUC** on UCF-Crime |

## Why It Matters for DA-ZVAD
OVVAD proved that you can **add temporal modelling on top of a frozen VLM** without destroying its zero-shot ability. We take this exact principle — freeze CLIP, add a lightweight adapter — as Module 1 + Module 2 of DA-ZVAD.

## Strengths & Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Open-vocabulary (detects unseen anomalies) | Still needs weakly-supervised training |
| Lightweight adapter (minimal compute) | No explanations — outputs score only |
| Frozen CLIP preserves generalization | Uses ViT-B/16 (smaller than our ViT-L/14) |

## Likely Viva Questions

**Q: What does "open-vocabulary" mean here?**
*"It means the system can detect anomaly types it was never trained on. For example, if trained on 'fighting' and 'robbery', it can still detect 'explosion' at test time, because it uses CLIP's language understanding to generalize."*

**Q: Why freeze CLIP? Why not fine-tune it?**
*"Fine-tuning CLIP on a small video dataset would overfit it to that specific domain and destroy its generalisation. The 400M parameter model was trained on 2 billion image-text pairs — we don't want to lose that knowledge. The lightweight adapter adds domain-specific temporal understanding without touching the base model."*

**Q: What is Semantic Knowledge Injection?**
*"It's a module that takes text descriptions of anomaly categories — either hand-written or LLM-generated — and injects that semantic knowledge into the visual feature pipeline. This helps the model understand what different anomaly types look like, even ones it hasn't seen visually."*

---
---

# Paper 2: LAVAD
**Harnessing Large Language Models for Training-Free Video Anomaly Detection**
Zanella et al. | CVPR 2024 | Fully Training-Free

---

## One-Liner
*"LAVAD converts video to text captions, then asks an LLM to reason about whether the described events are anomalous — zero training required."*

## The Problem It Solves
OVVAD still needs training data. What if you want to deploy immediately in a new environment with zero data? LAVAD achieves this by replacing ALL learned components with pretrained models used off-the-shelf.

## How It Works (4 Steps)

```
Step 1: Caption Each Frame
   Video frame → BLIP-2 → "A person walking in a parking lot"
   (Every frame gets a text description)

Step 2: Group Captions Temporally
   Captions from frames t-3 to t+3 are grouped into a "snippet"
   → "Frame 1: A person walking. Frame 2: A person pulling out a weapon.
      Frame 3: A person pointing a gun at another person."

Step 3: LLM Scores the Snippet
   Prompt: "Rate how anomalous this sequence is from 0 to 1"
   LLM output: "0.95 — this appears to be an armed robbery"

Step 4: Cross-Modal Refinement
   Raw LLM scores are noisy (because captions hallucinate)
   → Use CLIP video-text similarity to refine/validate scores
   → Only keep scores where visual evidence matches the caption
```

## Key Architecture Details (for viva)

| Component | Detail |
|---|---|
| Captioning model | BLIP-2 (pretrained, frozen) |
| Reasoning model | LLM (GPT/LLaMA, frozen) |
| Training required | **NONE** — fully training-free |
| Key innovation | Cross-modal refinement to fix hallucinated captions |
| Temporal window | Snippet of ~7 adjacent frame captions |
| Result | **Beats all unsupervised VAD methods** on UCF-Crime |

## The Pipeline Visually

```
Frames → [BLIP-2] → Captions → [Group] → Snippets → [LLM] → Scores
                                                         ↓
                                              [Cross-Modal Refinement]
                                                         ↓
                                                   Final Scores
```

## Why It Matters for DA-ZVAD
LAVAD proves that **an LLM can be the anomaly reasoning engine** through language alone — no learned classifier needed. We adopt this caption-then-reason paradigm as Module 4 (LLM Reasoning Head) of DA-ZVAD.

## Strengths & Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Zero training — deploy anywhere instantly | Captions can be noisy/hallucinated |
| LLM provides temporal reasoning | High inference cost (LLM per snippet) |
| Beats supervised methods | No domain adaptation mechanism |
| Partial explanations via temporal summaries | Dependent on caption quality |

## Likely Viva Questions

**Q: Why not just use CLIP directly for scoring? Why convert to text first?**
*"CLIP gives a similarity score but can't reason about temporal events. By converting to captions first, we can feed a sequence of events to an LLM which can reason like: 'Frame 1 shows a person walking, frame 5 shows them running, frame 10 shows them attacking someone — this is a developing assault.' CLIP alone would score each frame independently and miss the temporal pattern."*

**Q: What is cross-modal refinement?**
*"BLIP-2 sometimes hallucinated — it might caption a normal scene as 'a person fighting' due to visual ambiguity. Cross-modal refinement uses CLIP's video-text similarity to check: does the original video actually match the caption? If CLIP similarity is low, the caption was likely wrong, and we downweight that LLM score."*

**Q: How does LAVAD handle temporal context if it doesn't have a temporal adapter?**
*"Instead of learning temporal features from visual data (like OVVAD), LAVAD delegates temporal reasoning to the LLM. It groups captions from adjacent frames into snippets and asks the LLM to reason about the sequence of events described in text. The LLM inherently understands temporal narratives from its training data."*

---
---

# Paper 3: VERA
**Explainable Video Anomaly Detection via Verbalized Learning**
Ye et al. | CVPR 2025 | Training-Free, Explainable

---

## One-Liner
*"VERA adapts anomaly detection to any new environment by simply describing the environment in text — 'a dimly lit warehouse with conveyor belts' — without changing any model parameters."*

## The Problem It Solves
Both OVVAD and LAVAD treat all environments the same. But "normal" depends on context:
- A person **running** is normal in a park, anomalous in a hospital
- A **forklift** is normal in a warehouse, anomalous on a sidewalk

VERA makes detection **context-aware** through text descriptions alone.

## How It Works (3 Steps)

```
Step 1: Verbalize the Deployment Context
   Human expert writes: "A dimly lit warehouse corridor with 
   conveyor belts and industrial shelving. Workers wear 
   orange safety vests. Forklifts operate in designated lanes."

Step 2: Context-Relative Scoring
   VLM processes each frame RELATIVE to the verbalized context
   → "Is this frame anomalous given that this is a warehouse?"
   → Running person in warehouse = ANOMALOUS (score: 0.92)
   → Running person in gym = NORMAL (score: 0.12)

Step 3: Explainable Output
   System doesn't just output a score — it explains:
   "Anomaly detected: Person running in restricted warehouse zone. 
    Normal behavior in this context: walking at controlled pace."
```

## Key Architecture Details (for viva)

| Component | Detail |
|---|---|
| Input | Video + text description of environment |
| Model | VLM (frozen, no training) |
| Domain adaptation | Through text prompts ONLY — no params changed |
| Explainability | Full natural language explanation with every detection |
| Key innovation | Verbalized context makes "normal" domain-specific |
| Cross-domain | Works across surveillance → industrial → traffic without retraining |

## The Key Insight (memorize this)

> **"Normal is not absolute — it's relative to context. VERA makes context a first-class input through natural language."**

## Why It Matters for DA-ZVAD
VERA provides Module 3 (Verbalized Domain Context) of DA-ZVAD. It's the most elegant form of domain adaptation — no training, no data, just a text description. It also gives us the explainability we need for industrial deployment.

## Strengths & Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Zero training, zero data for new domains | Needs human to write scene description |
| Full natural language explanations | Less explored quantitatively |
| Cross-domain robust | Dependent on VLM quality |
| Adapts at inference time | No temporal adapter (implicit only) |

## Likely Viva Questions

**Q: How is this different from just changing the prompt in CLIP?**
*"Regular CLIP prompts say what to look for — 'a damaged bottle'. VERA prompts describe WHERE you're looking — 'a dimly lit warehouse'. This shifts the entire baseline of what's normal. It's not about detecting specific objects, it's about understanding context-dependent normality."*

**Q: Who writes the verbalized description?**
*"In deployment, the site engineer or security operator writes a 2-3 sentence description of the environment when installing the system. This is a one-time effort per deployment site. Alternatively, the system could use an LLM to auto-generate the description from a few sample frames."*

**Q: Can VERA handle changing environments — like day vs night?**
*"Yes — you can have multiple verbalized contexts and switch between them. 'Daytime: busy parking lot with cars and pedestrians' vs 'Nighttime: empty parking lot, no pedestrians expected'. The system applies the appropriate context at the appropriate time."*

---
---

# Paper 4: Gao et al. VAD Survey
**A Comprehensive Survey on Video Anomaly Detection**
Gao et al. | ACM Computing Surveys 2025 | Journal Paper

---

## One-Liner
*"The most comprehensive survey on video anomaly detection to date — covers 400+ methods and identifies open-vocabulary VAD as the key future direction."*

## Why This Paper Matters
This is the **journal paper** in our literature review. ACM Computing Surveys is a top-tier venue (Impact Factor ~16). This paper validates our entire research direction.

## Key Taxonomy (memorize this)

The survey categorizes ALL video AD methods into:

```
Video Anomaly Detection Methods
├── Supervised Methods (need frame-level labels — impractical)
├── Weakly Supervised (video-level labels — practical)
│   ├── MIL-based (Multiple Instance Learning)
│   ├── Attention-based
│   └── CLIP-based (← our category, includes OVVAD)
├── Unsupervised / One-Class
│   ├── Reconstruction-based (autoencoders)
│   ├── Prediction-based (future frame prediction)
│   └── Memory-based
└── Zero-Shot / Open-Vocabulary (← emerging, our target)
    ├── VLM-based (LAVAD, VERA)
    └── LLM-based (caption-then-reason)
```

## Key Findings from the Survey

1. **Open-vocabulary VAD is the key emerging direction** — traditional methods fail when anomaly types are not predefined
2. **VLMs (CLIP) are the most promising backbone** for zero-shot detection
3. **Domain adaptation remains under-explored** — most methods assume fixed deployment environment
4. **Explainability is almost absent** in existing methods
5. **Training-free methods are catching up** to supervised ones

## Why This Validates DA-ZVAD

The survey explicitly identifies **four gaps** that DA-ZVAD addresses:

| Survey Gap | DA-ZVAD Solution |
|---|---|
| "Open-vocabulary VAD is underexplored" | OVVAD's open-vocabulary detection |
| "Domain adaptation in VAD needs more work" | VERA's verbalized prompting |
| "Explainability is lacking" | VERA's natural language explanations |
| "Training-free methods show promise" | LAVAD's zero-training pipeline |

## Likely Viva Questions

**Q: Why did you include a survey paper?**
*"ACM Computing Surveys is a top-tier journal with an impact factor of ~16. Including it shows we understand the broader landscape, not just individual papers. More importantly, the survey's identified gaps directly map to our DA-ZVAD contributions."*

**Q: What does the survey say about the future of VAD?**
*"It identifies three key trends: (1) moving from fixed-vocabulary to open-vocabulary detection using VLMs, (2) incorporating LLMs for reasoning and explainability, and (3) developing domain-adaptive methods that work across different environments. Our DA-ZVAD addresses all three."*

---
---

# COMPARATIVE CHEAT SHEET

| Dimension | OVVAD | LAVAD | VERA | Survey |
|---|---|---|---|---|
| **Venue** | CVPR '24 | CVPR '24 | CVPR '25 | ACM Surveys '25 |
| **Training** | Weakly supervised | Training-free | Training-free | — |
| **Backbone** | Frozen CLIP ViT-B/16 | BLIP-2 + LLM | VLM (frozen) | — |
| **Temporal** | Learned adapter | LLM aggregation | Implicit | — |
| **Domain adapt.** | SKI + synthesis | None | Verbalized text | — |
| **Explainable** | No | Partial | Yes | — |
| **Key strength** | Temporal features | Zero training | Domain-adaptive | Landscape view |
| **DA-ZVAD role** | Module 1+2 | Module 4 | Module 3 | Justification |

---

# MEMORY TRICK: How They Connect

Think of it as building a **surveillance camera system**:

1. **OVVAD** = the **camera** (sees temporal patterns in video)
2. **LAVAD** = the **analyst** (reasons about what happened)
3. **VERA** = the **site manual** (tells the system what's normal HERE)
4. **Survey** = the **industry report** (proves this approach is needed)

Together → **DA-ZVAD** = a complete surveillance system that sees, reasons, adapts, and explains.
