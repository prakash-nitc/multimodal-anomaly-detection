# Positioning DA-ZVAD in Classical Domain-Adaptation Theory

> Grounded in the six survey papers in `SurveyPapersStudy/`. Companion to
> `domain_adaptation_video_ad.md` (which maps the VAD-specific DA literature);
> this note connects our work to the *general* DA theory those surveys establish.
> Study companion: `docs/08_understanding/03_domain_adaptation_deep_dive.md`.

---

## 1. The formal frame

A domain is a distribution $P(x,y) = P(y|x)\,P(x)$. Kouw (2018), as presented in Liu
et al. (2022) §2, decomposes domain shift into four types:

| Shift | Factor that changes | Canonical example |
|---|---|---|
| Covariate | $p(x)$ | different camera, lighting, weather |
| Conditional | $p(x|y)$ | the same class appears differently per domain |
| Label / target | $p(y)$ | class proportions differ |
| **Concept** | $p(y|x)$ | tomato = vegetable in one country, fruit in another |

Singhal et al. (2023) §II list the three theoretical conditions under which DA is
justified. The first is *covariate shift*: $P(Y_s|X_s) = Q(Y_t|X_t)$ — i.e. the
conditional label distribution is assumed **stable**. Concept shift is assumed away at
the foundations of the theory.

## 2. What the field admits about its own scope

Liu et al. (2022) §2, verbatim:

> "the concept shift (Kouw, 2018) can arise ... **it is, however, usually not a common
> problem in popular object classification or semantic segmentation tasks. As such, this
> review mainly focuses on the covariate shift alignment in UDA, as is most commonly
> studied.** The challenges of aligning the other shifts and their combinations are also
> discussed as directions for future research."

And on foundation models (§5.5):

> foundation models "are robust to the covariate shift in many cases" ... "it is
> challenging to alleviate the label shift, without access to target domain data. In
> addition, the **concept shift can also cause a problem**, even though there are
> sufficient training data."

Two admissions, both useful to us: (i) the field's machinery targets covariate shift;
(ii) in the foundation-model era, concept shift is explicitly named as open.

## 3. The mismatch between that scope and our task

In anomaly detection, **normality is defined by deployment context, not by object
identity**:

| Input $x$ | Domain A | Domain B |
|---|---|---|
| a person running | park → **normal** | bank vault → **anomalous** |
| a person lying down | beach → **normal** | factory aisle → **anomalous** |
| a vehicle | road → **normal** | pedestrian walkway → **anomalous** (a ShanghaiTech anomaly class) |

Identical $x$; the label flips with the domain. This is **concept shift** ($p(y|x)$
changes) by the exact definition above.

**Consequence:** the shift type that classical DA sets aside as uncommon is the shift
type that *dominates* video anomaly detection. Feature-alignment methods — the dominant
family across Wang & Deng (2018), Wilson & Cook (2020), Liu et al. (2022) — align
$p(x)$, and therefore cannot address a case where $p(x)$ is unchanged and only $p(y|x)$
moves.

## 4. Two adaptation paradigms, and where ours sits

The five vision surveys (2015–2023) and the LLM survey (Fan et al. 2025) describe
structurally different mechanisms:

| | Classical visual DA | LLM-era DA (Fan et al. 2025) |
|---|---|---|
| Mechanism | change the **model** to fit the target distribution | change the **input context**; model may stay frozen |
| Methods | discrepancy alignment, adversarial confusion, domain mapping, BN statistics, self-training | prompt engineering → RAG → DAPT → fine-tuning |
| Target data | required (unlabelled at minimum) | not necessarily required |
| Training | required | not required at the cheap end |
| Cost profile | high | an explicit **continuum**; prompt engineering is lowest-cost, and "often lacks depth and robustness" |

The vision surveys contain **no category** for adapting via a language description —
their taxonomies predate the foundation-model shift. Fan et al. establish that in the
LLM world this is a recognized strategy with a documented cost/benefit profile.

**DA-ZVAD's position, in the field's own vocabulary:** source-free, test-time,
training-free, **concept-shift-oriented**, **language-mediated** domain adaptation —
transplanting the LLM literature's lowest-cost adaptation strategy (context/prompt
engineering) into a vision task, aimed at the shift type visual DA excluded.

## 5. Gaps targeted (grounded, not invented)

| Gap | Basis in the surveys | Our response |
|---|---|---|
| **G-A. Concept shift dominates VAD but is unstudied** | Liu §2 scopes it out; §5.1/§5.5 list it open | Argument + mechanism (M3 verbalized context) + measurement |
| **G-B. No protocol measures language-mediated adaptation** | Classical protocol presumes a training step; none exists for "did a description adapt anything?" | The none/generic/**matched**/**mismatched** context sweep, with mismatched as a falsifying control |
| **G-C. Foundation-model-era DA under-characterized** | Liu §5.5 poses it as an open question | Empirical data point: which shifts a frozen VLM absorbs unaided |
| **G-D. Explanation quality under shift unexamined** | Absent from all six surveys; VERA evaluates in-domain only | Matched-vs-mismatched explanation gallery (qualitative) |

**Not claimed:** SOTA AUROC; a new adaptation algorithm; a solution to concept shift.

## 6. Proposed framing for the paper

> **Concept Shift, Not Covariate Shift: Characterizing Domain Adaptation for
> Training-Free Video Anomaly Detection**

The thesis is falsifiable (the mismatched-context arm can refute it), the contribution
is a characterization plus a measurement protocol rather than a leaderboard claim, and
every clause is supported by the surveys above.

## 7. References

1. V. M. Patel, R. Gopalan, R. Li, R. Chellappa, "Visual Domain Adaptation: A Survey of
   Recent Advances," *IEEE Signal Processing Magazine*, 2015.
2. M. Wang, W. Deng, "Deep Visual Domain Adaptation: A Survey," *Neurocomputing*,
   312:135–153, 2018.
3. G. Wilson, D. J. Cook, "A Survey of Unsupervised Deep Domain Adaptation," *ACM
   Computing Surveys*, 2020.
4. X. Liu, C. Yoo, F. Xing, H. Oh, G. El Fakhri, J.-W. Kang, J. Woo, "Deep Unsupervised
   Domain Adaptation: A Review of Recent Advances and Perspectives," *APSIPA
   Transactions on Signal and Information Processing*, 2022. arXiv:2208.07422
5. P. Singhal, R. Walambe, S. Ramanna, K. Kotecha, "Domain Adaptation: Challenges,
   Methods, Datasets, and Applications," *IEEE Access*, 11, 2023.
6. L. Fan, F. Liu, C. Chen, "Domain Adaptation of Large Language Models for Geotechnical
   Applications," 2025.
7. W. M. Kouw, "An Introduction to Domain Adaptation and Transfer Learning," 2018
   (shift taxonomy, via [4]).
