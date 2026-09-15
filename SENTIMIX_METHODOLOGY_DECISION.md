# SentiMix Methodology Adaptation & 3-Class Extension Decision

## 1. Problem: Why Binary Scalar Fusion Cannot Represent Three Classes
The original paper's research specification defines binary sentiment classification ($y \in \{0, 1\}$) using a scalar interpolation equation:
$$S_{\text{final}} = \alpha \cdot S_{\text{VADER}} + (1 - \alpha) \cdot S_{\text{DistilBERT}}$$
where $S_{\text{VADER}}, S_{\text{DistilBERT}} \in [0, 1]$ represent 1D scalar probabilities of the positive sentiment class, and $\hat{y} = \mathbb{I}(S_{\text{final}} \ge 0.50)$.

In the SemEval-2020 Task 9 (SentiMix Hinglish) dataset, labels form a 3-class partition:
$$\mathcal{Y} = \{\text{Positive (0)}, \text{Negative (1)}, \text{Neutral (2)}\}$$
A single 1D scalar score $S \in [0, 1]$ cannot mathematically represent a 3-class categorical distribution without imposing arbitrary intermediate interval assumptions (e.g., $[0, \tau_1)$ for negative, $[\tau_1, \tau_2]$ for neutral, $(\tau_2, 1]$ for positive), which lack empirical justification and cannot capture multi-class uncertainty.

---

## 2. Why Neutral Is Preserved
Neutral constitutes **37.46%** (7,492 / 20,000) of the total SentiMix Hinglish dataset:
- Train: 5,264 / 14,000 (37.60%)
- Dev: 1,128 / 3,000 (37.60%)
- Test: 1,100 / 3,000 (36.67%)

Neutral is the plurality class in the dataset. Under strict research integrity:
- Silently collapsing Neutral into Positive or Negative corrupts the semantic ground truth.
- Dropping Neutral would discard over 37% of the official benchmark data and violate the official SemEval-2020 Task 9 competition evaluation protocol.
- Therefore, Neutral is preserved as an explicit, first-class category ($y=2$).

---

## 3. The Exact 3-Class Extension
To represent three classes while preserving the paper's linear dynamic interpolation principle, the scalar equation is extended to the 2-dimensional probability simplex $\Delta^2$:
$$\mathbf{P}_{\text{final}} = \alpha \cdot \mathbf{P}_{\text{VADER}} + (1 - \alpha) \cdot \mathbf{P}_{\text{DistilBERT}}$$
where:
$$\mathbf{P}_{\text{VADER}} = [P_{\text{pos}}^{\text{vader}}, P_{\text{neg}}^{\text{vader}}, P_{\text{neu}}^{\text{vader}}]^T \in \Delta^2$$
$$\mathbf{P}_{\text{DistilBERT}} = [P_{\text{pos}}^{\text{bert}}, P_{\text{neg}}^{\text{bert}}, P_{\text{neu}}^{\text{bert}}]^T \in \Delta^2$$
$$\sum_{k=0}^2 P_{k} = 1, \quad P_k \ge 0$$

The classification decision rule is:
$$\hat{y} = \arg\max_{k \in \{0, 1, 2\}} P_{k, \text{final}}$$
where class indices are strictly:
- `0`: Positive
- `1`: Negative
- `2`: Neutral

---

## 4. The Role of $\alpha$
The coefficient $\alpha$ retains **exactly** the identical role, mathematical definition, and bounds as in the original paper:
- $\alpha$ is the noise-adaptive contribution weight of the rule-based lexicon model.
- $(1 - \alpha)$ is the contribution weight of the contextual transformer model.
- Routing formula:
  - If $N \le 0.20 \implies \alpha = 0.02$
  - If $N > 0.20 \implies z = 0.05 \cdot (20 - L) + 12.0 \cdot N, \quad \alpha_{\text{raw}} = \sigma(z), \quad \alpha = \text{clamp}(\alpha_{\text{raw}}, 0.02, 0.25)$
- $\alpha$ is **not** tuned, re-parameterized, or trained. The identical router function is used.

---

## 5. How VADER Is Represented
NLTK's VADER analyzer calculates rule-based token valence intensities and returns native dictionary proportions:
`{'neg': float, 'neu': float, 'pos': float, 'compound': float}`
In standard VADER, `pos`, `neg`, and `neu` represent the proportion of text tokens scoring as positive, negative, and neutral, summing to ~1.0.

The `VADER3ClassAdapter` takes these native proportions and applies deterministic L1 normalization:
$$\text{total} = \text{pos} + \text{neg} + \text{neu}$$
$$P_{\text{pos}}^{\text{vader}} = \frac{\text{pos}}{\text{total}}, \quad P_{\text{neg}}^{\text{vader}} = \frac{\text{neg}}{\text{total}}, \quad P_{\text{neu}}^{\text{vader}} = \frac{\text{neu}}{\text{total}}$$
If $\text{total} = 0$ (e.g. empty string), it defaults deterministically to $[0.0, 0.0, 1.0]$ (100% neutral).

---

## 6. Why the VADER Adapter Is Deterministic and Non-Learned
- **Zero Learned Parameters**: No weights, biases, or regression coefficients are trained on SentiMix data.
- **Zero Dataset-Fitted Thresholds**: No ad-hoc thresholds (e.g. compound $\pm 0.05$) are used to manufacture fake probabilities.
- **Scientific Honesty**: The representation is explicitly documented as a *lexical proportion vector*, not a calibrated posterior probability distribution.

---

## 7. Why No New Learned Fusion Parameters Are Introduced
Introducing neural gating networks, attention weights, or learned mixing layers would violate the paper's core scientific contribution: showing that an explicit, interpretable, deterministic noise-aware mathematical router can dynamically balance lexicon and transformer signals. The vector fusion retains the exact convex combination structure without adding any learned degrees of freedom.

---

## 8. Which Original Components Remain Unchanged
1. **Multi-Dimensional Noise Quantification (`src/features/noise_quantifier.py`)**:
   - $E = N_{\text{emoji}} / N_{\text{tokens}}$
   - $R = N_{\text{repeat}} / N_{\text{tokens}}$
   - $C = \min(\text{Eng}, \text{Hin}) / (\max(\text{Eng}, \text{Hin}) + 1)$
   - $S = N_{\text{symbol}} / N_{\text{tokens}}$
   - $N = 0.25E + 0.25R + 0.30C + 0.20S$
   - All weights ($0.25, 0.25, 0.30, 0.20$) and Hinglish marker lexicons remain 100% identical.
2. **Noise-Aware Adaptive Router (`src/models/adaptive_router.py`)**:
   - $L_0 = 20, \quad w_1 = 0.05, \quad w_2 = 12.0, \quad N_{\text{threshold}} = 0.20, \quad \alpha \in [0.02, 0.25]$
3. **Binary VADER Model (`src/models/vader_model.py`)**: Completely preserved and isolated.
4. **Binary DistilBERT Model (`src/models/distilbert_model.py`)**: Completely preserved and isolated.
5. **Binary Fusion Framework (`src/models/dynamic_fusion.py`)**: Completely preserved and isolated.
6. **Inference API Pipeline (`src/pipeline.py`)**: Completely preserved and isolated.

---

## 9. Which Components Were Minimally Extended
1. **SentiMix Loader (`src/data/sentimix_loader.py`)**: Parses CoNLL-formatted tweets preserving surface text, tokens, language tags, and 3-class labels.
2. **DistilBERT 3-Class Architecture (`src/models/distilbert_3class.py`)**: Configures 3-class classification head (`num_labels=3`) with classes `[0: pos, 1: neg, 2: neu]` and standard research hyperparameters (epochs=3, batch=16, lr=2e-5, max_length<=128).
3. **VADER 3-Class Adapter (`src/models/vader_3class.py`)**: Maps native lexical proportions to an L1-normalized 3-element vector.
4. **3-Class Vector Fusion Framework (`src/models/fusion_3class.py`)**: Implements $\mathbf{P}_{\text{final}} = \alpha \mathbf{P}_{\text{VADER}} + (1 - \alpha) \mathbf{P}_{\text{DistilBERT}}$ and orchestrates dynamic, static, DistilBERT-only, and VADER-only ablations.
5. **3-Class Metrics (`src/evaluation/metrics_3class.py`)**: Computes Macro-F1, per-class F1, Accuracy, and 3x3 Confusion Matrix, along with optional secondary non-neutral subset metrics.

---

## 10. Remaining Methodological Limitations
1. **VADER Lexical Bias on Code-Mixed Text**: VADER's English lexicon has limited coverage over Romanized Hindi (Hinglish) tokens; the router naturally attenuates VADER's influence when noise $N$ is low ($\alpha=0.02$), but VADER's proportion vector remains uncalibrated.
2. **Evaluation Metric Alignment**: Official SemEval-2020 Task 9 benchmarks report Macro F1 over all 3 classes. Comparing directly to 2-class SAIL 2017 literature figures is invalid due to different label spaces and dataset distributions.

---

## 11. Paper Fidelity Summary
| Paper Specification Element | SentiMix Adaptation Implementation | Fidelity Status |
|-----------------------------|------------------------------------|-----------------|
| Noise Formula $N$ | Exact same $0.25E + 0.25R + 0.30C + 0.20S$ | 100% Identical |
| Router Formula $\alpha$ | Exact same $w_1(20-L) + w_2 N$, clamp [0.02, 0.25] | 100% Identical |
| Fusion Concept | $\alpha \cdot \text{Lexicon} + (1 - \alpha) \cdot \text{Transformer}$ | Preserved via vector extension on $\Delta^2$ |
| DistilBERT Backbone | `distilbert-base-uncased`, 3 epochs, lr=2e-5 | 100% Aligned |
| Native Splits | SentiMix official 14k train / 3k dev / 3k test | 100% Preserved |
| Ablation Variants | Full Dynamic, Static Alpha, DistilBERT-only, VADER-only | 100% Preserved |
| Binary Pipeline | `src/models/dynamic_fusion.py` | 100% Untouched |
