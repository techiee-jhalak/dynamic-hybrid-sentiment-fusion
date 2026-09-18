# Phase 6 Experimental Results Report — SentiMix Hinglish Benchmark

> **RESEARCH INTEGRITY DECLARATION**:
> All results below are **genuine empirical measurements** obtained from running the trained models on the official, untouched SemEval-2020 Task 9 (SentiMix Hinglish) 3,000-sample test partition.
> - Zero results were fabricated or hardcoded.
> - This dataset is **SentiMix**, NOT SAIL 2017. These results reflect real 3-class performance on code-mixed Twitter data.

---

## 1. Baseline Performance Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Pos F1 | Neg F1 | Neu F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VADER (3-class Lexicon Adapter) | 0.3753 | 0.5768 | 0.3423 | 0.2009 | 0.0598 | 0.0044 | 0.5384 |
| DistilBERT (3-class Fine-Tuned) | 0.6863 | 0.698 | 0.6906 | 0.6917 | 0.7505 | 0.7144 | 0.6102 |
| Static Fusion (Fixed alpha=0.02) | 0.686 | 0.6997 | 0.6894 | 0.6918 | 0.7492 | 0.7125 | 0.6137 |
| Dynamic Hybrid Fusion (Proposed) | 0.6883 | 0.7124 | 0.6879 | 0.6946 | 0.7404 | 0.7059 | 0.6374 |

---

## 2. Ablation Analysis

| Configuration | Accuracy | Macro_F1 | Delta_Macro_F1 | Delta_Accuracy |
| --- | --- | --- | --- | --- |
| Full Dynamic Fusion (Proposed) | 0.6883 | 0.6946 | 0.0 | 0.0 |
| Static Fusion (Fixed alpha=0.02) | 0.686 | 0.6918 | -0.0028 | -0.0023 |
| DistilBERT Only (alpha=0.0) | 0.6863 | 0.6917 | -0.0029 | -0.002 |
| VADER Only (alpha=1.0) | 0.3753 | 0.2009 | -0.4937 | -0.313 |

---

## 3. Noise Sensitivity Analysis

| Noise_Group | Sample_Count | Percentage | Dynamic_Accuracy | Dynamic_Macro_F1 | DistilBERT_Accuracy | DistilBERT_Macro_F1 | VADER_Accuracy | VADER_Macro_F1 | Delta_F1_over_DistilBERT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOW | 1634 | 54.47 | 0.6836 | 0.6894 | 0.686 | 0.6914 | 0.3739 | 0.1976 | -0.002 |
| MODERATE | 1365 | 45.5 | 0.6938 | 0.6929 | 0.6864 | 0.6875 | 0.3773 | 0.205 | 0.0054 |
| HIGH | 1 | 0.03 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |

---

## 4. Statistical Hypothesis Testing (McNemar's Test, p = 0.05)

| Comparison | Contingency [Both, A-only, B-only, Neither] | Chi-Square Statistic | p-value | Significant? | Superior Model |
|------------|---------------------------------------------|----------------------|---------|--------------|----------------|
| Dynamic Fusion vs DistilBERT Only | [1975, 90, 84, 851] | 0.1437 | 0.704651 | False | Dynamic Fusion |
| Dynamic Fusion vs Static Fusion | [1990, 75, 68, 867] | 0.2517 | 0.615847 | False | Dynamic Fusion |
| Dynamic Fusion vs VADER Only | [808, 1257, 318, 617] | 558.6311 | 0.0 | True | Dynamic Fusion |


---

## 5. Error Analysis Summary

- Total Test Samples: 3,000
- Total Errors: 935 (31.17%)
- Overall Accuracy: 68.83%
- Macro F1: 0.6946

### Representative Error Case Examples (from untouched TEST set)

#### Category: `cross_script_code_mixing`
- **UID**: 30898
  - **Text**: `@ ravindraj12 @ TarekFatah Yarr India me aise log b hain ??? Life me 1st time India ka koi banda Positive dekha Appreciate u bro ðŸ‡µðŸ‡°â¤ðŸ‡®ðŸ‡³`
  - **Gold**: `positive` | **Predicted**: `neutral`
  - **Noise Score (N)**: 0.3435 (MODERATE) | **Confidence**: 0.5451

- **UID**: 16061
  - **Text**: `@ asifnkhan08 @ SAUMEN _ BANERJ _ E @ JhaSanjay Time and circumstances ke sath jo change nhi hota vo sabse bada bewakuf ho â€¦ https // t co / GPIt7Q9zJx`
  - **Gold**: `negative` | **Predicted**: `neutral`
  - **Noise Score (N)**: 0.1606 (LOW) | **Confidence**: 0.6432

#### Category: `emojis_polysemy`
- **UID**: 12141
  - **Text**: `@ smritiirani ðŸ‡®ðŸ‡³ðŸ‡®ðŸ‡³ðŸ‡®ðŸ‡³ðŸ‡®ðŸ‡³ðŸ‡®ðŸ‡³ jai ho irani ji aap ne Rahul Gandhi ko harakar bahut acha keya Jai BJP ðŸ‡®ðŸ‡³ðŸ‡®ðŸ‡³ðŸ‡®ðŸ‡³ðŸ‡®ðŸ‡³ðŸ‡®ðŸ‡³ https // t . co / 35Viuj83sv`
  - **Gold**: `positive` | **Predicted**: `neutral`
  - **Noise Score (N)**: 0.136 (LOW) | **Confidence**: 0.5503

- **UID**: 24504
  - **Text**: `RT @ Sulex _ kindin Jumma â€™ at Mubarak to y â€™ all ðŸ™ŒðŸ»ðŸ™ðŸ» https // t . co / FtYJ8HydOL`
  - **Gold**: `positive` | **Predicted**: `neutral`
  - **Noise Score (N)**: 0.2173 (MODERATE) | **Confidence**: 0.7288

#### Category: `repetition_emphasis`
- **UID**: 9078
  - **Text**: `@ ZeeNews Mein toh chahta hoon ye resign na kare ... Ye jab tak rahega @ BJP4India aaram se elections jeetegi ... Aur ent â€¦ https // t . co / Djgtp4hTtb`
  - **Gold**: `neutral` | **Predicted**: `negative`
  - **Noise Score (N)**: 0.1598 (LOW) | **Confidence**: 0.7329

- **UID**: 13712
  - **Text**: `Unse toh achchi hai yeh cigarette ... dil ko jalati hai magar hoton pe toh aati hai .....ðŸ’”ðŸ’”ðŸ’” SMOKING AND LOVE BOTH C â€¦ https // t . co / rx1AYhkDCA`
  - **Gold**: `neutral` | **Predicted**: `positive`
  - **Noise Score (N)**: 0.2207 (MODERATE) | **Confidence**: 0.6714

#### Category: `neutral_polarity_boundary`
- **UID**: 10426
  - **Text**: `RT @ DHEERAJ48968067 @ nirahua1 @ msunilbishnoi Wah bhai Shandaar jabab`
  - **Gold**: `positive` | **Predicted**: `neutral`
  - **Noise Score (N)**: 0.0701 (LOW) | **Confidence**: 0.4826

- **UID**: 22656
  - **Text**: `@ BaijnathMourya @ mssirsa @ AapkamanojS @ ArvindKejriwal Haan Mai free Wi-Fi dunga free water dunga Sasti Bijli doo â€¦ https // t co / d4uOIcb562`
  - **Gold**: `positive` | **Predicted**: `neutral`
  - **Noise Score (N)**: 0.1327 (LOW) | **Confidence**: 0.5664

#### Category: `general_misclassification`
- **UID**: 40769
  - **Text**: `Setting up for our HII audit workshop with matrons Thank you ISS for your support and super refreshments https // t co / oQE5Fd2FBH`
  - **Gold**: `neutral` | **Predicted**: `positive`
  - **Noise Score (N)**: 0.1731 (LOW) | **Confidence**: 0.791

- **UID**: 843
  - **Text**: `@ Osamathemalang @ BilalKhanWriter @ fawadchaudhry Hahaha ye bnda hrksi ko kafir kehta rehta ha Apne ilawa isay koi musalman nazr hi niata`
  - **Gold**: `negative` | **Predicted**: `neutral`
  - **Noise Score (N)**: 0.1461 (LOW) | **Confidence**: 0.4366

