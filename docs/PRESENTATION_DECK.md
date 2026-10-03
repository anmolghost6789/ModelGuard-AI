# 🖥️ Capstone Project Presentation Deck: Slide-by-Slide Outline & Script

**Project Title:** *An AI-Based Framework for Predicting Machine Learning Model Failure Using Data Quality, Distribution Shift, and Prediction Uncertainty*  
**Presenter:** 4th-Year Computer Science Senior  
**Presentation Duration:** 15–20 Minutes (Final Year B.Tech / B.E. Project Defense)  

---

### Slide 1: Title Slide & Candidate Introduction
- **Slide Title:** An AI-Based Framework for Predicting Machine Learning Model Failure Using Data Quality, Distribution Shift, and Prediction Uncertainty
- **Subtitle:** Decoupled Multimodal Early-Warning System for Label-Free Operational Reliability
- **Candidate:** [Your Name] | Department of Computer Science & Engineering
- **Supervisor:** [Advisor Name] | Senior Capstone Project 2026
- **Spoken Script (1 min):**
  > "Respected committee members, advisor, and faculty. Today, I am presenting my final-year capstone project and research paper. Traditional machine learning evaluation assesses accuracy on static test datasets. But in production, models encounter distribution shifts, corrupted sensor readings, and novel operational regimes that lead to silent performance failure. Today, I present an AI-based early-warning framework that predicts model failure *before* or *during* deployment—without waiting for delayed ground-truth labels."

---

### Slide 2: Problem Statement: The Silent Failure Dilemma
- **Key Points:**
  - Machine learning models operate under the i.i.d. assumption ($P_{\text{train}} = P_{\text{test}}$), which is routinely violated in production.
  - Silent Failures: Models do not crash or throw exceptions when corrupted; they generate confident, incorrect predictions.
  - Ground-Truth Latency: True labels arrive days, weeks, or months late (e.g., credit defaults, long-term health outcomes, seasonal traffic).
  - Unsafe Deployment: Relying only on delayed labels leads to catastrophic operational failure.
- **Visual:** Flowchart showing "Data Shift $\rightarrow$ Silent Degraded Predictions $\rightarrow$ Delayed Ground Truth $\rightarrow$ Operational Damage".

---

### Slide 3: Research Gap: Why Current Solutions Fall Short
- **Key Points:**
  - *Univariate Drift Tools (Evidently, Alibi Detect):* Monitor marginal $P(X)$ shift, but trigger false alarms because benign shifts orthogonal to decision boundaries do not hurt accuracy.
  - *Uncertainty-Only Tools:* Monitor individual prediction entropy, but cannot diagnose macroscopic sensor corruptions or multivariate drift.
  - *Data Quality Tools:* Flag missing values, but tree ensembles can often bypass missing attributes using correlated backup features.
  - **The Missing Link:** A unified framework that learns the *interaction* between Data Quality, Distribution Shift, and Prediction Uncertainty to predict empirical performance collapse.

---

### Slide 4: Core Research Questions & Objectives
- **Research Questions:**
  - **RQ1:** Does distribution shift significantly affect model performance?
  - **RQ2:** Is prediction uncertainty statistically associated with model errors?
  - **RQ3:** Can combining data quality, distribution shift, and uncertainty improve early detection of model failure?
  - **RQ4:** Can the proposed failure-prediction system generalize to unseen future temporal holdouts?
- **Primary Objective:** Build a working, production-grade web dashboard and reproducible research pipeline that forecasts model failure probability ($0$ to $100$) in real time.

---

### Slide 5: System Architecture & Dataflow
- **Architecture Highlights:**
  - Ingestion & Automated Schema Profiler (`data_profiler.py`).
  - Decoupled Primary Model (Logistic Regression, Random Forest, XGBoost).
  - Telemetry Vector Extraction: 21 orthogonal operational signals.
  - Secondary Failure Meta-Classifier: Predicts $P(\text{Model Failure} \mid X_{\text{telemetry}})$.
  - 3-Tier Alert Triage: 🟢 **NORMAL** ($<35\%$), 🟡 **WARNING** ($35–65\%$), 🔴 **HIGH RISK** ($\ge 65\%$).
- **Visual:** Block diagram showing the dual-engine architecture.

---

### Slide 6: Dataset & Strict Leakage-Free Splitting
- **Benchmark Dataset:** Capital Bikeshare Real-World Empirical Log (17,379 Hourly Observations, Washington D.C., 2011–2012).
- **Target Variable:** High-Demand Surge Period ($cnt \ge 109.0$, historical 2011 median).
- **Leakage Prevention Protocols:**
  - Arithmetic identity columns (`casual` + `registered` = `cnt`, $r = 0.972$) permanently dropped.
  - Sequential ID `instant` dropped.
  - Year feature `yr` dropped to prevent temporal memorization.
  - Scalers fit strictly on 2011 Train data ($N=8,645$).
- **Partitions:** Train (2011), Validation (Q1 2012), Clean Test (Q2 2012), Future Holdout (H2 2012).

---

### Slide 7: Primary Model Benchmarking (Clean Unseen Test Data)
- **Table Summary:**
  - Logistic Regression: Acc $81.53\%$, F1 $85.51\%$, ROC-AUC $0.8971$.
  - Random Forest: Acc $92.48\%$, **F1 $94.36\%$**, **ROC-AUC $0.9831$**, Brier $0.0558$.
  - Gradient Boosting: Acc $92.94\%$, F1 $94.65\%$, ROC-AUC $0.9896$.
  - XGBoost: Acc $93.35\%$, F1 $94.98\%$, ROC-AUC $0.9891$.
- **Takeaway:** Random Forest established an operational baseline F1-score of **$0.9436$** ($94.36\%$).

---

### Slide 8: Controlled Stress Testing: Inducing Model Collapse
- **Experimental Disturbances Tested:**
  - Missing Values (MCAR $5\%–30\%$): Robust under median imputation (F1 remained $\approx 94.6\%$).
  - Gaussian Noise ($\sigma = 0.15 \rightarrow 0.50$): F1 degraded monotonically from $92.96\%$ down to $85.98\%$.
  - Directional Covariate Shift (Cold Snap $\Delta t = -0.25$): **Catastrophic collapse!** F1 plunged from $94.36\%$ to **$72.53\%$** (a **$23.14\%$ relative drop**).
- **Insight:** Shift severity and directionality dictate failure; tree ensembles resist benign noise but collapse under extreme covariate shifts.

---

### Slide 9: Distribution Drift Engine (KS, PSI & JSD)
- **Statistical Quantifiers:**
  - Continuous Features: Two-Sample Kolmogorov-Smirnov Test ($D_{\text{KS}}$) and Population Stability Index (PSI).
  - Categorical Features: Jensen-Shannon Divergence (JSD) and binned PSI.
- **Empirical Findings:**
  - Temperature: $D_{\text{KS}} = 0.2838$ ($p = 2.70 \times 10^{-124}$), $\text{PSI} = 1.2384$ (Severe Drift).
  - Humidity: $D_{\text{KS}} = 0.1279$ ($p = 2.62 \times 10^{-25}$), $\text{PSI} = 0.2031$ (Moderate Drift).
  - Windspeed: $D_{\text{KS}} = 0.0537$, $\text{PSI} = 0.0241$ (Stable).

---

### Slide 10: Prediction Uncertainty & Error Association
- **Formulation:** Normalized Shannon Entropy $H(x) \in [0, 1]$, Prediction Margin $|2p - 1|$, Expected Calibration Error (ECE).
- **Statistical Proof:**
  - Mean Entropy for Incorrect Inferences: **$0.8370 \pm 0.162$**.
  - Mean Entropy for Correct Inferences: **$0.3532 \pm 0.289$**.
  - **Mann-Whitney U Test:** $U = 1.34 \times 10^5, p = 5.7649 \times 10^{-65}$ (Statistically significant).
  - Calibration ECE: $0.0682$ (strong baseline calibration).
- **Conclusion:** Prediction errors are accompanied by a $+137.0\%$ surge in entropy.

---

### Slide 11: The Secondary Failure-Prediction Meta-Model
- **Meta-Dataset:** 280 experimental evaluation regimes spanning clean and perturbed batches ($N=200$).
- **Ground Truth Failure Label:** Relative F1 degradation $\Delta F1_{\text{rel}} \ge 15\%$ ($105$ failures, $37.5\%$).
- **5-Fold Cross-Validation Performance:**
  - Logistic Regression Meta-Model: Accuracy $86.43\%$, ROC-AUC $0.9215$.
  - Random Forest Meta-Model: Accuracy $87.86\%$, **ROC-AUC $0.9460$**, **PR-AUC $0.9126$**.
  - XGBoost Meta-Model: Accuracy $89.29\%$, **ROC-AUC $0.9565$**, **PR-AUC $0.9458$**.

---

### Slide 12: Feature Group Ablation Study (Proving RQ3)
- **Ablation Comparison Table:**
  - *Quality Only (6 features):* ROC-AUC **$0.4738$** (Collapses to random chance).
  - *Drift Only (7 features):* ROC-AUC **$0.8896$** (Good, but high false alarms).
  - *Uncertainty Only (8 features):* ROC-AUC **$0.8610$** (Good, but misses silent failures).
  - *Composite Multimodal System (21 features):* **ROC-AUC $0.9537$**, **PR-AUC $0.9257$**.
- **Takeaway:** Multimodal integration provides a **$+7.21\%$ boost** over drift-only monitoring and confirms Hypothesis 3.

---

### Slide 13: Out-of-Time Future Holdout Experiment (Step 12)
- **Setup:** 26 continuous weekly operational batches spanning July–December 2012 ($N = 4,376$).
- **Zero-Label Evaluation:** Meta-model inferred failure risk without access to true rental counts.
- **Results:**
  - Early-Warning Classification Accuracy: **$96.15\%$** (25 of 26 weeks correctly categorized).
  - Out-of-Time Holdout ROC-AUC: **$1.0000$**.
  - Successfully alerted operators before performance collapsed, while maintaining low false-alarm rates during stable weeks.

---

### Slide 14: Interactive Streamlit Dashboard Live Demo
- **Demonstration Highlights:**
  - Live slider perturbation (Inject missingness, noise, temperature shift).
  - Real-time recalculation of KS-drift, PSI, and Shannon entropy.
  - Dynamic Alert Banner: 🟢 **NORMAL**, 🟡 **WARNING**, 🔴 **HIGH RISK**.
  - Risk Factor Percentage Breakdown: (e.g. Covariate Drift $42\%$, Uncertainty $35\%$, Missingness $23\%$).
  - Built-in SQLite audit trail and PDF/CSV report exports.

---

### Slide 15: Conclusion, Limitations & Future Work
- **Summary of Contributions:**
  1. Built an operational label-free framework predicting model failure with $0.9537$ ROC-AUC.
  2. Proved empirically that errors produce $+137\%$ higher Shannon entropy ($p = 5.76 \times 10^{-65}$).
  3. Validated out-of-time future holdout with $96.15\%$ early-warning accuracy.
- **Limitations:** Focus on tabular/sensor data; batch-level rather than microsecond instance-level.
- **Future Directions:** Conformal prediction intervals, active learning self-healing loops.
- **Closing:** Thank you. I welcome your questions!
