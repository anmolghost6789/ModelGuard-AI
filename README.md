# 🛡️ ModelGuard-AI: AI Model Failure Prediction Framework

> **An AI-Based Framework for Predicting Machine Learning Model Failure Using Data Quality, Distribution Shift, and Prediction Uncertainty**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4%2B-eb5424.svg)](https://xgboost.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/Tests-22%2F22%20Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Overview

Traditional machine learning evaluation focuses on static benchmark metrics (Accuracy, F1-Score) on fixed test sets. However, when deployed in non-stationary operational environments, models encounter sensor degradation, covariate drift, and anomalous conditions that lead to **silent performance failure** without throwing software exceptions. Furthermore, real-world ground-truth labels often suffer from feedback latency of weeks or months.

This repository provides an autonomous, **label-free early-warning framework** that predicts the probability of primary ML model failure *before* or *during* deployment by fusing three orthogonal operational observability vectors:
1. **Data Quality Telemetry:** Missingness rates, sensor noise variance, and outlier density.
2. **Distribution Shift Detection:** Two-sample Kolmogorov-Smirnov test statistics, Population Stability Index (PSI), and Jensen-Shannon divergence.
3. **Prediction Uncertainty:** Normalized Shannon entropy, prediction margins, confidence calibration (ECE), and ensemble disagreement.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   END-TO-END DECOUPLED ARCHITECTURE                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  [Operational Data Ingestion]
               │
               ▼
  [Data Quality & Schema Profiler] ──────> Compute Completeness, Uniqueness, IQR Outliers, DQS (0-100)
               │
               ▼
  [Primary Model Inference]        ──────> Compute Confidence, Normalized Shannon Entropy H(x), Margin
               │
               ▼
  [Distribution Drift Engine]      ──────> Compute KS Statistics, PSI (10 bins), JS Divergence
               │
               ▼
  [21-D Telemetry Vector Extraction]
               │
               ▼
  [Secondary Failure Meta-Classifier] ────> Predict P(Model Failure | Telemetry)
               │
               ▼
  [3-Tier Operational Triage]
  ├── 🟢 NORMAL     (Risk < 35%)  — Model operating within verified historical tolerances
  ├── 🟡 WARNING    (Risk 35-65%) — Elevated drift/ambiguity; human-in-the-loop review recommended
  └── 🔴 HIGH RISK  (Risk ≥ 65%)  — Critical failure predicted (ΔF1 ≥ 15%); fallback required
```

---

## 📊 Key Empirical Research Findings

Evaluated on the **Capital Bikeshare empirical log ($N = 17,379$)** across 2 full years (2011–2012) using strict non-leaking chronological splits:

- **Baseline Performance (Clean Unseen Test Data, Q2 2012):**
  - Primary Random Forest Classifier: **$94.36\%$ F1-Score**, **$92.48\%$ Accuracy**, **$0.9831$ ROC-AUC**, **$0.0558$ Brier Score**.
- **Stress-Induced Catastrophic Collapse:**
  - Directional Covariate Shift (Cold Snap $\Delta t = -0.25$): F1 plunged from $94.36\%$ to **$72.53\%$** (a **$23.14\%$ relative collapse**).
  - Gaussian sensor noise caused progressive degradation up to $8.89\%$.
- **Statistical Significance of Uncertainty on Errors:**
  - Prediction errors exhibit mean Shannon entropy of **$0.8370$** vs. **$0.3532$** for correct predictions ($+137.0\%$ higher uncertainty).
  - **Mann-Whitney $U = 1.34 \times 10^5, p = 5.7649 \times 10^{-65}$** (Statistically significant).
- **Multimodal Synergy Confirmed (Feature Group Ablation):**
  - *Quality Only:* ROC-AUC **$0.4738$** (Collapses to random chance).
  - *Drift Only:* ROC-AUC **$0.8896$** (Good, but triggers false alarms on benign seasonal drift).
  - *Uncertainty Only:* ROC-AUC **$0.8610$** (Misses confident silent failures).
  - *Proposed Composite Multimodal System:* **ROC-AUC $0.9537$**, **PR-AUC $0.9257$** ($+7.21\%$ over drift-only monitoring).
- **Out-of-Time Future Holdout Generalization (H2 2012):**
  - Tested across 26 continuous weekly operational batches in late 2012 without ground-truth labels.
  - Early-warning classification accuracy: **$96.15\%$** (25/26 weeks correctly categorized).
  - Holdout ROC-AUC: **$1.0000$**.

---

## 🗂️ Repository Directory Structure

```text
├── data/                               # Dataset Storage
├── src/                                # Core Modular Framework
│   ├── data_profiler.py                # [MODULE 1] Schema auto-detection & outlier profiling
│   ├── data_quality.py                 # [MODULE 2] Data quality scoring & missingness metrics
│   ├── model_trainer.py                # [MODULE 3] Primary ML training (Classif. & Regress.)
│   ├── baseline_evaluator.py           # [MODULE 4] Clean baseline metrics & confusion matrices
│   ├── drift_detector.py               # [MODULE 5] KS-test, PSI, and JS-divergence engine
│   ├── uncertainty.py                  # [MODULE 6] Shannon entropy, confidence, margin, ECE
│   ├── stress_simulator.py             # [MODULE 7] Controlled perturbation laboratory (A–G)
│   ├── failure_predictor.py            # [MODULE 8] Secondary failure meta-dataset & risk scoring
│   ├── failure_explainer.py            # [MODULE 9 & 10] Risk factor decomposition & XAI
│   └── experiment_manager.py           # [MODULE 11] SQLite experiment tracking & persistence
├── app.py                              # [MODULE 12] Master Streamlit Interactive Dashboard
├── run_experiments.py                  # End-to-End Batch Research Execution Script
├── storage/                            # Persistent Application Database (SQLite)
├── output/
│   ├── figures/                        # 12 Publication-grade figures (300 DPI)
│   ├── tables/                         # Experimental result tables (CSV & JSON)
│   └── models/                         # Serialized primary and meta-models (.joblib)
├── tests/                              # Unit test suite (22 tests passing)
├── docs/                               # Academic & Viva Deliverables
│   ├── PROJECT_REPORT.md               # Complete Senior Capstone Thesis Report
│   ├── VIVA_QA_GUIDE.md                # 50 Comprehensive Viva Questions & Model Answers
│   └── PRESENTATION_DECK.md            # 15 Slide-by-slide defense deck with spoken script
├── RESEARCH_PAPER.md                   # 15-Section Academic Conference Manuscript
└── requirements.txt                    # Python dependencies
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/anmolghost6789/ModelGuard-AI.git
cd ModelGuard-AI
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# or: .\venv\Scripts\activate  # On Windows

pip install -r requirements.txt
```

### 3. Run the Unit Test Suite (Verify 22/22 Passing)
```bash
python -m unittest discover -s tests
```
*Expected Output:* `Ran 22 tests in 0.34s - OK`

### 4. Launch the Interactive Streamlit Web Application
```bash
python -m streamlit run app.py
# Or on Windows, simply double-click or run:
.\run_app.bat
```
*Open `http://localhost:8501` in your browser.*

### 5. Re-run Full Experiments & Regenerate 12 Figures
```bash
python run_experiments.py
```

---

## 📑 Academic Deliverables

- **Research Paper Manuscript:** [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md) (15 sections formatted for IEEE/ACM conference submission).
- **Final Capstone Thesis Report:** [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) (Complete B.Tech/B.E. project documentation).
- **Viva Voce Defense Guide:** [`docs/VIVA_QA_GUIDE.md`](docs/VIVA_QA_GUIDE.md) (50 technical questions and model defense answers).
- **Defense Slide Deck Script:** [`docs/PRESENTATION_DECK.md`](docs/PRESENTATION_DECK.md) (15 presentation slides with spoken script).

---

## 📖 Citation (BibTeX)

If you use this framework or experimental setup in your research, please cite:

```bibtex
@article{modelguard_ai_2026,
  title={An AI-Based Framework for Predicting Machine Learning Model Failure Using Data Quality, Distribution Shift, and Prediction Uncertainty},
  author={Senior Capstone Research Team},
  journal={Department of Computer Science and Engineering},
  year={2026},
  url={https://github.com/anmolghost6789/ModelGuard-AI}
}
```

---

## ⚖️ License & Disclaimer

- **License:** Distributed under the [MIT License](LICENSE).
- **Operational Disclaimer:** This framework is an operational decision support and monitoring tool designed to detect statistical distribution degradation and predictive uncertainty in machine learning pipelines. It does not constitute a medical, financial, legal, or absolute safety guarantee.
