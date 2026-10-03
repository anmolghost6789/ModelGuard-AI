# app.py
"""
Interactive Streamlit Dashboard for:
"An AI-Based Framework for Predicting Machine Learning Model Failure
Using Data Quality, Distribution Shift, and Prediction Uncertainty"
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data_loader import (
    load_raw_dataset,
    create_chronological_splits,
    FEATURE_COLS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS
)
from src.baseline_models import (
    get_baseline_classifiers,
    evaluate_classifier
)
from src.stress_engine import StressTestEngine
from src.drift_detector import DistributionShiftDetector
from src.uncertainty import (
    compute_instance_uncertainty,
    compute_expected_calibration_error,
    compute_batch_uncertainty_summary
)
from src.failure_model import ALL_META_FEATURES

# Page Configuration
st.set_page_config(
    page_title="AI Model Failure Prediction Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark-friendly sleek card aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .banner-normal {
        background: linear-gradient(135deg, rgba(46, 160, 67, 0.25), rgba(46, 160, 67, 0.05));
        border: 1px solid #2ea043;
        border-radius: 12px;
        padding: 20px;
        color: #3fb950;
        font-weight: 600;
        font-size: 22px;
        text-align: center;
        margin-bottom: 20px;
    }
    .banner-warning {
        background: linear-gradient(135deg, rgba(210, 153, 34, 0.25), rgba(210, 153, 34, 0.05));
        border: 1px solid #d29922;
        border-radius: 12px;
        padding: 20px;
        color: #e3b341;
        font-weight: 600;
        font-size: 22px;
        text-align: center;
        margin-bottom: 20px;
    }
    .banner-danger {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.25), rgba(248, 81, 73, 0.05));
        border: 1px solid #f85149;
        border-radius: 12px;
        padding: 20px;
        color: #f85149;
        font-weight: 600;
        font-size: 22px;
        text-align: center;
        margin-bottom: 20px;
    }
    .disclaimer-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #6e7681;
        padding: 12px 16px;
        font-size: 12px;
        color: #8b949e;
        border-radius: 0 8px 8px 0;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data_and_splits():
    df = load_raw_dataset("hour.csv")
    splits = create_chronological_splits(df)
    return df, splits


@st.cache_resource
def load_models_and_artifacts():
    primary_model = None
    meta_model = None
    scaler = None
    meta_scaler = None

    if os.path.exists("output/models/primary_rf_model.joblib"):
        primary_model = joblib.load("output/models/primary_rf_model.joblib")
    if os.path.exists("output/models/failure_rf_meta_model.joblib"):
        meta_model = joblib.load("output/models/failure_rf_meta_model.joblib")
    if os.path.exists("output/models/scaler.joblib"):
        scaler = joblib.load("output/models/scaler.joblib")
    if os.path.exists("output/models/meta_scaler.joblib"):
        meta_scaler = joblib.load("output/models/meta_scaler.joblib")

    return primary_model, meta_model, scaler, meta_scaler


def main():
    st.title("🛡️ AI-Based Framework for Predicting Machine Learning Model Failure")
    st.markdown("**Early Detection of Model Degradation via Data Quality, Distribution Shift, and Prediction Uncertainty**")
    st.caption("College Research Project • Production Verification & Monitoring Interface")

    # Load resources
    raw_df, splits = get_data_and_splits()
    primary_model, meta_model, scaler, meta_scaler = load_models_and_artifacts()

    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]
    fut_data = splits["future"]
    target_thresh = splits["meta"]["threshold"]

    # Sidebar Controls
    st.sidebar.header("⚙️ Experimental Controls")

    dataset_source = st.sidebar.selectbox(
        "Dataset Source",
        ["Capital Bikeshare (hour.csv - 17,379 rows)", "Upload Custom CSV"]
    )
    if dataset_source == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            st.sidebar.success("Custom CSV uploaded.")

    target_definition = st.sidebar.selectbox(
        "Target Variable Task",
        ["High-Demand Surge (Binary Classification: cnt ≥ 109)", "Continuous Demand (Regression: cnt)"]
    )

    primary_model_choice = st.sidebar.selectbox(
        "Primary ML Architecture",
        ["Random Forest (Trained 2011)", "Gradient Boosting", "Logistic Regression", "XGBoost"]
    )

    failure_threshold_pct = st.sidebar.slider(
        "Failure Definition Threshold (Relative F1 Drop %)",
        min_value=5,
        max_value=30,
        value=15,
        step=1,
        help="A batch is classified as a Failure if its F1-score degrades by more than this percentage relative to the clean baseline."
    )

    # Main Tabs
    tabs = st.tabs([
        "📊 Dataset & Splits",
        "⚡ Baseline Benchmark",
        "🧪 Stress Testing Lab",
        "📈 Drift & Uncertainty",
        "🚨 Failure Early Warning",
        "🔮 Future Holdout Tracker",
        "📄 Research Findings & RQs"
    ])

    # -------------------------------------------------------------
    # TAB 1: DATASET & SPLITS
    # -------------------------------------------------------------
    with tabs[0]:
        st.subheader("Dataset Architecture & Strict Chronological Partitions")
        st.markdown(
            "To prevent temporal leakage and evaluate authentic out-of-time generalizability, "
            "the data is partitioned strictly chronologically across 2011–2012:"
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Historical Train</div>
                <div class="metric-value" style="color: #58a6ff;">{len(train_data['X']):,}</div>
                <div style="font-size:12px; color:#8b949e;">Year 2011 (yr=0)</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Validation Set</div>
                <div class="metric-value" style="color: #bc8cff;">{len(val_data['X']):,}</div>
                <div style="font-size:12px; color:#8b949e;">Q1 2012</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Clean Test Set</div>
                <div class="metric-value" style="color: #3fb950;">{len(test_data['X']):,}</div>
                <div style="font-size:12px; color:#8b949e;">Q2 2012 (Baseline)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Future Holdout</div>
                <div class="metric-value" style="color: #f0883e;">{len(fut_data['X']):,}</div>
                <div style="font-size:12px; color:#8b949e;">H2 2012 (Out-of-Time)</div>
            </div>
            """, unsafe_allow_html=True)

        if os.path.exists("output/figures/fig1_dataset_distribution.png"):
            st.image("output/figures/fig1_dataset_distribution.png", caption="Figure 1: Demand Distribution and Seasonal Dynamics", use_container_width=True)

        st.markdown("### Feature Matrix & Leakage Exclusions")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Included Predictive Features (11):**")
            st.json(FEATURE_COLS)
        with c2:
            st.write("**Permanently Excluded Columns (Strict Leakage Prevention):**")
            st.markdown("""
            - `instant`: Sequential row counter (would directly leak timeline).
            - `casual` & `registered`: Sub-components whose sum strictly equals `cnt` ($r=0.972$).
            - `yr`: Binary year indicator (causes artificial overfit between 2011 and 2012).
            """)

    # -------------------------------------------------------------
    # TAB 2: BASELINE BENCHMARK
    # -------------------------------------------------------------
    with tabs[1]:
        st.subheader("Baseline Primary Model Performance (Clean Unseen Test Data)")
        st.markdown("Performance established on normal, uncorrupted evaluation data from Q2 2012:")

        if os.path.exists("output/tables/step4_baseline_models.csv"):
            df_base = pd.read_csv("output/tables/step4_baseline_models.csv")
            st.dataframe(df_base.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
                "Brier Score": "{:.4f}"
            }), use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            if os.path.exists("output/figures/fig3_baseline_performance.png"):
                st.image("output/figures/fig3_baseline_performance.png", caption="Figure 3: Primary Classifier Performance Comparison", use_container_width=True)
        with colB:
            if os.path.exists("output/figures/fig10_confusion_matrices.png"):
                st.image("output/figures/fig10_confusion_matrices.png", caption="Figure 10: Primary & Failure Meta-Model Confusion Matrices", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: STRESS TESTING LAB
    # -------------------------------------------------------------
    with tabs[2]:
        st.subheader("Controlled Evaluation Stress & Degradation Laboratory")
        st.markdown("Perturb evaluation conditions in real-time to observe primary model degradation:")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            missing_rate = st.slider("Missingness Rate (%)", 0, 40, 15, step=5) / 100.0
            noise_std = st.slider("Gaussian Sensor Noise (std)", 0.0, 0.6, 0.2, step=0.05)
        with col_s2:
            outlier_rate = st.slider("Sensor Spike Outliers (%)", 0, 25, 10, step=5) / 100.0
            temp_shift = st.slider("Covariate Temperature Shift (Δ temp)", -0.4, 0.4, 0.2, step=0.05)
        with col_s3:
            weather_inflation = st.slider("Severe Weather Inflation (%)", 0, 50, 20, step=5) / 100.0
            imbalance_pos = st.slider("Class Imbalance Positive Ratio", 0.1, 0.9, 0.5, step=0.05)

        # Apply user perturbations to test set
        stress_eng = StressTestEngine(random_state=42)
        X_test_clean = test_data["X"].copy()
        y_test_clean = test_data["y"].copy()
        train_meds = train_data["X"].median()

        # Step-by-step stress
        X_p = X_test_clean.copy()
        if temp_shift != 0:
            X_p, _ = stress_eng.apply_covariate_shift(X_p, shifts={"temp": temp_shift, "hum": -temp_shift})
        if noise_std > 0:
            X_p, _ = stress_eng.apply_gaussian_noise(X_p, noise_std=noise_std)
        if missing_rate > 0:
            X_p, _ = stress_eng.apply_missingness(X_p, missing_rate=missing_rate, train_medians=train_meds)
        if outlier_rate > 0:
            X_p, _ = stress_eng.apply_outliers(X_p, outlier_rate=outlier_rate, magnitude=4.0)
        if weather_inflation > 0:
            X_p, _ = stress_eng.apply_categorical_shift(X_p, col="weathersit", target_value=3, inflation_rate=weather_inflation)

        # Resample for class imbalance if not 0.5
        y_p = y_test_clean
        if abs(imbalance_pos - 0.5) > 0.05:
            X_p, y_p, _ = stress_eng.apply_class_imbalance_shift(X_p, y_p, pos_ratio=imbalance_pos)

        # Re-evaluate primary model
        if primary_model is not None and scaler is not None:
            X_p_sc = pd.DataFrame(scaler.transform(X_p), columns=FEATURE_COLS)
            eval_stressed = evaluate_classifier(primary_model, X_p_sc, y_p)
            clean_f1 = 0.9436
            stressed_f1 = eval_stressed["f1"]
            rel_f1_drop = max(0.0, (clean_f1 - stressed_f1) / clean_f1) * 100

            st.markdown("#### Live Stress Evaluation Realized")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Baseline F1", f"{clean_f1:.4f}")
            m2.metric("Stressed Batch F1", f"{stressed_f1:.4f}", delta=f"-{clean_f1 - stressed_f1:.4f}")
            m3.metric("Relative F1 Degradation", f"{rel_f1_drop:.1f}%")
            m4.metric("Model Failure Status", "FAILURE DETECTED" if rel_f1_drop >= failure_threshold_pct else "STABLE", delta_color="inverse")

        if os.path.exists("output/figures/fig8_performance_degradation_stress.png"):
            st.image("output/figures/fig8_performance_degradation_stress.png", caption="Figure 8: Performance Degradation Across Standard Experimental Stress Conditions", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: DRIFT & UNCERTAINTY ENGINE
    # -------------------------------------------------------------
    with tabs[3]:
        st.subheader("Statistical Distribution Drift & Prediction Uncertainty Diagnostics")
        
        st.markdown("### Feature-Level Distribution Drift")
        if os.path.exists("output/tables/step7_drift_metrics.csv"):
            df_drift = pd.read_csv("output/tables/step7_drift_metrics.csv")
            st.dataframe(df_drift.style.format({
                "KS Statistic": "{:.4f}",
                "KS p-value": "{:.2e}",
                "JS Divergence": "{:.4f}",
                "PSI": "{:.4f}"
            }), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("output/figures/fig5_drift_scores_psi_ks.png"):
                st.image("output/figures/fig5_drift_scores_psi_ks.png", caption="Figure 5: Feature Drift Metrics (PSI & KS Statistics)", use_container_width=True)
        with c2:
            if os.path.exists("output/figures/fig4_feature_distribution_shift.png"):
                st.image("output/figures/fig4_feature_distribution_shift.png", caption="Figure 4: Temperature & Humidity KDE Distribution Shift", use_container_width=True)

        st.markdown("### Prediction Uncertainty & Calibration")
        c3, c4 = st.columns(2)
        with c3:
            if os.path.exists("output/figures/fig6_uncertainty_distributions.png"):
                st.image("output/figures/fig6_uncertainty_distributions.png", caption="Figure 6: Prediction Confidence and Shannon Entropy Distributions", use_container_width=True)
        with c4:
            if os.path.exists("output/figures/fig7_uncertainty_vs_prediction_error.png"):
                st.image("output/figures/fig7_uncertainty_vs_prediction_error.png", caption="Figure 7: Shannon Entropy vs. Errors & Reliability Curve", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: FAILURE EARLY WARNING
    # -------------------------------------------------------------
    with tabs[4]:
        st.subheader("Secondary AI Meta-Model: Predicting Model Failure Probability")
        st.markdown(
            "This model takes operational telemetry (**Data Quality**, **Distribution Drift**, and **Prediction Uncertainty**) "
            "and predicts whether the primary model is currently experiencing a performance failure **WITHOUT requiring ground-truth labels**."
        )

        # Run meta-model on current stressed batch
        if primary_model is not None and meta_model is not None and scaler is not None and meta_scaler is not None:
            # 1. Drift
            drift_det = DistributionShiftDetector(reference_df=train_data["X"])
            b_drift = drift_det.compute_overall_drift(X_p)

            # 2. Uncertainty
            X_p_sc = pd.DataFrame(scaler.transform(X_p), columns=FEATURE_COLS)
            e_res = evaluate_classifier(primary_model, X_p_sc, y_p)
            b_unc = compute_batch_uncertainty_summary(e_res["y_prob"], model=primary_model, X=X_p_sc)

            # 3. Meta-vector
            dq_score = max(0.0, 1.0 - (missing_rate * 1.5 + noise_std * 1.2 + outlier_rate * 2.0))
            meta_vec = {
                "missing_pct": float(missing_rate),
                "noise_std": float(noise_std),
                "outlier_pct": float(outlier_rate),
                "cov_shift_mag": float(abs(temp_shift)),
                "corrupt_feature_cnt": float(int(missing_rate > 0) + int(noise_std > 0) + int(outlier_rate > 0)),
                "data_quality_score": float(dq_score),
                "mean_ks_stat": float(b_drift["mean_ks_stat"]),
                "max_ks_stat": float(b_drift["max_ks_stat"]),
                "mean_psi": float(b_drift["mean_psi"]),
                "max_psi": float(b_drift["max_psi"]),
                "mean_js_divergence": float(b_drift["mean_js_divergence"]),
                "pct_features_shifted": float(b_drift["pct_features_shifted"]),
                "num_features_shifted": float(b_drift["num_features_shifted"]),
                "mean_confidence": float(b_unc["mean_confidence"]),
                "min_confidence": float(b_unc["min_confidence"]),
                "mean_entropy": float(b_unc["mean_entropy"]),
                "max_entropy": float(b_unc["max_entropy"]),
                "pct_high_entropy": float(b_unc["pct_high_entropy"]),
                "pct_low_confidence": float(b_unc["pct_low_confidence"]),
                "mean_margin": float(b_unc["mean_margin"]),
                "mean_ensemble_variance": float(b_unc["mean_ensemble_variance"])
            }
            df_m = pd.DataFrame([meta_vec])[ALL_META_FEATURES]
            df_m_sc = meta_scaler.transform(df_m)
            fail_prob = float(meta_model.predict_proba(df_m_sc)[0, 1])

            # Alert Banner
            if fail_prob < 0.35:
                st.markdown(f"""
                <div class="banner-normal">
                    🟢 NORMAL OPERATION — FAILURE RISK: {fail_prob*100:.1f}%<br>
                    <span style="font-size:14px; font-weight:400;">Primary ML model operating within safe historical tolerances. No significant degradation detected.</span>
                </div>
                """, unsafe_allow_html=True)
            elif fail_prob < 0.65:
                st.markdown(f"""
                <div class="banner-warning">
                    🟡 WARNING STATUS — FAILURE RISK: {fail_prob*100:.1f}%<br>
                    <span style="font-size:14px; font-weight:400;">Elevated drift and prediction entropy detected. Model performance is moderately degrading. Continuous human-in-the-loop review recommended.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="banner-danger">
                    🔴 HIGH RISK ALERT — FAILURE RISK: {fail_prob*100:.1f}%<br>
                    <span style="font-size:14px; font-weight:400;">Severe performance collapse predicted (&ge;15% drop). Primary model outputs unreliable. Fallback heuristic or model retraining required immediately.</span>
                </div>
                """, unsafe_allow_html=True)

        colM1, colM2 = st.columns(2)
        with colM1:
            if os.path.exists("output/figures/fig11_roc_pr_curves.png"):
                st.image("output/figures/fig11_roc_pr_curves.png", caption="Figure 11: Failure Meta-Model ROC and PR Curves", use_container_width=True)
        with colM2:
            if os.path.exists("output/figures/fig12_meta_feature_importance.png"):
                st.image("output/figures/fig12_meta_feature_importance.png", caption="Figure 12: Top Early-Warning Indicators of Model Failure", use_container_width=True)

        st.markdown("""
        <div class="disclaimer-box">
            <strong>Operational Disclaimer:</strong> This AI-based model failure risk score is an experimental monitoring metric 
            designed to detect statistical distribution degradation and predictive uncertainty in machine learning pipelines. 
            It does NOT constitute a medical, financial, legal, or absolute safety guarantee.
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 6: FUTURE HOLDOUT TRACKER
    # -------------------------------------------------------------
    with tabs[5]:
        st.subheader("Out-of-Time Future Holdout Experiment (July – December 2012)")
        st.markdown(
            "Evaluating the entire failure-prediction framework on 26 continuous weekly operational batches "
            "from the unseen second half of 2012:"
        )

        if os.path.exists("output/tables/step12_future_holdout.csv"):
            df_fut_track = pd.read_csv("output/tables/step12_future_holdout.csv")
            st.dataframe(df_fut_track.style.format({
                "Mean KS Drift": "{:.4f}",
                "Mean PSI": "{:.4f}",
                "Mean Entropy": "{:.4f}",
                "Predicted Failure Prob": "{:.2%}",
                "Actual F1-Score": "{:.4f}",
                "Relative F1 Drop %": "{:.1f}%"
            }), use_container_width=True)

            # Timeline plot of predicted risk vs actual F1
            fig, ax1 = plt.subplots(figsize=(11, 4))
            ax2 = ax1.twinx()
            ax1.plot(df_fut_track["Window"], df_fut_track["Predicted Failure Prob"]*100, "r-o", label="Predicted Failure Probability (%)", linewidth=2)
            ax2.plot(df_fut_track["Window"], df_fut_track["Actual F1-Score"], "b-s", label="Realized F1-Score", linewidth=2)
            ax1.axhline(50, color="gray", linestyle="--", label="Decision Threshold (50%)")
            ax1.set_xlabel("Weekly Operational Window (H2 2012)")
            ax1.set_ylabel("Predicted Failure Risk (%)", color="red")
            ax2.set_ylabel("Actual F1-Score", color="blue")
            ax1.set_title("Out-of-Time Trajectory: Predicted Failure Probability vs. Actual F1-Score")
            plt.tight_layout()
            st.pyplot(fig)

    # -------------------------------------------------------------
    # TAB 7: RESEARCH PAPER FINDINGS
    # -------------------------------------------------------------
    with tabs[6]:
        st.subheader("Research Questions & Empirical Conclusions")
        if os.path.exists("output/tables/step14_research_questions.json"):
            with open("output/tables/step14_research_questions.json") as f:
                rq_data = json.load(f)

            for rq_id, data in rq_data.items():
                with st.expander(f"**{rq_id}: {data['Question']}** — [{('SUPPORTED' if data['Supported'] else 'NOT SUPPORTED')}]", expanded=True):
                    st.write(f"**Empirical Finding:** {data['Empirical Evidence']}")

        st.markdown("### Master Regime Comparison Table")
        if os.path.exists("output/tables/step13_master_comparison.csv"):
            df_mst = pd.read_csv("output/tables/step13_master_comparison.csv")
            st.table(df_mst)

        st.markdown("### Feature Ablation Results (RQ3)")
        if os.path.exists("output/tables/step11_ablation_rq3.csv"):
            df_abl = pd.read_csv("output/tables/step11_ablation_rq3.csv")
            st.dataframe(df_abl.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}"
            }), use_container_width=True)


if __name__ == "__main__":
    main()
