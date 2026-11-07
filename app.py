"""FrogID: an interactive species classifier for acoustic MFCC measurements."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from src.common import (
    MFCC_COLUMNS,
    MODEL_FILES,
    TARGET,
    canonicalize_columns,
    mfcc_columns,
    score_predictions,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DEFAULT_TEST_DATA = ROOT / "test_data.csv"

st.set_page_config(
    page_title="FrogID | Call-to-Species Lab",
    page_icon="🐸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1420px; }
      .hero { padding: 1.6rem 1.8rem; border: 1px solid #405440; border-radius: 22px;
              background: radial-gradient(circle at 85% 0%, #365331 0, #1b271b 43%, #151d15 100%); }
      .eyebrow { color: #b7f36b; text-transform: uppercase; letter-spacing: .18em;
                 font-size: .75rem; font-weight: 700; }
      .hero h1 { margin: .45rem 0 .35rem; font-size: 2.55rem; color: #f1f5e9; }
      .hero p { color: #c7d4c0; font-size: 1.04rem; max-width: 850px; }
      [data-testid="stMetric"] { background: #1b271b; border: 1px solid #344534;
                                 border-radius: 16px; padding: .9rem 1rem; }
      .note { border-left: 3px solid #b7f36b; padding: .65rem 1rem; color: #dce7d4;
              background: #192319; border-radius: 0 10px 10px 0; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models() -> dict[str, object]:
    missing = [filename for filename in MODEL_FILES.values() if not (MODEL_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "Saved model files are missing. Run `python train_models.py` from the project folder first."
        )
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def read_upload(upload) -> pd.DataFrame:
    frame = canonicalize_columns(pd.read_csv(upload))
    features = mfcc_columns(frame)
    expected = MFCC_COLUMNS
    missing = [column for column in expected + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if len(features) != 22:
        raise ValueError(f"Expected exactly 22 MFCC input columns; found {len(features)}.")
    frame = frame[expected + [TARGET]].copy()
    for column in expected:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[expected + [TARGET]].isna().any().any():
        raise ValueError("The file has blank or non-numeric values in required columns.")
    frame[TARGET] = frame[TARGET].astype(str).str.strip()
    if frame.empty:
        raise ValueError("The uploaded CSV has no rows.")
    if len(frame) > 10_000:
        raise ValueError("Upload test data only; this demo accepts up to 10,000 rows.")
    return frame


def evaluate_all(frame: pd.DataFrame, models: dict[str, object]) -> tuple[pd.DataFrame, dict, dict]:
    features = mfcc_columns(frame)
    x_test = frame[features]
    y_true = frame[TARGET]
    expected_labels = set(map(str, next(iter(models.values())).classes_))
    unknown_labels = set(y_true.astype(str)) - expected_labels
    if unknown_labels:
        raise ValueError(f"Unknown Species label(s): {', '.join(sorted(unknown_labels))}")
    table: dict[str, dict[str, float | None]] = {}
    details: dict[str, dict] = {}
    for name, estimator in models.items():
        predictions = estimator.predict(x_test)
        probabilities = estimator.predict_proba(x_test)
        classes = list(estimator.classes_)
        table[name] = score_predictions(y_true, predictions, probabilities, classes)
        details[name] = {
            "predictions": predictions,
            "classes": classes,
            "report": classification_report(
                y_true,
                predictions,
                labels=classes,
                target_names=classes,
                zero_division=0,
                output_dict=True,
            ),
            "matrix": confusion_matrix(y_true, predictions, labels=classes),
        }
    return pd.DataFrame.from_dict(table, orient="index"), details, {"rows": len(frame)}


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AIML · Classification studio</div>
      <h1>FrogID: from call to species</h1>
      <p>Can 22 acoustic fingerprints tell one frog species from another? Compare five
      classic classifiers on syllable-level MFCC measurements from real field recordings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
try:
    models = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.markdown("### 🐸 Field kit")
    st.caption("Evaluation data must include all 22 MFCC features and the `Species` target.")
    uploaded = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
    model_name = st.selectbox("Inspect a model", list(MODEL_FILES))
    st.markdown("---")
    st.caption("The app loads saved estimators. It never retrains on uploaded test data.")

if uploaded is None:
    if not DEFAULT_TEST_DATA.exists():
        st.info("Upload a labeled test CSV to begin.")
        st.stop()
    frame = pd.read_csv(DEFAULT_TEST_DATA)
    st.info("Showing the bundled held-out test set. Upload another labeled test CSV to compare it.")
else:
    try:
        frame = read_upload(uploaded)
    except Exception as exc:
        st.error(f"Could not use this file: {exc}")
        st.stop()

if not frame.empty:
    features = mfcc_columns(frame)
    classes = sorted(frame[TARGET].astype(str).unique())
    st.write("")
    left, middle, right = st.columns(3)
    left.metric("Test syllables", f"{len(frame):,}")
    middle.metric("Acoustic features", f"{len(features)} MFCCs")
    right.metric("Species represented", f"{len(classes)} / 10")

    try:
        results, details, _ = evaluate_all(frame, models)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        st.stop()

    st.markdown("### Model face-off")
    st.caption("Macro averages give each species equal weight; AUC is macro one-vs-rest.")
    display = results.copy()
    for column in display.columns:
        display[column] = display[column].map(format_metric)
    st.dataframe(display, use_container_width=True)

    chosen = details[model_name]
    selected_metrics = results.loc[model_name]
    st.markdown(f"### {model_name}")
    metric_columns = st.columns(6)
    for col, metric in zip(metric_columns, selected_metrics.index):
        col.metric(metric, format_metric(selected_metrics[metric]))

    chart_col, report_col = st.columns([1.25, 1])
    with chart_col:
        st.markdown("#### Confusion map")
        matrix = chosen["matrix"]
        chart = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=chosen["classes"],
                y=chosen["classes"],
                colorscale=[[0, "#172217"], [0.5, "#557746"], [1, "#b7f36b"]],
                hoverongaps=False,
                colorbar={"title": "Syllables"},
            )
        )
        chart.update_layout(
            xaxis_title="Predicted species",
            yaxis_title="Actual species",
            height=550,
            margin=dict(l=5, r=5, t=10, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e9f1e2",
        )
        chart.update_xaxes(tickangle=-35)
        st.plotly_chart(chart, use_container_width=True)
    with report_col:
        st.markdown("#### Species report")
        report = pd.DataFrame(chosen["report"]).T
        st.dataframe(report.style.format(precision=3), use_container_width=True, height=550)

    if len(classes) < 10:
        st.warning(
            "This upload omits one or more species. AUC is marked N/A unless every species has "
            "both positive and negative examples in the uploaded test set."
        )

    with st.expander("About these acoustic features"):
        st.write(
            "MFCCs (mel-frequency cepstral coefficients) summarize the spectral shape of each "
            "frog-call syllable. The 22 normalized coefficients are the model inputs; the species "
            "label is used only to score predictions. Recording IDs and the related family/genus "
            "labels are not model inputs."
        )
