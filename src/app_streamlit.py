# src/app_streamlit.py
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import shap
import streamlit as st

from config import PROCESSED_DIR, MODEL_DIR

st.set_page_config(
    page_title="NBA Shot Selection XAI",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    """Load processed shot-level data."""
    data_path = PROCESSED_DIR / "shots_for_model.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Please run extract_shots.py first."
        )
    df = pd.read_parquet(data_path)
    return df


@st.cache_resource
def load_model_and_explainer():
    """Load LightGBM model and SHAP TreeExplainer."""
    model_path = MODEL_DIR / "lgbm_shot_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Please run train_model.py first."
        )
    obj = joblib.load(model_path)
    model = obj["model"]
    feature_cols = obj["features"]

    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_cols


def plot_shap_bar(shap_values, feature_names, title):
    """Create horizontal bar chart for SHAP values."""
    shap_values = np.array(shap_values, dtype=float)
    feature_names = list(feature_names)

    order = np.argsort(shap_values)
    vals = shap_values[order]
    names = [feature_names[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
        )
    )
    fig.update_layout(
        title=title,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="SHAP value (impact on log-odds)",
        template="plotly_white",
    )
    return fig


def get_class_shap(shap_output):
    """
    Handle different SHAP return formats for binary LightGBM:
    - list of arrays: pick class 1 if exists, else class 0
    - single ndarray: use it directly
    Returns 1D array for a single sample.
    """
    if isinstance(shap_output, list):
        if len(shap_output) == 1:
            arr = shap_output[0]
        else:
            # class 1 usually corresponds to positive class
            arr = shap_output[1]
    else:
        arr = shap_output
    arr = np.array(arr)
    # arr shape should be (n_samples, n_features); we need first sample
    if arr.ndim == 2:
        return arr[0]
    return arr


def main():
    df = load_data()
    model, explainer, feature_cols = load_model_and_explainer()

    st.title("Explainable Shot-Selection in the NBA")

    st.markdown(
        "This app predicts whether a shot is a high-value attempt and explains the decision "
        "using contextual features and SHAP values."
    )

    # ----- Sidebar: shot selection -----
    st.sidebar.header("Shot selection")
    total_shots = len(df)
    st.sidebar.write(f"Total shots in dataset: {total_shots}")

    # Use session_state to avoid random jumping when the app reruns
    if "shot_index" not in st.session_state:
        st.session_state.shot_index = random.randint(0, total_shots - 1)

    shot_index = st.sidebar.number_input(
        "Shot row index",
        min_value=0,
        max_value=total_shots - 1,
        value=st.session_state.shot_index,
        step=1,
        key="shot_index",
    )

    shot = df.iloc[[shot_index]].copy()  # DataFrame with single row

    meta_cols = [
        "game_id",
        "eventnum",
        "player_name",
        "pctimestring",
        "score",
        "scoremargin",
    ]
    meta = {c: shot[c].iloc[0] if c in shot.columns else None for c in meta_cols}

    # ----- Sidebar: what-if controls -----
    st.sidebar.header("What-if scenario")

    def safe_get(col, default):
        """Get a numeric value from the shot row with a safe default."""
        if col not in shot.columns:
            return default
        val = shot[col].iloc[0]
        if pd.isna(val):
            return default
        try:
            return float(val)
        except Exception:
            return default

    sec_val = safe_get("seconds_remaining", 600.0)
    margin_val = safe_get("score_margin_num", 0.0)
    home_val = safe_get("is_home_offense", 0.5)
    three_val = safe_get("is_three", 0.0)

    new_seconds = st.sidebar.slider(
        "Seconds remaining in period",
        min_value=0.0,
        max_value=720.0,
        value=float(sec_val),
        step=5.0,
        key="seconds_remaining_slider",
    )
    new_margin = st.sidebar.slider(
        "Score margin (home - away)",
        min_value=-30,
        max_value=30,
        value=int(margin_val),
        step=1,
        key="score_margin_slider",
    )
    new_is_home = st.sidebar.slider(
        "Offense is home team (0 = visitor, 1 = home)",
        min_value=0.0,
        max_value=1.0,
        value=float(home_val),
        step=0.1,
        key="is_home_slider",
    )
    new_is_three = st.sidebar.selectbox(
        "Is three-point attempt?",
        options=[0, 1],
        index=int(three_val) if three_val in [0, 1] else 0,
        key="is_three_select",
    )

    # ----- Build original and what-if feature frames -----
    X_orig = shot[feature_cols].astype(float).copy()
    X_cf = X_orig.copy()

    if "seconds_remaining" in feature_cols:
        X_cf.loc[:, "seconds_remaining"] = new_seconds
    if "score_margin_num" in feature_cols:
        X_cf.loc[:, "score_margin_num"] = new_margin
    if "is_home_offense" in feature_cols:
        X_cf.loc[:, "is_home_offense"] = new_is_home
    if "is_three" in feature_cols:
        X_cf.loc[:, "is_three"] = float(new_is_three)

    # ----- Predictions -----
    proba_orig = float(model.predict_proba(X_orig.values)[:, 1][0])
    proba_cf = float(model.predict_proba(X_cf.values)[:, 1][0])

    col_top1, col_top2 = st.columns(2)
    with col_top1:
        st.subheader("Original context")
        st.metric("P(high-value shot)", f"{proba_orig:.3f}")
        st.write(f"Player: {meta.get('player_name', 'Unknown')}")
        st.write(
            f"Game: {meta.get('game_id', '')}, "
            f"Event: {meta.get('eventnum', '')}, "
            f"Clock: {meta.get('pctimestring', '')}"
        )
        st.write(
            f"Score: {meta.get('score', '')}, "
            f"Margin: {meta.get('scoremargin', '')}"
        )

    with col_top2:
        st.subheader("What-if scenario")
        st.metric("P(high-value shot)", f"{proba_cf:.3f}")
        st.write(
            "You changed seconds remaining, score margin, offense side, "
            "and/or three-point flag on the left."
        )

    # ----- SHAP explanations -----
    try:
        sv_orig = explainer.shap_values(X_orig)
        sv_cf = explainer.shap_values(X_cf)

        shap_orig = get_class_shap(sv_orig)
        shap_cf = get_class_shap(sv_cf)

        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            fig1 = plot_shap_bar(shap_orig, feature_cols, "Original SHAP attributions")
            st.plotly_chart(fig1, use_container_width=True)
        with col_fig2:
            fig2 = plot_shap_bar(shap_cf, feature_cols, "What-if SHAP attributions")
            st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute SHAP values safely: {e}")

    # Footer with your name
    st.markdown(
        "<div style='text-align:right; color:gray; font-size:12px; margin-top:20px;'>"
        "Hongyi Duan"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()