import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.data_loader import FEATURE_COLS, TARGET_COL
from utils.model_utils import (
    MODEL_NAMES, MODEL_COLORS,
    build_results_table, get_best_model, get_test_predictions,
    get_feature_importance, load_results, load_baselines
)

def render(df, bundles):
    # If retraining progress exists in session state, show it here
    tp = st.session_state.get("training_progress")
    if tp and tp.get("percent", 100) < 100:
        st.warning(f"Training in progress: {tp.get('message', '')}")
        st.progress(tp.get("percent", 0))
        st.divider()

    results = load_results()
    baselines = load_baselines()

    st.title("Models & Evaluation")
    st.write("Training setup and comparison of the three models.")
    st.divider()

    # Train/test split explanation (chronological)
    st.subheader("1. Train / test split")
    df_clean = df.dropna(subset=[TARGET_COL])
    n = len(df_clean)
    split = int(n * 0.8)
    train_df = df_clean.iloc[:split]
    test_df = df_clean.iloc[split:]

    col_info, col_bar = st.columns([3, 2])
    with col_info:
        st.markdown(f"""
        | | Train set | Test set |
        |---|---:|---:|
        | Period | {train_df['date'].min().date()} → {train_df['date'].max().date()} | {test_df['date'].min().date()} → {test_df['date'].max().date()} |
        | Size | {len(train_df):,} (80%) | {len(test_df):,} (20%) |
        """)
        st.warning("Split is chronological (no shuffle). This prevents data leakage for time-series.")
    with col_bar:
        fig, ax = plt.subplots(figsize=(5, 1.6))
        ax.barh([""], [0.8], color="#2563eb", height=0.5)
        ax.barh([""], [0.2], left=0.8, color="#dc2626", height=0.5)
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_title("Chronological 80/20 split")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()

    # Features and target
    st.subheader("2. Features and target")
    col_feat, col_tgt = st.columns([3, 2])
    with col_feat:
        st.markdown(f"**{len(FEATURE_COLS)} features** used as inputs:")
        feat_groups = {
            "Raw atmospheric": FEATURE_COLS[:6],
            "Cyclical encoding": FEATURE_COLS[6:10],
            "Calendar / ordinal": FEATURE_COLS[10:12],
            "Lag & rolling": FEATURE_COLS[12:],
        }
        for group, features in feat_groups.items():
            with st.expander(group):
                st.code(", ".join(features))
    with col_tgt:
        st.markdown("**Target variable:**")
        st.info(f"`{TARGET_COL}` — Daily mean temperature (°C)")

    st.divider()

    # Results table
    st.subheader("3. Model results (test set)")
    results_df = build_results_table(results)
    best_name = get_best_model(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    best_r2 = results.get(best_name, {}).get("R2", 0)
    st.success(f"Best model: {best_name} (R²={best_r2:.4f})")

    st.divider()

    # Metric comparisons
    st.subheader("4. Metric comparison")
    mae_vals = [results.get(m, {}).get("MAE", 0) for m in MODEL_NAMES]
    rmse_vals = [results.get(m, {}).get("RMSE", 0) for m in MODEL_NAMES]
    r2_vals = [results.get(m, {}).get("R2", 0) for m in MODEL_NAMES]
    colours = [MODEL_COLORS[m] for m in MODEL_NAMES]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, vals, title, lower_is_better in [
        (axes[0], mae_vals, "MAE (°C)", True),
        (axes[1], rmse_vals, "RMSE (°C)", True),
        (axes[2], r2_vals, "R²", False),
    ]:
        best_val = min(vals) if lower_is_better else max(vals)
        bar_clrs = ["#16a34a" if abs(v - best_val) < 1e-9 else c for v, c in zip(vals, colours)]
        bars = ax.bar(MODEL_NAMES, vals, color=bar_clrs, alpha=0.85, width=0.55)
        ax.set_title(title, fontsize=9); ax.grid(True, axis="y", alpha=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003 * max(vals), f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)
        if not lower_is_better:
            ax.set_ylim(min(vals) - 0.01, 1.0)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    st.divider()

    # Predicted vs actual
    st.subheader(f"5. Predicted vs actual — {best_name}")
    test_preds = get_test_predictions(bundles, df)
    y_actual, y_pred, dates = test_preds[best_name]
    valid_mask = ~(np.isnan(y_actual) | np.isnan(y_pred))
    y_actual = y_actual[valid_mask]; y_pred = y_pred[valid_mask]

    rng = np.random.default_rng(42)
    n_pts = min(2000, len(y_actual))
    idx = np.sort(rng.choice(len(y_actual), n_pts, replace=False))
    residuals = y_pred - y_actual

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    lims = [float(min(y_actual.min(), y_pred.min())) - 1, float(max(y_actual.max(), y_pred.max())) + 1]
    axes[0].scatter(y_actual[idx], y_pred[idx], alpha=0.25, s=8, color="#2563eb")
    axes[0].plot(lims, lims, "r--", lw=1.8)
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted"); axes[0].set_title(f"Scatter — R²={results.get(best_name, {}).get('R2', 0):.4f}")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].grid(True, alpha=0.4)

    axes[1].hist(residuals, bins=55, alpha=0.72, edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="#dc2626", lw=1.8, ls="--")
    axes[1].axvline(residuals.mean(), color="#f59e0b", lw=1.5, ls=":")
    axes[1].set_xlabel("Residual (Predicted − Actual) °C"); axes[1].set_title("Residual distribution")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Mean residual", f"{residuals.mean():.4f}°C")
    col_r2.metric("Residual std", f"{residuals.std():.4f}°C")
    col_r3.metric("Max abs residual", f"{np.abs(residuals).max():.2f}°C")

    st.divider()

    # Random Forest feature importance
    st.subheader("6. Random Forest feature importance")
    fi_df = get_feature_importance(bundles)
    def feature_group_colour(name):
        if any(x in name for x in ("lag", "rolling")):
            return "#14b8a6", "Lag / rolling"
        elif any(x in name for x in ("sin", "cos", "season", "doy", "year")):
            return "#f59e0b", "Temporal encoding"
        else:
            return "#2563eb", "Raw atmospheric"
    fi_df["Colour"], fi_df["Group"] = zip(*fi_df["Feature"].apply(lambda f: feature_group_colour(f)))
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color=fi_df["Colour"][::-1].values, alpha=0.85)
    ax.set_xlabel("Feature importance"); ax.set_title("Random Forest feature importance")
    for bar, val in zip(bars, fi_df["Importance"][::-1].values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=7.5)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#14b8a6", label="Lag / rolling"), Patch(facecolor="#f59e0b", label="Temporal encoding"), Patch(facecolor="#2563eb", label="Raw atmospheric")]
    ax.legend(handles=legend_elements, fontsize=9, framealpha=0)
    ax.grid(True, axis="x", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    top5 = fi_df.head(5)
    st.caption("Top 5 features:")
    for _, row in top5.iterrows():
        st.caption(f"• {row['Feature']} ({row['Group']}) — {row['Importance']:.4f}")

    st.info("Lag features are the most important; cyclical temporal encodings follow.")