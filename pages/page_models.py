import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import traceback

from sklearn.impute import SimpleImputer

from utils.data_loader import FEATURE_COLS, TARGET_COL
from utils.model_utils import (
    MODEL_NAMES, MODEL_COLORS,
    build_results_table, get_best_model, get_test_predictions,
    get_feature_importance, load_results, load_baselines, time_series_cv, load_model_bundles
)


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure df has a canonical 'date' column (try common names / parsing)."""
    df = df.copy()
    if 'date' in df.columns:
        return df
    # try some common alternatives
    for cand in ['datetime', 'timestamp', 'time', 'day', 'recorded_at']:
        if cand in df.columns:
            try:
                df['date'] = pd.to_datetime(df[cand], errors='coerce', infer_datetime_format=True)
                return df
            except Exception:
                continue
    # try parse any column that looks like dates
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c].astype(str), errors='coerce', infer_datetime_format=True)
            if parsed.notna().sum() >= max(1, int(len(df) * 0.2)):
                df['date'] = parsed
                return df
        except Exception:
            continue
    # fallback: use index as timestamp
    try:
        df['date'] = pd.to_datetime(df.index, errors='coerce')
    except Exception:
        df['date'] = pd.Timestamp.now()
    return df


def render(df, bundles):
    # show training progress if present
    tp = st.session_state.get("training_progress")
    if tp and tp.get("percent", 100) < 100:
        st.warning(f"Training in progress: {tp.get('message', '')}")
        st.progress(tp.get("percent", 0))
        st.divider()

    results = load_results() or {}
    baselines = load_baselines() or {}

    st.title("Models & Evaluation — Details")
    st.write("Training setup, baselines, cross-validation and diagnostic plots.")
    st.divider()

    # Prepare data: ensure date and drop rows where target is missing (we cannot train on those)
    df_proc = _ensure_date_column(df)
    df_clean = df_proc.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    if df_clean.empty:
        st.error("No rows with a valid target column present. Check your dataset and target mapping.")
        return

    # Chronological split
    n = len(df_clean)
    split = max(1, int(n * 0.8))
    train_df = df_clean.iloc[:split]
    test_df = df_clean.iloc[split:]

    col_info, col_bar = st.columns([3, 2])
    with col_info:
        try:
            tmin = pd.to_datetime(train_df['date']).min().date()
            tmax = pd.to_datetime(train_df['date']).max().date()
            tmin_t = pd.to_datetime(test_df['date']).min().date()
            tmax_t = pd.to_datetime(test_df['date']).max().date()
        except Exception:
            tmin = train_df['date'].iloc[0] if not train_df.empty else "N/A"
            tmax = train_df['date'].iloc[-1] if not train_df.empty else "N/A"
            tmin_t = test_df['date'].iloc[0] if not test_df.empty else "N/A"
            tmax_t = test_df['date'].iloc[-1] if not test_df.empty else "N/A"

        st.markdown(f"""
        | | Train set | Test set |
        |---|---:|---:|
        | Period | {tmin} → {tmax} | {tmin_t} → {tmax_t} |
        | Size | {len(train_df):,} (≈{int(100*len(train_df)/n)}%) | {len(test_df):,} (≈{int(100*len(test_df)/n)}%) |
        """)
        st.warning("Chronological split avoids leakage for time-series.")
    with col_bar:
        fig, ax = plt.subplots(figsize=(5, 1.6))
        ax.barh([""], [len(train_df)/max(1, n)], color="#2563eb", height=0.5)
        ax.barh([""], [len(test_df)/max(1, n)], left=len(train_df)/max(1, n), color="#dc2626", height=0.5)
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_title("Chronological split")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()

    # Features & Feature groups display (safe guards)
    st.subheader("Features & groups (feature engineering)")
    available_feats = [c for c in FEATURE_COLS if c in df_clean.columns]
    st.markdown(f"**{len(available_feats)} features** detected for modelling (from FEATURE_COLS).")
    feat_groups = {
        "Raw atmospheric": available_feats[:6],
        "Cyclical encoding": available_feats[6:10],
        "Calendar / ordinal": available_feats[10:12],
        "Lag & rolling": available_feats[12:],
    }
    for group, feats in feat_groups.items():
        with st.expander(group, expanded=False):
            st.write(", ".join(feats) if feats else "—")

    st.divider()

    # Baseline comparison and model results
    st.subheader("Baseline comparison vs. models (test set)")
    if baselines:
        st.markdown("**Baselines (test set)**")
        bl_rows = []
        for name, metrics in baselines.items():
            bl_rows.append({
                "Name": name,
                "MAE (°C)": metrics.get("MAE"),
                "RMSE (°C)": metrics.get("RMSE"),
                "MAPE (%)": metrics.get("MAPE%"),
                "R²": metrics.get("R2")
            })
        st.table(pd.DataFrame(bl_rows))
    else:
        st.info("Baselines not available — run training to compute baselines.")

    st.markdown("**Model results (test set)**")
    results_df = build_results_table(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    best_name = get_best_model(results)
    best_r2 = results.get(best_name, {}).get("R2", 0)
    st.success(f"Best model: {best_name} (R²={best_r2:.4f})")

    # Metric comparison graphs
    st.subheader("Metric comparison (MAE, RMSE, MAPE%)")
    mae_vals = [results.get(m, {}).get("MAE", np.nan) for m in MODEL_NAMES]
    rmse_vals = [results.get(m, {}).get("RMSE", np.nan) for m in MODEL_NAMES]
    mape_vals = [results.get(m, {}).get("MAPE%", np.nan) for m in MODEL_NAMES]
    colours = [MODEL_COLORS.get(m, "#333") for m in MODEL_NAMES]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    # MAE
    bars = axes[0].bar(MODEL_NAMES, mae_vals, color=colours, alpha=0.85, width=0.55)
    axes[0].set_title("MAE (°C)", fontsize=9); axes[0].grid(True, axis="y", alpha=0.4)
    for bar, val in zip(bars, mae_vals):
        if np.isfinite(val):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003 * max(mae_vals + [1e-6]), f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)
    # RMSE
    bars = axes[1].bar(MODEL_NAMES, rmse_vals, color=colours, alpha=0.85, width=0.55)
    axes[1].set_title("RMSE (°C)", fontsize=9); axes[1].grid(True, axis="y", alpha=0.4)
    for bar, val in zip(bars, rmse_vals):
        if np.isfinite(val):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003 * max(rmse_vals + [1e-6]), f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)
    # MAPE%
    bars = axes[2].bar(MODEL_NAMES, mape_vals, color=colours, alpha=0.85, width=0.55)
    axes[2].set_title("MAPE (%)", fontsize=9); axes[2].grid(True, axis="y", alpha=0.4)
    for bar, val in zip(bars, mape_vals):
        if np.isfinite(val):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003 * max(mape_vals + [1e-6]), f"{val:.3f}%", ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()

    # Time-series cross-validation on training set
    st.subheader("Five-fold time cross-validation (training set diagnostics)")
    if bundles is None:
        st.info("Trained model bundles not available. Retrain from the sidebar to compute time CV.")
    else:
        # Prepare a DF for CV: impute feature NaNs with median (same as training)
        try:
            df_cv = df_clean.copy().sort_values('date').reset_index(drop=True)
            features_for_cv = [c for c in FEATURE_COLS if c in df_cv.columns]
            if len(features_for_cv) == 0:
                st.info("No features found for CV (check FEATURE_COLS).")
            else:
                X_raw = df_cv[features_for_cv]
                imp = SimpleImputer(strategy="median")
                X_imp = pd.DataFrame(imp.fit_transform(X_raw), columns=features_for_cv, index=df_cv.index)
                df_cv[features_for_cv] = X_imp

                # Now run CV — pass df_cv which has no NaNs in feature columns used
                try:
                    cv_results = time_series_cv(df_cv, bundles, n_splits=5)
                    fold_fig, ax = plt.subplots(figsize=(9, 4))
                    for name in MODEL_NAMES:
                        df_cv_res = cv_results.get(name)
                        if df_cv_res is None or df_cv_res.empty:
                            continue
                        st.markdown(f"**{name} — time CV summary**")
                        st.table(df_cv_res[["fold", "MAE", "RMSE", "MAPE%", "R2"]].set_index("fold"))
                        ax.plot(df_cv_res["fold"], df_cv_res["MAE"], marker="o", label=name, color=MODEL_COLORS.get(name))
                    ax.set_xlabel("Fold"); ax.set_ylabel("MAE (°C)"); ax.set_title("Time CV — MAE by fold"); ax.grid(True, alpha=0.3); ax.legend()
                    plt.tight_layout(); st.pyplot(fold_fig, use_container_width=True); plt.close()
                except Exception as e:
                    st.error("Time CV failed — see traceback for details.")
                    st.text(traceback.format_exc())
        except Exception as e:
            st.error("Failed preparing data for time CV.")
            st.text(traceback.format_exc())

    st.divider()

    # Predicted vs Actual — test set for available bundles
    st.subheader("Predicted vs Actual — all models (test set)")
    if bundles is None:
        st.info("Trained model bundles not available. Retrain to see predicted vs actual plots.")
    else:
        try:
            test_preds = get_test_predictions(bundles, df_clean)
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
            for ax, name in zip(axes.ravel(), MODEL_NAMES):
                y_actual, y_pred, dates = test_preds.get(name, (np.array([]), np.array([]), np.array([])))
                mask = ~(np.isnan(y_actual) | np.isnan(y_pred))
                y_actual = np.asarray(y_actual)[mask]
                y_pred = np.asarray(y_pred)[mask]
                if len(y_actual) == 0:
                    ax.text(0.5, 0.5, "No valid data", ha="center")
                    continue
                lims = [min(y_actual.min(), y_pred.min()) - 1, max(y_actual.max(), y_pred.max()) + 1]
                ax.scatter(y_actual, y_pred, alpha=0.18, s=8, color=MODEL_COLORS.get(name))
                ax.plot(lims, lims, "r--", lw=1.5)
                ax.set_xlim(lims); ax.set_ylim(lims)
                ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(f"{name} — Pred vs Actual")
                ax.grid(True, alpha=0.35)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        except Exception:
            st.error("Failed to compute Pred vs Actual plots; see traceback.")
            st.text(traceback.format_exc())

        # Residual summaries
        st.markdown("**Residual summary (test set)**")
        cols = st.columns(3)
        try:
            for i, name in enumerate(MODEL_NAMES):
                y_actual, y_pred, _ = test_preds.get(name, (np.array([]), np.array([]), np.array([])))
                mask = ~(np.isnan(y_actual) | np.isnan(y_pred))
                y_actual = np.asarray(y_actual)[mask]
                y_pred = np.asarray(y_pred)[mask]
                residuals = y_pred - y_actual if len(y_actual) else np.array([0.0])
                with cols[i]:
                    st.metric(f"{name} mean residual", f"{np.mean(residuals):.4f}°C")
                    st.metric(f"{name} residual std", f"{np.std(residuals):.4f}°C")
                    st.metric(f"{name} max abs residual", f"{np.max(np.abs(residuals)):.2f}°C")
        except Exception:
            st.text(traceback.format_exc())

    st.divider()

    # Model choices explanation
    st.subheader("Why these models were chosen")
    st.markdown("""
    - **Linear Regression**: simple baseline parametric model and interpretable.
    - **Decision Tree**: non-linear splits capturing thresholds.
    - **Random Forest**: ensemble reducing variance and often the best out-of-the-box for tabular inputs.
    """)
    st.caption("Choose Random Forest as default when it consistently yields highest R² and lower MAE/MAPE across folds.")