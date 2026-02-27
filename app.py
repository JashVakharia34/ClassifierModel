import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Analyser", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0e0e14; color: #e8e8f0; }
[data-testid="stSidebar"] { background-color: #15151f; border-right: 1px solid #2a2a3d; }
h1 { font-family: 'Space Mono', monospace !important; color: #a78bfa !important; letter-spacing: -1px; font-size: 1.8rem !important; }
h2, h3, h4 { font-family: 'Space Mono', monospace !important; color: #c4b5fd !important; }
[data-testid="stMetric"] { background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px; padding: 16px !important; }
[data-testid="stMetricValue"] { color: #a78bfa !important; font-family: 'Space Mono', monospace !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #888 !important; }
.stButton > button { background: linear-gradient(135deg, #7c3aed, #4f46e5); color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem; font-family: 'Space Mono', monospace; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; width: 100%; transition: all 0.2s ease; }
.stButton > button:hover { background: linear-gradient(135deg, #6d28d9, #4338ca); transform: translateY(-1px); box-shadow: 0 4px 20px rgba(124,58,237,0.4); }
hr { border-color: #2a2a3d; }
[data-testid="stFileUploadDropzone"] { background: #1a1a2e !important; border: 2px dashed #4f46e5 !important; border-radius: 12px !important; }
.stAlert { border-radius: 10px !important; }
[data-baseweb="tab-list"] { background: #15151f !important; border-bottom: 1px solid #2a2a3d !important; gap: 4px; }
[data-baseweb="tab"] { font-family: 'Space Mono', monospace !important; font-size: 0.82rem !important; color: #888 !important; padding: 10px 28px !important; border-radius: 6px 6px 0 0 !important; }
[aria-selected="true"] { background: #1a1a2e !important; color: #a78bfa !important; border-bottom: 2px solid #7c3aed !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ML Analyser")
st.markdown("<p style='color:#888;font-size:0.92rem;margin-top:-10px;'>Upload a dataset — explore it or evaluate a model.</p>", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded — {df.shape[0]} rows x {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.divider()

    if df is not None:
        st.markdown("### Model Type")
        mode = st.radio("Select mode", ["Classification", "Regression"], index=0)
        is_regression = mode == "Regression"

        if not is_regression:
            clf_model_name = st.selectbox(
                "Classification algorithm",
                ["Naive Bayes (Gaussian)"],
                key="clf_model",
            )
        else:
            clf_model_name = None

        st.markdown("### Target & Features")
        columns = df.columns.tolist()

        categorical_cols = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 10]
        numeric_cols     = [c for c in columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 10]

        smart_cols  = numeric_cols if is_regression else categorical_cols
        smart_cols  = smart_cols if smart_cols else columns
        default_idx = columns.index(smart_cols[0])

        target_col = st.selectbox(
            "Target column", columns,
            index=default_idx,
            key=f"target_{'reg' if is_regression else 'cls'}",
        )

        is_num_target = pd.api.types.is_numeric_dtype(df[target_col])
        n_unique_t    = df[target_col].nunique()
        if is_regression and not (is_num_target and n_unique_t > 10):
            st.warning("Target looks categorical. Classification may work better.")
        elif not is_regression and is_num_target and n_unique_t > 10:
            st.warning("Target looks continuous. Regression may work better.")

        feature_options   = [c for c in columns if c != target_col]
        selected_features = st.multiselect("Feature columns", feature_options, default=feature_options)

        st.divider()
        st.markdown("### Parameters")
        test_size    = st.slider("Test set size", 0.10, 0.50, 0.20, 0.05, format="%.0f%%")
        random_state = st.number_input("Random seed", value=42, step=1)

        st.divider()
        run_btn = st.button("EVALUATE MODEL")
    else:
        st.info("Upload a CSV file to get started.")
        run_btn       = False
        is_regression = False

# ── Landing ───────────────────────────────────────────────────────────────────
if df is None:
    c1, c2, c3 = st.columns(3)
    cards = [
        ("Upload",   "Drop any CSV with numeric features and a target column."),
        ("Explore",  "Visualise distributions, correlations, and missing values."),
        ("Evaluate", "Run Naive Bayes or Linear Regression and inspect results."),
    ]
    for col, (title, desc) in zip([c1, c2, c3], cards):
        col.markdown(f"""
        <div style='background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:24px;'>
        <h4 style='margin-top:0;color:#c4b5fd;font-family:Space Mono,monospace;'>{title}</h4>
        <p style='color:#888;font-size:0.88rem;margin:0;'>{desc}</p>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Style constants ───────────────────────────────────────────────────────────
BG   = "#0e0e14"
BG2  = "#1a1a2e"
LINE = "#2a2a3d"
PUR  = "#7c3aed"
PUR2 = "#4f46e5"
LPUR = "#a78bfa"

def styled_fig(w=6, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax

def style_ax(ax):
    ax.tick_params(colors="#888")
    for sp in ax.spines.values(): sp.set_edgecolor(LINE)
    ax.yaxis.grid(True, color=LINE, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_eda, tab_model = st.tabs(["  Exploratory Analysis  ", "  Model Evaluator  "])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Overview
    st.markdown("### Dataset Overview")
    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Rows",             df.shape[0])
    o2.metric("Columns",          df.shape[1])
    o3.metric("Numeric cols",     len(num_cols))
    o4.metric("Categorical cols", len(cat_cols))
    o5.metric("Missing values",   int(df.isnull().sum().sum()))
    st.divider()

    # Raw data + types
    with st.expander("Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True)

    with st.expander("Column Info & Missing Values", expanded=False):
        info_df = pd.DataFrame({
            "Type":      df.dtypes.astype(str),
            "Missing":   df.isnull().sum(),
            "Missing %": (df.isnull().mean() * 100).round(2),
            "Unique":    df.nunique(),
        })
        st.dataframe(info_df, use_container_width=True)

    st.divider()

    # Descriptive stats
    st.markdown("### Descriptive Statistics")
    st.dataframe(df[num_cols].describe().T.round(4), use_container_width=True)
    st.divider()

    # Distributions
    st.markdown("### Distributions")
    if num_cols:
        dist_col = st.selectbox("Select column", num_cols, key="dist_col")
        col_d1, col_d2 = st.columns(2, gap="large")

        with col_d1:
            fig, ax = styled_fig(6, 4)
            ax.hist(df[dist_col].dropna(), bins=30, color=PUR, alpha=0.85,
                    edgecolor=LPUR, linewidth=0.4)
            ax.set_xlabel(dist_col, color="#888", fontsize=10)
            ax.set_ylabel("Frequency", color="#888", fontsize=10)
            ax.set_title("Histogram", color=LPUR, fontsize=11, fontfamily="monospace")
            style_ax(ax)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_d2:
            fig, ax = styled_fig(6, 4)
            ax.boxplot(df[dist_col].dropna(), patch_artist=True,
                       medianprops=dict(color=LPUR, linewidth=2),
                       boxprops=dict(facecolor=BG2, color=PUR),
                       whiskerprops=dict(color="#555"),
                       capprops=dict(color="#555"),
                       flierprops=dict(marker='o', color=PUR, alpha=0.4, markersize=4))
            ax.set_xticklabels([dist_col], color="#888")
            ax.set_ylabel("Value", color="#888", fontsize=10)
            ax.set_title("Box Plot", color=LPUR, fontsize=11, fontfamily="monospace")
            style_ax(ax)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()

    # Correlation
    st.markdown("### Correlation Matrix")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        sz   = max(6, len(num_cols))
        fig, ax = plt.subplots(figsize=(sz, sz - 1))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdPu",
                    ax=ax, linewidths=0.4, linecolor=LINE,
                    cbar_kws={"shrink": 0.7}, annot_kws={"size": 9})
        ax.tick_params(colors="#888", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(LINE)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Need at least 2 numeric columns.")

    st.divider()

    # Scatter
    st.markdown("### Scatter Plot")
    if len(num_cols) >= 2:
        sc1, sc2, sc3 = st.columns(3)
        x_col   = sc1.selectbox("X axis",  num_cols, index=0, key="sc_x")
        y_col   = sc2.selectbox("Y axis",  num_cols, index=min(1, len(num_cols)-1), key="sc_y")
        hue_col = sc3.selectbox("Colour by", ["None"] + cat_cols, key="sc_hue")

        fig, ax = styled_fig(8, 5)
        if hue_col == "None":
            ax.scatter(df[x_col], df[y_col], color=PUR, alpha=0.6,
                       edgecolors=LPUR, linewidth=0.3, s=30)
        else:
            palette = sns.color_palette("cool", n_colors=df[hue_col].nunique())
            for i, val in enumerate(df[hue_col].unique()):
                mask = df[hue_col] == val
                ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                           color=palette[i % len(palette)], alpha=0.65,
                           edgecolors="none", s=30, label=str(val))
            ax.legend(facecolor=BG2, edgecolor=LINE, labelcolor=LPUR,
                      fontsize=8, title=hue_col, title_fontsize=8)
        ax.set_xlabel(x_col, color="#888", fontsize=10)
        ax.set_ylabel(y_col, color="#888", fontsize=10)
        style_ax(ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()

    # Categorical counts
    if cat_cols:
        st.markdown("### Categorical Counts")
        cat_sel = st.selectbox("Select column", cat_cols, key="cat_bar")
        vc = df[cat_sel].value_counts()
        fig, ax = styled_fig(7, 4)
        bars = ax.bar(vc.index.astype(str), vc.values, color=PUR, alpha=0.9,
                      edgecolor=LPUR, linewidth=0.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(int(bar.get_height())), ha='center', va='bottom',
                    color=LPUR, fontsize=9)
        ax.set_xlabel(cat_sel, color="#888", fontsize=10)
        ax.set_ylabel("Count", color="#888", fontsize=10)
        style_ax(ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.divider()

    # Pairplot
    st.markdown("### Pairplot")
    if len(num_cols) >= 2:
        pp_cols = st.multiselect("Columns (max 5)", num_cols,
                                 default=num_cols[:min(4, len(num_cols))], key="pp_cols")
        pp_hue  = st.selectbox("Colour by", ["None"] + cat_cols, key="pp_hue")

        if len(pp_cols) >= 2:
            subset = df[pp_cols + ([pp_hue] if pp_hue != "None" else [])].dropna()
            with st.spinner("Generating pairplot..."):
                pp_fig = sns.pairplot(
                    subset,
                    hue=pp_hue if pp_hue != "None" else None,
                    plot_kws=dict(alpha=0.5, s=15),
                    diag_kind="kde",
                    palette="cool",
                )
                pp_fig.figure.patch.set_facecolor(BG)
                for ax_ in pp_fig.axes.flatten():
                    if ax_:
                        ax_.set_facecolor(BG2)
                        ax_.tick_params(colors="#888", labelsize=7)
                        for sp in ax_.spines.values(): sp.set_edgecolor(LINE)
                plt.tight_layout()
                st.pyplot(pp_fig.figure); plt.close()
        else:
            st.info("Select at least 2 columns.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:

    if not run_btn:
        st.markdown("### Model Evaluator")
        st.markdown("<p style='color:#888;'>Configure your model in the sidebar, then click <b>EVALUATE MODEL</b>.</p>",
                    unsafe_allow_html=True)
        with st.expander("Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Missing values", int(df.isnull().sum().sum()))

    if run_btn:
        if not selected_features:
            st.warning("Please select at least one feature column.")
            st.stop()

        try:
            X = df[selected_features].select_dtypes(include=[np.number]).values
            y = df[target_col].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state)
            )

            st.markdown("### Results")

            # CLASSIFICATION
            if not is_regression:
                model   = GaussianNB()
                model.fit(X_train, y_train)
                y_pred  = model.predict(X_test)
                acc     = accuracy_score(y_test, y_pred)
                cm      = confusion_matrix(y_test, y_pred)
                report  = classification_report(y_test, y_pred, output_dict=True)
                classes = model.classes_

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy",      f"{acc*100:.2f}%")
                m2.metric("Test samples",  len(y_test))
                m3.metric("Train samples", len(y_train))
                m4.metric("Classes",       len(classes))
                st.divider()

                col_cm, col_rep = st.columns(2, gap="large")

                with col_cm:
                    st.markdown("### Confusion Matrix")
                    fig, ax = styled_fig()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                                xticklabels=classes, yticklabels=classes,
                                linewidths=0.5, linecolor=LINE, ax=ax,
                                cbar_kws={"shrink": 0.8})
                    ax.set_xlabel("Predicted", color="#888", fontsize=11)
                    ax.set_ylabel("Actual",    color="#888", fontsize=11)
                    ax.tick_params(colors=LPUR)
                    for sp in ax.spines.values(): sp.set_edgecolor(LINE)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                with col_rep:
                    st.markdown("### Classification Report")
                    rows = []
                    for label in [str(c) for c in classes]:
                        if label in report:
                            r = report[label]
                            rows.append({"Class": label, "Precision": f"{r['precision']:.3f}",
                                         "Recall": f"{r['recall']:.3f}", "F1": f"{r['f1-score']:.3f}",
                                         "Support": int(r['support'])})
                    for avg in ["macro avg", "weighted avg"]:
                        if avg in report:
                            r = report[avg]
                            rows.append({"Class": avg, "Precision": f"{r['precision']:.3f}",
                                         "Recall": f"{r['recall']:.3f}", "F1": f"{r['f1-score']:.3f}",
                                         "Support": int(r['support'])})
                    st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)
                    st.divider()

                    class_labels = [str(c) for c in classes]
                    precisions = [report[l]["precision"] for l in class_labels if l in report]
                    recalls    = [report[l]["recall"]    for l in class_labels if l in report]
                    f1s        = [report[l]["f1-score"]  for l in class_labels if l in report]

                    fig2, ax2 = styled_fig(6, 3.5)
                    x, w = np.arange(len(class_labels)), 0.25
                    ax2.bar(x - w, precisions, w, label="Precision", color=PUR,  alpha=0.9)
                    ax2.bar(x,     recalls,    w, label="Recall",    color=PUR2, alpha=0.9)
                    ax2.bar(x + w, f1s,        w, label="F1",        color=LPUR, alpha=0.9)
                    ax2.set_xticks(x); ax2.set_xticklabels(class_labels, color=LPUR, fontsize=9)
                    ax2.set_ylim(0, 1.15); ax2.set_ylabel("Score", color="#888", fontsize=10)
                    ax2.legend(facecolor=BG2, edgecolor=LINE, labelcolor=LPUR, fontsize=9)
                    style_ax(ax2)
                    plt.tight_layout(); st.pyplot(fig2); plt.close()

            # REGRESSION
            else:
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred    = model.predict(X_test)
                mse       = mean_squared_error(y_test, y_pred)
                rmse      = np.sqrt(mse)
                r2        = r2_score(y_test, y_pred)
                residuals = y_test - y_pred

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R2 Score",      f"{r2:.4f}")
                m2.metric("RMSE",          f"{rmse:.4f}")
                m3.metric("Test samples",  len(y_test))
                m4.metric("Train samples", len(y_train))
                st.divider()

                col_l, col_r = st.columns(2, gap="large")

                with col_l:
                    st.markdown("### Actual vs Predicted")
                    fig, ax = styled_fig()
                    ax.scatter(y_test, y_pred, color=PUR, alpha=0.7, edgecolors=LPUR, linewidth=0.4, s=30)
                    mn = float(min(y_test.min(), y_pred.min()))
                    mx = float(max(y_test.max(), y_pred.max()))
                    ax.plot([mn, mx], [mn, mx], color=PUR2, linewidth=1.5, linestyle="--", label="Perfect fit")
                    ax.set_xlabel("Actual", color="#888", fontsize=11)
                    ax.set_ylabel("Predicted", color="#888", fontsize=11)
                    ax.legend(facecolor=BG2, edgecolor=LINE, labelcolor=LPUR, fontsize=9)
                    style_ax(ax)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                with col_r:
                    st.markdown("### Residual Distribution")
                    fig, ax = styled_fig()
                    ax.hist(residuals, bins=20, color=PUR, alpha=0.85, edgecolor=LPUR, linewidth=0.4)
                    ax.axvline(0, color=PUR2, linewidth=1.5, linestyle="--", label="Zero residual")
                    ax.set_xlabel("Residual (Actual - Predicted)", color="#888", fontsize=10)
                    ax.set_ylabel("Frequency", color="#888", fontsize=10)
                    ax.legend(facecolor=BG2, edgecolor=LINE, labelcolor=LPUR, fontsize=9)
                    style_ax(ax)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                    st.divider()
                    st.markdown("### Feature Coefficients")
                    coef_df = pd.DataFrame({
                        "Feature":     selected_features[:len(model.coef_)],
                        "Coefficient": model.coef_.round(4)
                    }).sort_values("Coefficient", key=abs, ascending=False)
                    st.dataframe(coef_df.set_index("Feature"), use_container_width=True)

        except Exception as e:
            st.error(f"Model error: {e}")