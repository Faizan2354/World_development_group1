"""
╔══════════════════════════════════════════════════════════════╗
║   P659 — Global Development Clustering  |  Streamlit App    ║
╚══════════════════════════════════════════════════════════════╝
Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Tuple, List          # Fix #1: avoid Python-3.10 built-in generics

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import Patch

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Development Clustering",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    section[data-testid="stSidebar"] { background: #0f1117; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg,#1e2330 0%,#252b3b 100%);
        border: 1px solid #2e3a4e; border-radius: 12px;
        padding: 18px 20px; box-shadow: 0 2px 12px rgba(0,0,0,.3);
    }
    div[data-testid="metric-container"] label {
        color:#8a9bbf !important; font-size:.82rem !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color:#63b3ed !important; font-size:1.6rem !important; font-weight:700;
    }

    .section-header {
        background: linear-gradient(90deg,#1a73e8 0%,#0d47a1 100%);
        color: white !important; padding: 10px 20px; border-radius: 8px;
        font-size: 1.05rem; font-weight: 600; margin: 20px 0 12px 0;
        letter-spacing: .5px;
    }
    .insight-box {
        background:#1a2436; border-left:4px solid #63b3ed; border-radius:6px;
        padding:12px 16px; margin:10px 0; font-size:.92rem; color:#c8d8ef;
    }
    .country-card {
        background:#1a2436; border:1px solid #2e3a4e; border-radius:10px;
        padding:14px 18px; margin:6px 0;
    }
    footer{visibility:hidden;} #MainMenu{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS: List[str] = [
    "GDP", "Health Exp/Capita", "Internet Usage",
    "Birth Rate", "Life Expectancy Female", "Life Expectancy Male",
    "Avg Life Expectancy", "CO2 Emissions", "Business Tax Rate",
    "Tourism Inbound", "Tourism Outbound",
]
COLS_TO_CLEAN: List[str] = [
    "GDP", "Tourism Inbound", "Tourism Outbound",
    "Health Exp/Capita", "Business Tax Rate",
]
POP_DENSITY_CANDIDATES: List[str] = [
    "Population Density", "Pop. Density (per sq. mi.)",
    "Population", "Pop", "pop_density",
]


# ─────────────────────────────────────────────────────────────────
# SAFE get_cmap  (Fix #3: cm.get_cmap deprecated in mpl 3.7+)
# ─────────────────────────────────────────────────────────────────
def _cmap(name: str, n: int = 10):
    try:
        return matplotlib.colormaps[name].resampled(n)
    except (AttributeError, KeyError):
        return plt.cm.get_cmap(name, n)


# ─────────────────────────────────────────────────────────────────
# DARK THEME
# ─────────────────────────────────────────────────────────────────
def set_dark():
    plt.rcParams.update({
        "figure.facecolor": "#0e1117", "axes.facecolor": "#161b27",
        "axes.edgecolor": "#2e3a4e",   "axes.labelcolor": "#c8d8ef",
        "xtick.color": "#8a9bbf",      "ytick.color": "#8a9bbf",
        "text.color": "#e0e0e0",       "grid.color": "#1e2d42",
        "grid.linewidth": 0.6,
    })


# ─────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_and_preprocess(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(file_bytes)
    for col in COLS_TO_CLEAN:
        if col in df.columns:
            df[col] = df[col].replace(r"[\$,%]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.drop(columns=["Ease of Business"], inplace=True, errors="ignore")
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = np.clip(df[col], Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1))
    if {"Life Expectancy Female", "Life Expectancy Male"} <= set(df.columns):
        df["Avg Life Expectancy"] = (
            df["Life Expectancy Female"] + df["Life Expectancy Male"]
        ) / 2
    return df


def get_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    avail = [c for c in FEATURE_COLS if c in df.columns]
    return StandardScaler().fit_transform(df[avail]), avail


def get_pca(X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled), pca.explained_variance_ratio_


def eval_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    n_valid = len(set(labels)) - (1 if -1 in labels else 0)
    if n_valid < 2:
        return {"Silhouette": None, "Davies-Bouldin": None, "Calinski-Harabasz": None}
    return {
        "Silhouette":        round(silhouette_score(X_scaled, labels), 4),
        "Davies-Bouldin":    round(davies_bouldin_score(X_scaled, labels), 4),
        "Calinski-Harabasz": round(calinski_harabasz_score(X_scaled, labels), 2),
    }


# Fix #4: pass n_features explicitly — no fragile outer-scope closure in cached fn
@st.cache_data(show_spinner="Computing elbow & silhouette curves…")
def compute_elbow(X_bytes: bytes, n_features: int) -> Tuple[list, list, list]:
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(-1, n_features)
    k_range, inertia, silh = list(range(2, 11)), [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        inertia.append(km.inertia_)
        silh.append(silhouette_score(X, lbl))
    return k_range, inertia, silh


# ─────────────────────────────────────────────────────────────────
# POPULATION DENSITY HELPERS
# ─────────────────────────────────────────────────────────────────
def get_pop_density_col(df: pd.DataFrame):
    for c in POP_DENSITY_CANDIDATES:
        if c in df.columns:
            return c, df[c]
    if "Birth Rate" in df.columns:
        return "Birth Rate (density proxy)", df["Birth Rate"]
    return None, None


def density_color(values: pd.Series) -> List[str]:
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = _cmap("YlOrRd", 256)
    return [mcolors.to_hex(cmap(norm(v))) for v in values]


# ─────────────────────────────────────────────────────────────────
# Fix #5: always return a flat ndarray of axes regardless of shape
# ─────────────────────────────────────────────────────────────────
def safe_flatten(axes) -> np.ndarray:
    return np.array(axes).flatten()


# ─────────────────────────────────────────────────────────────────
# SHARED SCATTER / BAR HELPERS
# ─────────────────────────────────────────────────────────────────
def pca_scatter(ax, labels, title, X_pca, evr):
    unique = sorted(set(labels))
    pal    = _cmap("tab10", max(len(unique), 2))
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1], s=45, alpha=0.78,
            color=pal(i / max(len(unique) - 1, 1)),
            label=("Noise" if lbl == -1 else f"Cluster {lbl}"),
            edgecolors="none",
        )
    ax.set_title(title, color="white", fontweight="bold")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", color="#c8d8ef")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", color="#c8d8ef")
    ax.tick_params(colors="#8a9bbf")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(alpha=0.2)


def bar_counts(ax, labels, title):
    unique, counts = np.unique(labels, return_counts=True)
    pal = _cmap("tab10", max(len(unique), 2))
    ax.bar(
        [str(u) for u in unique], counts,
        color=[pal(i / max(len(unique) - 1, 1)) for i in range(len(unique))],
        edgecolor="#0e1117",
    )
    ax.set_title(title, color="white", fontweight="bold")
    ax.set_xlabel("Cluster", color="#c8d8ef")
    ax.set_ylabel("Countries", color="#c8d8ef")
    ax.tick_params(colors="#8a9bbf")
    ax.grid(axis="y", alpha=0.2)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/921/921490.png", width=70)
    st.title("🌍 Global Dev Clustering")
    st.caption("P659 · Unsupervised ML Project")
    st.divider()

    uploaded = st.file_uploader(
        "📂 Upload dataset (.xlsx)", type=["xlsx"],
        help="Upload P659_World_development_dataset.xlsx",
    )
    st.divider()

    page = st.radio(
        "🧭 Navigate",
        [
            "📊 Overview & EDA",
            "🔍 Feature Analysis",
            "📈 Optimal K Selection",
            "🤖 Clustering Models",
            "📋 Model Comparison",
            "🗺️ Country Explorer",
            "🌐 Population Density",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**⚙️ Model Settings**")
    n_clusters_manual = st.slider("Override K (0 = auto)", 0, 10, 0)
    st.divider()
    st.caption("Built with Streamlit · sklearn · seaborn")


# ─────────────────────────────────────────────────────────────────
# GATE
# ─────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("## 🌍 Global Development Clustering Dashboard")
    st.info(
        "👈  **Upload your dataset** in the sidebar to begin.\n\n"
        "Expected: `P659_World_development_dataset.xlsx`",
        icon="📂",
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────
df              = load_and_preprocess(uploaded.read())
X_scaled, feat_cols = get_features(df)
X_pca, evr      = get_pca(X_scaled)

k_range, inertia_vals, silh_vals = compute_elbow(
    X_scaled.tobytes(), X_scaled.shape[1]
)
auto_k = k_range[silh_vals.index(max(silh_vals))]
best_k = n_clusters_manual if n_clusters_manual >= 2 else auto_k


# ═════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & EDA
# ═════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.markdown("## 📊 Dataset Overview & EDA")
    set_dark()

    num_countries = df["Country"].nunique() if "Country" in df.columns else len(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌐 Countries",           num_countries)
    c2.metric("📐 Raw Features",         df.shape[1])
    c3.metric("📐 Clustering Features",  len(feat_cols))
    c4.metric("🏅 Best K (auto)",        auto_k)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📉 Distributions", "🔥 Correlation"])

    with tab1:
        st.markdown('<div class="section-header">📋 Preprocessed Dataset</div>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(40), use_container_width=True, height=400)
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
        with col2:
            st.dataframe(
                df.dtypes.rename("dtype").reset_index()
                  .rename(columns={"index": "Column"}),
                use_container_width=True, height=220,
            )

    with tab2:
        num_df = df.select_dtypes(include="number")
        n_num  = len(num_df.columns)
        ncols  = 4
        nrows  = int(np.ceil(n_num / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
        fig.patch.set_facecolor("#0e1117")
        flat = safe_flatten(axes)
        for ax, col in zip(flat, num_df.columns):
            ax.set_facecolor("#161b27")
            ax.hist(num_df[col].dropna(), bins=25,
                    color="#3b82f6", edgecolor="#0e1117", alpha=0.85)
            ax.set_title(col, fontsize=8, color="#c8d8ef")
            ax.tick_params(labelsize=7)
        for ax in flat[n_num:]:
            ax.set_visible(False)
        plt.suptitle("Histograms of Numerical Features", fontsize=13,
                     color="white", fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Skewness
        skew   = num_df.skew().sort_values()
        colors = ["#ef4444" if v > 1 else "#3b82f6" if v < -1 else "#22c55e"
                  for v in skew.values]
        fig2, ax2 = plt.subplots(figsize=(11, max(3, len(skew) * 0.32)))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#161b27")
        ax2.barh(skew.index, skew.values, color=colors, edgecolor="#0e1117")
        ax2.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax2.set_title("Skewness  (red >1, blue <-1, green OK)",
                      color="white", fontsize=11)
        ax2.tick_params(colors="#c8d8ef", labelsize=8)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    with tab3:
        corr  = df.select_dtypes(include="number").corr()
        fig3, ax3 = plt.subplots(figsize=(14, 10))
        fig3.patch.set_facecolor("#0e1117")
        ax3.set_facecolor("#0e1117")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.4,
                    cmap="coolwarm", ax=ax3, annot_kws={"size": 7},
                    cbar_kws={"shrink": 0.7})
        ax3.set_title("Feature Correlation Matrix", color="white",
                      fontsize=13, fontweight="bold")
        ax3.tick_params(colors="#c8d8ef", labelsize=8)
        st.pyplot(fig3, use_container_width=True)
        plt.close()


# ═════════════════════════════════════════════════════════════════
# PAGE 2 — FEATURE ANALYSIS
# ═════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Analysis":
    st.markdown("## 🔍 Feature Analysis & Outlier Detection")
    set_dark()

    tab1, tab2, tab3 = st.tabs(["📦 Boxplots", "🔵 Scatter Plot", "📊 Stats"])

    with tab1:
        n_feat = len(feat_cols)
        ncols  = 4
        nrows  = int(np.ceil(n_feat / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
        fig.patch.set_facecolor("#0e1117")
        flat = safe_flatten(axes)
        for ax, col in zip(flat, feat_cols):
            ax.set_facecolor("#161b27")
            ax.boxplot(
                df[col].dropna(), patch_artist=True,
                boxprops=dict(facecolor="#3b82f6", color="#93c5fd"),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color="#60a5fa"),
                capprops=dict(color="#60a5fa"),
                flierprops=dict(markerfacecolor="#ef4444", markersize=4),
            )
            ax.set_title(col, fontsize=8, color="#c8d8ef")
            ax.tick_params(labelsize=7)
        for ax in flat[n_feat:]:
            ax.set_visible(False)
        plt.suptitle("Boxplots After IQR Outlier Capping",
                     fontsize=13, color="white", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        c1, c2 = st.columns(2)
        xi = feat_cols.index("Internet Usage") if "Internet Usage" in feat_cols else 0
        yi = feat_cols.index("Life Expectancy Male") if "Life Expectancy Male" in feat_cols else 1
        x_col = c1.selectbox("X-axis", feat_cols, index=xi)
        y_col = c2.selectbox("Y-axis", feat_cols, index=yi)
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        fig4.patch.set_facecolor("#0e1117")
        ax4.set_facecolor("#161b27")
        ax4.scatter(df[x_col], df[y_col], c="#3b82f6", alpha=0.65,
                    s=42, edgecolors="#60a5fa", linewidths=0.3)
        ax4.set_xlabel(x_col, color="#c8d8ef")
        ax4.set_ylabel(y_col, color="#c8d8ef")
        ax4.set_title(f"{x_col}  vs  {y_col}",
                      color="white", fontsize=12, fontweight="bold")
        ax4.grid(alpha=0.2)
        ax4.tick_params(colors="#8a9bbf")
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    with tab3:
        st.dataframe(df[feat_cols].describe().T.round(3), use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 3 — OPTIMAL K
# ═════════════════════════════════════════════════════════════════
elif page == "📈 Optimal K Selection":
    st.markdown("## 📈 Optimal Number of Clusters")
    set_dark()

    st.markdown(
        f'<div class="insight-box">🏆 Auto-detected <strong>optimal K = {auto_k}</strong> '
        f'via highest Silhouette Score '
        f'(<strong>{max(silh_vals):.4f}</strong>).</div>',
        unsafe_allow_html=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#161b27")
        ax.tick_params(colors="#8a9bbf")

    # Fix #2: removed conflicting "bo-" format + color= kwarg
    axes[0].plot(k_range, inertia_vals, "o-",
                 color="#3b82f6", linewidth=2.2, markersize=7,
                 markerfacecolor="white", markeredgecolor="#3b82f6")
    axes[0].axvline(auto_k, color="#ef4444", linewidth=1.5,
                    linestyle="--", label=f"Optimal K={auto_k}")
    axes[0].set_xlabel("Number of Clusters (k)", color="#c8d8ef")
    axes[0].set_ylabel("Inertia (WCSS)", color="#c8d8ef")
    axes[0].set_title("Elbow Method", color="white", fontweight="bold")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].plot(k_range, silh_vals, "o-",
                 color="#22c55e", linewidth=2.2, markersize=7,
                 markerfacecolor="white", markeredgecolor="#22c55e")
    axes[1].axvline(auto_k, color="#ef4444", linewidth=1.5,
                    linestyle="--", label=f"Optimal K={auto_k}")
    axes[1].set_xlabel("Number of Clusters (k)", color="#c8d8ef")
    axes[1].set_ylabel("Silhouette Score", color="#c8d8ef")
    axes[1].set_title("Silhouette Score vs k", color="white", fontweight="bold")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    k_df = pd.DataFrame({
        "K": k_range,
        "Inertia": inertia_vals,
        "Silhouette": [round(s, 4) for s in silh_vals],
    })
    k_df["Optimal"] = k_df["K"].apply(lambda x: "✅ Best" if x == auto_k else "")
    st.dataframe(k_df.set_index("K"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 4 — CLUSTERING MODELS
# ═════════════════════════════════════════════════════════════════
elif page == "🤖 Clustering Models":
    st.markdown(f"## 🤖 Clustering Models  (K = {best_k})")
    set_dark()

    model_tab = st.selectbox(
        "Select Model",
        ["K-Means", "Agglomerative", "DBSCAN", "Gaussian Mixture", "Mean Shift"],
    )

    if model_tab == "K-Means":
        km     = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        met    = eval_metrics(X_scaled, labels)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters",             best_k)
        c2.metric("Silhouette ↑",         met["Silhouette"])
        c3.metric("Davies-Bouldin ↓",     met["Davies-Bouldin"])
        c4.metric("Calinski-Harabasz ↑",  met["Calinski-Harabasz"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#161b27")
        pca_scatter(axes[0], labels, f"K-Means (k={best_k}) — PCA", X_pca, evr)
        bar_counts(axes[1], labels, "Countries per Cluster")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        if "Country" in df.columns:
            st.markdown("### 🌐 Cluster Assignments")
            dv = df[["Country"]].copy(); dv["Cluster"] = labels
            st.dataframe(dv.sort_values("Cluster"),
                         use_container_width=True, height=300)

    elif model_tab == "Agglomerative":
        agg    = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
        labels = agg.fit_predict(X_scaled)
        met    = eval_metrics(X_scaled, labels)
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette ↑",        met["Silhouette"])
        c2.metric("Davies-Bouldin ↓",    met["Davies-Bouldin"])
        c3.metric("Calinski-Harabasz ↑", met["Calinski-Harabasz"])
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#161b27")
        linked = linkage(X_scaled, method="ward")
        dendrogram(linked, ax=axes[0], truncate_mode="lastp", p=25,
                   leaf_rotation=90, color_threshold=0,
                   above_threshold_color="#3b82f6", leaf_font_size=7)
        axes[0].set_title("Dendrogram (Ward)", color="white", fontweight="bold")
        axes[0].tick_params(colors="#8a9bbf")
        pca_scatter(axes[1], labels, f"Agglomerative (k={best_k})", X_pca, evr)
        bar_counts(axes[2], labels, "Countries per Cluster")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    elif model_tab == "DBSCAN":
        c1, c2      = st.columns(2)
        eps         = c1.slider("eps",         0.1, 5.0, 1.2, 0.1)
        min_samples = c2.slider("min_samples", 2,   20,  3)
        db      = DBSCAN(eps=eps, min_samples=min_samples)
        labels  = db.fit_predict(X_scaled)
        n_clust = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        met     = eval_metrics(X_scaled, labels)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters Found",   n_clust)
        c2.metric("Noise Points",     n_noise)
        c3.metric("Silhouette ↑",     met["Silhouette"]     or "N/A")
        c4.metric("Davies-Bouldin ↓", met["Davies-Bouldin"] or "N/A")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#161b27")
        pca_scatter(axes[0], labels,
                    f"DBSCAN (eps={eps}, min={min_samples})", X_pca, evr)
        bar_counts(axes[1], labels, "Cluster Sizes")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    elif model_tab == "Gaussian Mixture":
        gmm    = GaussianMixture(n_components=best_k, random_state=42, n_init=5)
        labels = gmm.fit_predict(X_scaled)
        met    = eval_metrics(X_scaled, labels)
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette ↑",        met["Silhouette"])
        c2.metric("Davies-Bouldin ↓",    met["Davies-Bouldin"])
        c3.metric("Calinski-Harabasz ↑", met["Calinski-Harabasz"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#161b27")
        pca_scatter(axes[0], labels, f"Gaussian Mixture (k={best_k})", X_pca, evr)
        bar_counts(axes[1], labels, "Countries per Component")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    elif model_tab == "Mean Shift":
        with st.spinner("Estimating bandwidth…"):
            bw     = estimate_bandwidth(
                X_scaled, quantile=0.2, n_samples=min(300, len(X_scaled))
            )
            ms     = MeanShift(bandwidth=bw, bin_seeding=True)
            labels = ms.fit_predict(X_scaled)
        met = eval_metrics(X_scaled, labels)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters Found",      len(set(labels)))
        c2.metric("Silhouette ↑",        met["Silhouette"])
        c3.metric("Davies-Bouldin ↓",    met["Davies-Bouldin"])
        c4.metric("Calinski-Harabasz ↑", met["Calinski-Harabasz"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#161b27")
        pca_scatter(axes[0], labels, "Mean Shift Clustering", X_pca, evr)
        bar_counts(axes[1], labels, "Countries per Cluster")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════
elif page == "📋 Model Comparison":
    st.markdown(f"## 📋 Model Comparison  (K = {best_k})")
    set_dark()

    with st.spinner("Running all models…"):
        results = []
        for name, model in [
            ("K-Means",          KMeans(n_clusters=best_k, random_state=42, n_init=10)),
            ("Agglomerative",    AgglomerativeClustering(n_clusters=best_k, linkage="ward")),
            ("Gaussian Mixture", GaussianMixture(n_components=best_k, random_state=42, n_init=5)),
        ]:
            lbl = model.fit_predict(X_scaled)
            m   = eval_metrics(X_scaled, lbl)
            results.append({"Model": name, "Clusters": best_k,
                             "Silhouette ↑": m["Silhouette"],
                             "Davies-Bouldin ↓": m["Davies-Bouldin"],
                             "Calinski-Harabasz ↑": m["Calinski-Harabasz"]})
        lbl_db = DBSCAN(eps=1.2, min_samples=3).fit_predict(X_scaled)
        m      = eval_metrics(X_scaled, lbl_db)
        results.append({"Model": "DBSCAN",
                         "Clusters": len(set(lbl_db)) - (1 if -1 in lbl_db else 0),
                         "Silhouette ↑": m["Silhouette"],
                         "Davies-Bouldin ↓": m["Davies-Bouldin"],
                         "Calinski-Harabasz ↑": m["Calinski-Harabasz"]})

    comp_df = pd.DataFrame(results).set_index("Model")
    st.dataframe(
        comp_df.style
            .highlight_max(subset=["Silhouette ↑", "Calinski-Harabasz ↑"], color="#1a4a2e")
            .highlight_min(subset=["Davies-Bouldin ↓"], color="#1a4a2e"),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b27")
    valid  = comp_df["Silhouette ↑"].dropna()
    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
    ax.barh(valid.index, valid.values,
            color=colors[:len(valid)], edgecolor="#0e1117")
    ax.set_xlabel("Silhouette Score (higher = better)", color="#c8d8ef")
    ax.tick_params(colors="#c8d8ef")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("Silhouette Score per Model", color="white", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    if comp_df["Silhouette ↑"].dropna().any():
        bm = comp_df["Silhouette ↑"].idxmax()
        bs = comp_df["Silhouette ↑"].max()
        st.success(f"🏆 **Best Model: {bm}** · Silhouette = **{bs:.4f}**")


# ═════════════════════════════════════════════════════════════════
# PAGE 6 — COUNTRY EXPLORER  (per-country dropdown + cards)
# ═════════════════════════════════════════════════════════════════
elif page == "🗺️ Country Explorer":
    st.markdown("## 🗺️ Country Cluster Explorer")
    set_dark()

    if "Country" not in df.columns:
        st.warning("No 'Country' column found in the dataset.")
        st.stop()

    km         = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels  = km.fit_predict(X_scaled)
    df_exp     = df.copy()
    df_exp["Cluster"] = km_labels
    df_exp["PC1"]     = X_pca[:, 0]
    df_exp["PC2"]     = X_pca[:, 1]

    # ── Filters ──────────────────────────────────────────────────
    f1, f2, f3 = st.columns([1, 1, 2])
    sel_cluster = f1.selectbox("Filter by Cluster",
                               ["All"] + list(range(best_k)))
    search_txt  = f2.text_input("🔍 Search Country")
    f3.metric("Total Countries", len(df_exp))

    view_df = df_exp[
        ["Country", "Cluster"] + [c for c in feat_cols if c in df_exp.columns]
    ].copy()
    if sel_cluster != "All":
        view_df = view_df[view_df["Cluster"] == sel_cluster]
    if search_txt:
        view_df = view_df[
            view_df["Country"].str.contains(search_txt, case=False, na=False)
        ]

    st.dataframe(view_df.sort_values("Cluster"),
                 use_container_width=True, height=280)

    # ── PCA Scatter ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📡 PCA Space with Country Labels</div>',
                unsafe_allow_html=True)
    pal = _cmap("tab10", best_k)
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b27")
    for k_i in range(best_k):
        mask = df_exp["Cluster"] == k_i
        ax.scatter(df_exp.loc[mask, "PC1"], df_exp.loc[mask, "PC2"],
                   s=55, alpha=0.82,
                   color=pal(k_i / max(best_k - 1, 1)),
                   label=f"Cluster {k_i}", edgecolors="none")
        for _, row in df_exp[mask].iterrows():
            ax.annotate(row["Country"], (row["PC1"], row["PC2"]),
                        fontsize=5.5, color="#8a9bbf", alpha=0.75,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", color="#c8d8ef")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", color="#c8d8ef")
    ax.set_title("Countries in PCA Space", color="white",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(alpha=0.2)
    ax.tick_params(colors="#8a9bbf")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Cluster Mean Profiles ─────────────────────────────────────
    st.markdown('<div class="section-header">📊 Cluster Mean Profiles</div>',
                unsafe_allow_html=True)
    profile = df_exp.groupby("Cluster")[feat_cols].mean().round(3)
    st.dataframe(profile, use_container_width=True)

    # ── Per-Country Dropdown Card ─────────────────────────────────
    st.markdown('<div class="section-header">🔽 Per-Country Detail Card</div>',
                unsafe_allow_html=True)

    countries_list = sorted(view_df["Country"].dropna().unique())
    if not countries_list:
        st.info("No countries match the current filter.")
    else:
        sel_country = st.selectbox("Select a country to inspect:", countries_list)
        row_c = df_exp[df_exp["Country"] == sel_country].iloc[0]

        # Summary card
        st.markdown(
            f'<div class="country-card">'
            f'<h4 style="color:#63b3ed;margin:0 0 8px 0">🌐 {sel_country}</h4>'
            f'<b style="color:#8a9bbf">Cluster assigned:</b>&nbsp;'
            f'<span style="color:#f59e0b;font-size:1.1rem"><b>'
            f'{int(row_c["Cluster"])}</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Metric tiles for each feature
        feat_vals = {f: row_c[f] for f in feat_cols if f in row_c.index}
        items     = list(feat_vals.items())
        cols3     = st.columns(3)
        for idx, (feat, val) in enumerate(items):
            cols3[idx % 3].metric(feat, f"{val:,.2f}")

        # Bar: country vs cluster mean
        cl_mean  = profile.loc[int(row_c["Cluster"])]
        ctry_v   = [feat_vals.get(f, 0) for f in feat_cols]
        mean_v   = [cl_mean.get(f, 0)   for f in feat_cols]
        x_pos    = np.arange(len(feat_cols))

        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#161b27")
        ax2.bar(x_pos - 0.18, ctry_v, width=0.36,
                label=sel_country,  color="#3b82f6", edgecolor="#0e1117")
        ax2.bar(x_pos + 0.18, mean_v, width=0.36,
                label=f"Cluster {int(row_c['Cluster'])} Mean",
                color="#22c55e", edgecolor="#0e1117", alpha=0.75)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(feat_cols, rotation=40, ha="right",
                             fontsize=7, color="#c8d8ef")
        ax2.tick_params(axis="y", colors="#8a9bbf")
        ax2.set_title(f"{sel_country} vs Cluster Mean",
                      color="white", fontweight="bold")
        ax2.legend(fontsize=8, framealpha=0.3)
        ax2.grid(axis="y", alpha=0.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # Expandable full list
        with st.expander("📋 All Countries in Current View"):
            life_col = ("Avg Life Expectancy"
                        if "Avg Life Expectancy" in df_exp.columns
                        else "Life Expectancy Male"
                        if "Life Expectancy Male" in df_exp.columns else None)
            for country in countries_list:
                r = df_exp[df_exp["Country"] == country].iloc[0]
                life_str = (f"Life Exp: {r[life_col]:.1f}" if life_col else "")
                gdp_str  = (f"GDP: {r['GDP']:,.0f}" if "GDP" in r.index else "")
                st.markdown(
                    f"**{country}** — "
                    f"<span style='color:#f59e0b'>Cluster {int(r['Cluster'])}</span>"
                    f"{' | ' + gdp_str if gdp_str else ''}"
                    f"{' | ' + life_str if life_str else ''}",
                    unsafe_allow_html=True,
                )


# ═════════════════════════════════════════════════════════════════
# PAGE 7 — POPULATION DENSITY (Light → Dense colour ramp)
# ═════════════════════════════════════════════════════════════════
elif page == "🌐 Population Density":
    st.markdown("## 🌐 World Population Density")
    set_dark()

    if "Country" not in df.columns:
        st.warning("No 'Country' column found.")
        st.stop()

    pop_col, pop_series = get_pop_density_col(df)

    if pop_series is None:
        st.error("No population or density column found in the dataset.")
        st.stop()

    st.markdown(
        f'<div class="insight-box">'
        f'📌 Colouring countries by <strong>{pop_col}</strong>. &nbsp;'
        f'<span style="color:#fde68a">🟡 Light yellow</span> = low / sparse &nbsp;→&nbsp;'
        f'<span style="color:#ef4444">🔴 Dark red</span> = high / dense.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Build density table
    dens_df = pd.DataFrame({
        "Country": df["Country"].values,
        "Value":   pop_series.values,
    }).dropna().sort_values("Value", ascending=False).reset_index(drop=True)

    dens_df["Colour"] = density_color(dens_df["Value"])

    q25, q50, q75 = dens_df["Value"].quantile([0.25, 0.50, 0.75]).values

    def band_label(v):
        if v <= q25: return "🟡 Very Light"
        if v <= q50: return "🟠 Light-Medium"
        if v <= q75: return "🔶 Medium-Dense"
        return              "🔴 Very Dense"

    dens_df["Density Band"] = dens_df["Value"].apply(band_label)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries",          len(dens_df))
    c2.metric(f"Min {pop_col}",     f"{dens_df['Value'].min():,.1f}")
    c3.metric(f"Max {pop_col}",     f"{dens_df['Value'].max():,.1f}")
    c4.metric(f"Median {pop_col}",  f"{dens_df['Value'].median():,.1f}")

    t1, t2, t3 = st.tabs(["📊 Ranked Bar Chart", "🗂️ Band Table", "🔽 Country Dropdown"])

    # ── Tab 1: Ranked Bar Chart ───────────────────────────────────
    with t1:
        top_n  = st.slider("Show top N countries", 10, len(dens_df), 40, 5)
        top_df = dens_df.head(top_n)

        fig, ax = plt.subplots(figsize=(14, max(5, top_n * 0.35)))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#161b27")
        ax.barh(
            top_df["Country"][::-1].tolist(),
            top_df["Value"][::-1].tolist(),
            color=top_df["Colour"][::-1].tolist(),
            edgecolor="#0e1117", linewidth=0.4,
        )
        ax.set_xlabel(pop_col, color="#c8d8ef", fontsize=10)
        ax.set_title(f"Top {top_n} Countries by {pop_col}",
                     color="white", fontsize=13, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7.5, colors="#c8d8ef")
        ax.tick_params(axis="x", colors="#8a9bbf")
        ax.grid(axis="x", alpha=0.2)

        legend_items = [
            Patch(facecolor="#fff7bc", label="🟡 Very Light   (≤ Q25)"),
            Patch(facecolor="#fd8d3c", label="🟠 Light-Medium (Q25–Q50)"),
            Patch(facecolor="#e31a1c", label="🔶 Medium-Dense (Q50–Q75)"),
            Patch(facecolor="#67000d", label="🔴 Very Dense   (> Q75)"),
        ]
        ax.legend(handles=legend_items, loc="lower right",
                  fontsize=8, framealpha=0.35)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Tab 2: Band Table + Pie ───────────────────────────────────
    with t2:
        band_opts   = sorted(dens_df["Density Band"].unique().tolist())
        band_filter = st.multiselect("Filter by Density Band",
                                     options=band_opts, default=band_opts)
        show_df = dens_df[dens_df["Density Band"].isin(band_filter)][
            ["Country", "Value", "Density Band"]
        ].rename(columns={"Value": pop_col})
        st.dataframe(show_df, use_container_width=True, height=360)

        band_counts = dens_df["Density Band"].value_counts()
        fig2, ax2   = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")
        pie_colors  = ["#fff7bc", "#fd8d3c", "#e31a1c", "#67000d"]
        ax2.pie(
            band_counts.values,
            labels=band_counts.index,
            colors=pie_colors[:len(band_counts)],
            autopct="%1.0f%%",
            textprops={"color": "white", "fontsize": 9},
            startangle=140,
        )
        ax2.set_title("Density Band Distribution",
                      color="white", fontweight="bold")
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── Tab 3: Per-Country Dropdown ───────────────────────────────
    with t3:
        st.markdown("### 🔽 Inspect a Country's Density Position")
        sel_pop = st.selectbox(
            "Select Country:", sorted(dens_df["Country"].tolist()),
            key="pop_country_sel",
        )
        row_p    = dens_df[dens_df["Country"] == sel_pop].iloc[0]
        pct_rank = (dens_df["Value"] <= row_p["Value"]).mean() * 100

        st.markdown(
            f'<div class="country-card">'
            f'<h4 style="color:#63b3ed;margin:0 0 8px 0">🌐 {sel_pop}</h4>'
            f'<table style="width:100%;border:none">'
            f'<tr><td style="color:#8a9bbf">{pop_col}</td>'
            f'    <td style="color:#f59e0b;font-weight:700">'
            f'        {row_p["Value"]:,.2f}</td></tr>'
            f'<tr><td style="color:#8a9bbf">Density Band</td>'
            f'    <td style="color:#e0e0e0">{row_p["Density Band"]}</td></tr>'
            f'<tr><td style="color:#8a9bbf">Percentile Rank</td>'
            f'    <td style="color:#22c55e">{pct_rank:.1f}th</td></tr>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

        # Distribution + marker
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        fig3.patch.set_facecolor("#0e1117")
        ax3.set_facecolor("#161b27")
        ax3.hist(dens_df["Value"], bins=30, color="#3b82f6",
                 edgecolor="#0e1117", alpha=0.75)
        ax3.axvline(row_p["Value"], color="#ef4444", linewidth=2.2,
                    linestyle="--",
                    label=f"{sel_pop}: {row_p['Value']:,.1f}")
        ax3.set_xlabel(pop_col, color="#c8d8ef")
        ax3.set_ylabel("Countries", color="#c8d8ef")
        ax3.set_title(f"{sel_pop} vs Global Distribution",
                      color="white", fontweight="bold")
        ax3.tick_params(colors="#8a9bbf")
        ax3.legend(fontsize=9, framealpha=0.3)
        ax3.grid(alpha=0.2)
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        # Band peers table
        peers = (dens_df[dens_df["Density Band"] == row_p["Density Band"]]
                 .sort_values("Value", ascending=False))
        st.markdown(f"**Other countries in band {row_p['Density Band']}:**")
        st.dataframe(
            peers[["Country", "Value"]].rename(columns={"Value": pop_col}),
            use_container_width=True, height=240,
        )
