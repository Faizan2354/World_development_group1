"""
╔══════════════════════════════════════════════════════════════╗
║   P659 — Global Development Clustering  |  Streamlit App    ║
╚══════════════════════════════════════════════════════════════╝
Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from scipy.cluster.hierarchy import dendrogram, linkage

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
    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0f1117; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2330 0%, #252b3b 100%);
        border: 1px solid #2e3a4e;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label { color: #8a9bbf !important; font-size: 0.82rem !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #63b3ed !important; font-size: 1.6rem !important; font-weight: 700;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1a73e8 0%, #0d47a1 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 20px 0 12px 0;
        letter-spacing: 0.5px;
    }

    /* Info / success boxes */
    .insight-box {
        background: #1a2436;
        border-left: 4px solid #63b3ed;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #c8d8ef;
    }

    /* Table styling */
    table { width: 100%; }
    thead tr { background-color: #1e3a5f; }
    tbody tr:nth-child(even) { background-color: #1a2130; }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "GDP", "Health Exp/Capita", "Internet Usage",
    "Birth Rate", "Life Expectancy Female", "Life Expectancy Male",
    "Avg Life Expectancy", "CO2 Emissions", "Business Tax Rate",
    "Tourism Inbound", "Tourism Outbound",
]

COLS_TO_CLEAN = ["GDP", "Tourism Inbound", "Tourism Outbound",
                 "Health Exp/Capita", "Business Tax Rate"]


@st.cache_data(show_spinner="⚙️  Loading & preprocessing data…")
def load_and_preprocess(file_bytes: bytes) -> pd.DataFrame:
    """Load, clean, impute and engineer features from the raw Excel file."""
    df = pd.read_excel(file_bytes)

    # Clean currency / percentage symbols
    for col in COLS_TO_CLEAN:
        if col in df.columns:
            df[col] = df[col].replace(r"[\$,%]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop high-missing column
    df.drop(columns=["Ease of Business"], inplace=True, errors="ignore")

    # Impute with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # IQR outlier capping
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Feature engineering
    if "Life Expectancy Female" in df.columns and "Life Expectancy Male" in df.columns:
        df["Avg Life Expectancy"] = (
            df["Life Expectancy Female"] + df["Life Expectancy Male"]
        ) / 2

    return df


def get_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return scaled feature matrix and feature names."""
    avail = [c for c in FEATURE_COLS if c in df.columns]
    X = df[avail].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, avail


def get_pca(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_


def eval_metrics(X_scaled, labels) -> dict:
    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if n_labels < 2:
        return {"Silhouette": None, "Davies-Bouldin": None, "Calinski-Harabasz": None}
    return {
        "Silhouette": round(silhouette_score(X_scaled, labels), 4),
        "Davies-Bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
        "Calinski-Harabasz": round(calinski_harabasz_score(X_scaled, labels), 2),
    }


def styled_fig():
    """Return a dark-themed matplotlib figure / axes."""
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#161b27",
        "axes.edgecolor": "#2e3a4e",
        "axes.labelcolor": "#c8d8ef",
        "xtick.color": "#8a9bbf",
        "ytick.color": "#8a9bbf",
        "text.color": "#e0e0e0",
        "grid.color": "#1e2d42",
        "grid.linewidth": 0.6,
    })


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/921/921490.png",
        width=70,
    )
    st.title("🌍 Global Dev Clustering")
    st.caption("P659 · Unsupervised ML Project")
    st.divider()

    uploaded = st.file_uploader(
        "📂 Upload dataset (.xlsx)",
        type=["xlsx"],
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
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**⚙️ Model Settings**")
    n_clusters_manual = st.slider("Override K (0 = auto)", 0, 10, 0)
    st.divider()
    st.caption("Built with Streamlit · sklearn · seaborn")


# ─────────────────────────────────────────────────────────────────
# GATE — require data upload
# ─────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("## 🌍 Global Development Clustering Dashboard")
    st.info(
        "👈  **Upload your dataset** in the sidebar to get started.\n\n"
        "Expected file: `P659_World_development_dataset.xlsx`",
        icon="📂",
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
df = load_and_preprocess(uploaded.read())
X_scaled, feat_cols = get_features(df)
X_pca, evr = get_pca(X_scaled)

# Auto-detect best K via silhouette
@st.cache_data(show_spinner=False)
def compute_elbow(X_scaled_bytes):
    """Cached elbow / silhouette computation."""
    X = np.frombuffer(X_scaled_bytes, dtype=np.float64).reshape(-1, X_scaled.shape[1])
    k_range = range(2, 11)
    inertia, silh = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertia.append(km.inertia_)
        silh.append(silhouette_score(X, labels))
    return list(k_range), inertia, silh


k_range, inertia_vals, silh_vals = compute_elbow(X_scaled.tobytes())
auto_k = k_range[silh_vals.index(max(silh_vals))]
best_k = n_clusters_manual if n_clusters_manual >= 2 else auto_k


# ═════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & EDA
# ═════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.markdown("## 📊 Dataset Overview & Exploratory Data Analysis")
    styled_fig()

    # KPI row
    num_countries = len(df["Country"].unique()) if "Country" in df.columns else len(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌐 Countries", num_countries)
    c2.metric("📐 Features (raw)", df.shape[1])
    c3.metric("📐 Clustering Features", len(feat_cols))
    c4.metric("🏅 Best K (auto)", auto_k)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📉 Distributions", "🔥 Correlation"])

    with tab1:
        st.markdown('<div class="section-header">📋 Preprocessed Dataset</div>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(30), use_container_width=True, height=380)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Shape:**")
            st.code(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
        with col2:
            st.markdown("**Dtypes:**")
            st.dataframe(df.dtypes.rename("dtype").reset_index().rename(
                columns={"index": "Column"}), use_container_width=True, height=200)

    with tab2:
        st.markdown('<div class="section-header">📉 Feature Distributions (Histograms)</div>',
                    unsafe_allow_html=True)
        num_df = df.select_dtypes(include="number")
        ncols = 4
        nrows = int(np.ceil(len(num_df.columns) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
        fig.patch.set_facecolor("#0e1117")
        for ax, col in zip(axes.flatten(), num_df.columns):
            ax.hist(num_df[col].dropna(), bins=25, color="#3b82f6", edgecolor="#0e1117", alpha=0.85)
            ax.set_title(col, fontsize=8, color="#c8d8ef")
            ax.tick_params(labelsize=7)
        for ax in axes.flatten()[len(num_df.columns):]:
            ax.set_visible(False)
        plt.suptitle("Histograms of Numerical Features", fontsize=13,
                     color="white", fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Skewness bar
        st.markdown('<div class="section-header">📐 Skewness of Features</div>',
                    unsafe_allow_html=True)
        skew = num_df.skew().sort_values()
        fig2, ax2 = plt.subplots(figsize=(10, max(3, len(skew) * 0.3)))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#161b27")
        colors = ["#ef4444" if v > 1 else "#3b82f6" if v < -1 else "#22c55e"
                  for v in skew.values]
        ax2.barh(skew.index, skew.values, color=colors, edgecolor="#0e1117")
        ax2.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax2.set_title("Skewness (|>1| = highly skewed)", color="white", fontsize=11)
        ax2.tick_params(colors="#c8d8ef", labelsize=8)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    with tab3:
        st.markdown('<div class="section-header">🔥 Correlation Heatmap</div>',
                    unsafe_allow_html=True)
        corr = df.select_dtypes(include="number").corr()
        fig3, ax3 = plt.subplots(figsize=(14, 10))
        fig3.patch.set_facecolor("#0e1117")
        ax3.set_facecolor("#0e1117")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.4,
                    cmap="coolwarm", ax=ax3, annot_kws={"size": 7},
                    cbar_kws={"shrink": 0.7})
        ax3.set_title("Feature Correlation Matrix", color="white", fontsize=13,
                      fontweight="bold")
        ax3.tick_params(colors="#c8d8ef", labelsize=8)
        st.pyplot(fig3, use_container_width=True)
        plt.close()


# ═════════════════════════════════════════════════════════════════
# PAGE 2 — FEATURE ANALYSIS
# ═════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Analysis":
    st.markdown("## 🔍 Feature Analysis & Outlier Detection")
    styled_fig()

    tab1, tab2, tab3 = st.tabs(["📦 Boxplots", "🔵 Scatter Plots", "📊 Descriptive Stats"])

    with tab1:
        st.markdown('<div class="section-header">📦 Boxplots (After Outlier Capping)</div>',
                    unsafe_allow_html=True)
        ncols = 4
        num_df = df[feat_cols]
        nrows = int(np.ceil(len(num_df.columns) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
        fig.patch.set_facecolor("#0e1117")
        for ax, col in zip(axes.flatten(), num_df.columns):
            ax.boxplot(num_df[col].dropna(), patch_artist=True,
                       boxprops=dict(facecolor="#3b82f6", color="#93c5fd"),
                       medianprops=dict(color="white", linewidth=2),
                       whiskerprops=dict(color="#60a5fa"),
                       capprops=dict(color="#60a5fa"),
                       flierprops=dict(markerfacecolor="#ef4444", markersize=4))
            ax.set_title(col, fontsize=8, color="#c8d8ef")
            ax.tick_params(labelsize=7)
        for ax in axes.flatten()[len(num_df.columns):]:
            ax.set_visible(False)
        plt.suptitle("Boxplots After Outlier Capping (IQR Method)",
                     fontsize=13, color="white", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        st.markdown('<div class="section-header">🔵 Interactive Scatter Plot</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X-axis", feat_cols, index=feat_cols.index("Internet Usage")
                              if "Internet Usage" in feat_cols else 0)
        y_col = c2.selectbox("Y-axis", feat_cols, index=feat_cols.index("Life Expectancy Male")
                              if "Life Expectancy Male" in feat_cols else 1)

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        fig4.patch.set_facecolor("#0e1117")
        ax4.set_facecolor("#161b27")
        ax4.scatter(df[x_col], df[y_col], c="#3b82f6", alpha=0.65, s=40, edgecolors="#60a5fa", linewidths=0.3)
        ax4.set_xlabel(x_col, color="#c8d8ef")
        ax4.set_ylabel(y_col, color="#c8d8ef")
        ax4.set_title(f"{x_col} vs {y_col}", color="white", fontsize=12, fontweight="bold")
        ax4.grid(alpha=0.2)
        ax4.tick_params(colors="#8a9bbf")
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    with tab3:
        st.markdown('<div class="section-header">📊 Descriptive Statistics</div>',
                    unsafe_allow_html=True)
        st.dataframe(df[feat_cols].describe().T.round(3), use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 3 — OPTIMAL K
# ═════════════════════════════════════════════════════════════════
elif page == "📈 Optimal K Selection":
    st.markdown("## 📈 Optimal Number of Clusters")
    styled_fig()

    st.markdown(
        f'<div class="insight-box">🏆  Auto-detected <strong>optimal K = {auto_k}</strong> '
        f'via highest Silhouette Score '
        f'(<strong>{max(silh_vals):.4f}</strong>).</div>',
        unsafe_allow_html=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#161b27")
        ax.tick_params(colors="#8a9bbf")

    # Elbow
    axes[0].plot(k_range, inertia_vals, "bo-", linewidth=2.2, markersize=7,
                 color="#3b82f6", markerfacecolor="white")
    axes[0].axvline(auto_k, color="#ef4444", linewidth=1.5, linestyle="--", label=f"Optimal K={auto_k}")
    axes[0].set_xlabel("Number of Clusters (k)", color="#c8d8ef")
    axes[0].set_ylabel("Inertia (WCSS)", color="#c8d8ef")
    axes[0].set_title("Elbow Method", color="white", fontweight="bold")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    # Silhouette
    axes[1].plot(k_range, silh_vals, "o-", linewidth=2.2, markersize=7,
                 color="#22c55e", markerfacecolor="white")
    axes[1].axvline(auto_k, color="#ef4444", linewidth=1.5, linestyle="--", label=f"Optimal K={auto_k}")
    axes[1].set_xlabel("Number of Clusters (k)", color="#c8d8ef")
    axes[1].set_ylabel("Silhouette Score", color="#c8d8ef")
    axes[1].set_title("Silhouette Score vs k", color="white", fontweight="bold")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Table
    st.markdown("### 📋 K vs Metrics Table")
    k_df = pd.DataFrame({"K": k_range, "Inertia": inertia_vals,
                          "Silhouette": silh_vals})
    k_df["Optimal"] = k_df["K"].apply(lambda x: "✅ Best" if x == auto_k else "")
    st.dataframe(k_df.set_index("K"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 4 — CLUSTERING MODELS
# ═════════════════════════════════════════════════════════════════
elif page == "🤖 Clustering Models":
    st.markdown(f"## 🤖 Clustering Models  *(K = {best_k})*")
    styled_fig()

    model_tab = st.selectbox(
        "Select Model",
        ["K-Means", "Agglomerative", "DBSCAN", "Gaussian Mixture", "Mean Shift"],
    )

    def pca_scatter(ax, labels, title):
        unique = sorted(set(labels))
        palette = cm.get_cmap("tab10", len(unique))
        for i, lbl in enumerate(unique):
            mask = labels == lbl
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       s=45, alpha=0.75, color=palette(i),
                       label=f"Cluster {lbl}" if lbl != -1 else "Noise",
                       edgecolors="none")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", color="#c8d8ef")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", color="#c8d8ef")
        ax.tick_params(colors="#8a9bbf")
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(alpha=0.2)

    def bar_counts(ax, labels, title):
        unique, counts = np.unique(labels, return_counts=True)
        palette = cm.get_cmap("tab10", len(unique))
        ax.bar([str(u) for u in unique], counts,
               color=[palette(i) for i in range(len(unique))], edgecolor="#0e1117")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_xlabel("Cluster", color="#c8d8ef")
        ax.set_ylabel("Countries", color="#c8d8ef")
        ax.tick_params(colors="#8a9bbf")
        ax.grid(axis="y", alpha=0.2)

    if model_tab == "K-Means":
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        metrics = eval_metrics(X_scaled, labels)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters", best_k)
        c2.metric("Silhouette ↑", metrics["Silhouette"])
        c3.metric("Davies-Bouldin ↓", metrics["Davies-Bouldin"])
        c4.metric("Calinski-Harabasz ↑", metrics["Calinski-Harabasz"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#161b27")
        pca_scatter(axes[0], labels, f"K-Means (k={best_k}) — PCA")
        bar_counts(axes[1], labels, "Countries per Cluster")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        if "Country" in df.columns:
            st.markdown("### 🌐 Cluster Assignments")
            df_view = df[["Country"]].copy()
            df_view["Cluster"] = labels
            st.dataframe(df_view.sort_values("Cluster"), use_container_width=True, height=300)

    elif model_tab == "Agglomerative":
        agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
        labels = agg.fit_predict(X_scaled)
        metrics = eval_metrics(X_scaled, labels)

        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette ↑", metrics["Silhouette"])
        c2.metric("Davies-Bouldin ↓", metrics["Davies-Bouldin"])
        c3.metric("Calinski-Harabasz ↑", metrics["Calinski-Harabasz"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#161b27")

        # Dendrogram
        linked = linkage(X_scaled, method="ward")
        dendrogram(linked, ax=axes[0], truncate_mode="lastp", p=25,
                   leaf_rotation=90, color_threshold=0,
                   above_threshold_color="#3b82f6",
                   leaf_font_size=7)
        axes[0].set_title("Dendrogram (Ward)", color="white", fontweight="bold")
        axes[0].tick_params(colors="#8a9bbf")
        axes[0].set_facecolor("#161b27")

        pca_scatter(axes[1], labels, f"Agglomerative (k={best_k})")
        bar_counts(axes[2], labels, "Countries per Cluster")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    elif model_tab == "DBSCAN":
        c1, c2 = st.columns(2)
        eps = c1.slider("eps", 0.1, 5.0, 1.2, 0.1)
        min_samples = c2.slider("min_samples", 2, 20, 3)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        metrics = eval_metrics(X_scaled, labels)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters Found", n_clusters)
        c2.metric("Noise Points", n_noise)
        c3.metric("Silhouette ↑", metrics["Silhouette"] if metrics["Silhouette"] else "N/A")
        c4.metric("Davies-Bouldin ↓", metrics["Davies-Bouldin"] if metrics["Davies-Bouldin"] else "N/A")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#161b27")
        pca_scatter(axes[0], labels, f"DBSCAN (eps={eps}, min={min_samples})")
        bar_counts(axes[1], labels, "Cluster Sizes")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    elif model_tab == "Gaussian Mixture":
        gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=5)
        labels = gmm.fit_predict(X_scaled)
        metrics = eval_metrics(X_scaled, labels)

        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette ↑", metrics["Silhouette"])
        c2.metric("Davies-Bouldin ↓", metrics["Davies-Bouldin"])
        c3.metric("Calinski-Harabasz ↑", metrics["Calinski-Harabasz"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#161b27")
        pca_scatter(axes[0], labels, f"Gaussian Mixture (k={best_k})")
        bar_counts(axes[1], labels, "Countries per Component")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    elif model_tab == "Mean Shift":
        with st.spinner("Estimating bandwidth…"):
            bw = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=min(300, len(X_scaled)))
            ms = MeanShift(bandwidth=bw, bin_seeding=True)
            labels = ms.fit_predict(X_scaled)
        n_clusters = len(set(labels))
        metrics = eval_metrics(X_scaled, labels)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters Found", n_clusters)
        c2.metric("Silhouette ↑", metrics["Silhouette"])
        c3.metric("Davies-Bouldin ↓", metrics["Davies-Bouldin"])
        c4.metric("Calinski-Harabasz ↑", metrics["Calinski-Harabasz"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#161b27")
        pca_scatter(axes[0], labels, "Mean Shift Clustering")
        bar_counts(axes[1], labels, "Countries per Cluster")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════
elif page == "📋 Model Comparison":
    st.markdown(f"## 📋 Model Comparison  *(K = {best_k})*")
    styled_fig()

    with st.spinner("Running all models…"):
        results = []

        # K-Means
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        m = eval_metrics(X_scaled, lbl)
        results.append({"Model": "K-Means", "Clusters": best_k,
                        "Silhouette ↑": m["Silhouette"],
                        "Davies-Bouldin ↓": m["Davies-Bouldin"],
                        "Calinski-Harabasz ↑": m["Calinski-Harabasz"]})

        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
        lbl = agg.fit_predict(X_scaled)
        m = eval_metrics(X_scaled, lbl)
        results.append({"Model": "Agglomerative", "Clusters": best_k,
                        "Silhouette ↑": m["Silhouette"],
                        "Davies-Bouldin ↓": m["Davies-Bouldin"],
                        "Calinski-Harabasz ↑": m["Calinski-Harabasz"]})

        # GMM
        gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=5)
        lbl = gmm.fit_predict(X_scaled)
        m = eval_metrics(X_scaled, lbl)
        results.append({"Model": "Gaussian Mixture", "Clusters": best_k,
                        "Silhouette ↑": m["Silhouette"],
                        "Davies-Bouldin ↓": m["Davies-Bouldin"],
                        "Calinski-Harabasz ↑": m["Calinski-Harabasz"]})

        # DBSCAN
        db = DBSCAN(eps=1.2, min_samples=3)
        lbl = db.fit_predict(X_scaled)
        n_db = len(set(lbl)) - (1 if -1 in lbl else 0)
        m = eval_metrics(X_scaled, lbl)
        results.append({"Model": "DBSCAN", "Clusters": n_db,
                        "Silhouette ↑": m["Silhouette"],
                        "Davies-Bouldin ↓": m["Davies-Bouldin"],
                        "Calinski-Harabasz ↑": m["Calinski-Harabasz"]})

    comp_df = pd.DataFrame(results).set_index("Model")
    st.dataframe(
        comp_df.style.highlight_max(
            subset=["Silhouette ↑", "Calinski-Harabasz ↑"], color="#1a4a2e"
        ).highlight_min(
            subset=["Davies-Bouldin ↓"], color="#1a4a2e"
        ),
        use_container_width=True,
    )

    # Bar chart comparison
    st.markdown('<div class="section-header">📊 Silhouette Score Comparison</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b27")
    valid = comp_df["Silhouette ↑"].dropna()
    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
    ax.barh(valid.index, valid.values, color=colors[:len(valid)], edgecolor="#0e1117")
    ax.set_xlabel("Silhouette Score (higher = better)", color="#c8d8ef")
    ax.tick_params(colors="#c8d8ef")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("Silhouette Score per Model", color="white", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Best model callout
    if comp_df["Silhouette ↑"].dropna().any():
        best_model = comp_df["Silhouette ↑"].idxmax()
        best_score = comp_df["Silhouette ↑"].max()
        st.success(
            f"🏆 **Best Model: {best_model}** with Silhouette Score = **{best_score:.4f}**"
        )


# ═════════════════════════════════════════════════════════════════
# PAGE 6 — COUNTRY EXPLORER
# ═════════════════════════════════════════════════════════════════
elif page == "🗺️ Country Explorer":
    st.markdown("## 🗺️ Country Cluster Explorer")
    styled_fig()

    # Run KMeans
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(X_scaled)
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    if "Country" not in df.columns:
        st.warning("No 'Country' column found in the dataset.")
        st.stop()

    c1, c2 = st.columns([1, 2])
    with c1:
        selected_cluster = st.selectbox(
            "Filter by Cluster", ["All"] + list(range(best_k))
        )
        search = st.text_input("🔍 Search Country")

    view_df = df[["Country", "Cluster"] + [c for c in feat_cols if c in df.columns]].copy()
    if selected_cluster != "All":
        view_df = view_df[view_df["Cluster"] == selected_cluster]
    if search:
        view_df = view_df[view_df["Country"].str.contains(search, case=False, na=False)]

    with c2:
        st.metric("Countries Shown", len(view_df))

    st.dataframe(view_df.sort_values("Cluster"), use_container_width=True, height=380)

    # Cluster Profile
    st.markdown('<div class="section-header">📊 Cluster Profiles (Mean Feature Values)</div>',
                unsafe_allow_html=True)
    profile = df.groupby("Cluster")[feat_cols].mean().round(3)
    st.dataframe(profile, use_container_width=True)

    # Radar placeholder
    st.markdown('<div class="section-header">📡 PCA Scatter with Country Labels</div>',
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b27")
    palette = cm.get_cmap("tab10", best_k)
    for k_i in range(best_k):
        mask = df["Cluster"] == k_i
        ax.scatter(df.loc[mask, "PC1"], df.loc[mask, "PC2"],
                   s=55, alpha=0.8, color=palette(k_i), label=f"Cluster {k_i}",
                   edgecolors="none")
        for _, row in df.loc[mask].iterrows():
            ax.annotate(row["Country"], (row["PC1"], row["PC2"]),
                        fontsize=5, color="#8a9bbf", alpha=0.7,
                        xytext=(2, 2), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", color="#c8d8ef")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", color="#c8d8ef")
    ax.set_title("Countries on PCA Space", color="white", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(alpha=0.2)
    ax.tick_params(colors="#8a9bbf")
    st.pyplot(fig, use_container_width=True)
    plt.close()
