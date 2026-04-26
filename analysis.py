"""
Data Science Project — Breast Cancer Wisconsin (Diagnostic)
LNMIIT | B.Tech CSE | Even Semester 2025-26

Run this script to:
  1. Preprocess the dataset
  2. Perform statistical analysis
  3. Train SVM and kNN classifiers
  4. Run K-Means and DBSCAN clustering
  5. Save all plots to public/images/
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import entropy as scipy_entropy

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "public", "images")
os.makedirs(IMG_DIR, exist_ok=True)

def save(name):
    path = os.path.join(IMG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})
BLUE  = "#1976D2"
RED   = "#D32F2F"
GREEN = "#388E3C"
GRAY  = "#757575"

# =============================================================================
# 1. LOAD DATASET
# =============================================================================
print("\n=== 1. Loading Dataset ===")
raw = load_breast_cancer()
df  = pd.DataFrame(raw.data, columns=raw.feature_names)
df["target"]    = raw.target
df["diagnosis"] = df["target"].map({1: "Benign", 0: "Malignant"})

print(f"  Shape          : {df.shape}")
print(f"  Features       : {len(raw.feature_names)}")
print(f"  Benign (1)     : {(df.target == 1).sum()}")
print(f"  Malignant (0)  : {(df.target == 0).sum()}")
print(f"  Missing values : {df.isnull().sum().sum()}")

# =============================================================================
# 2. PREPROCESSING
# =============================================================================
print("\n=== 2. Preprocessing ===")
X = df[raw.feature_names].values
y = df["target"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train set : {X_train.shape}")
print(f"  Test set  : {X_test.shape}")

# =============================================================================
# 3. PRELIMINARY PLOTS
# =============================================================================
print("\n=== 3. Generating Preliminary Plots ===")

# ── 3.1 Class Distribution Pie Chart ────────────────────────────────────────
counts = df["diagnosis"].value_counts()
fig, ax = plt.subplots(figsize=(7, 6))
wedges, texts, autotexts = ax.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    colors=[BLUE, RED],
    startangle=90,
    pctdistance=0.82,
    wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2),
)
for t in texts:
    t.set_fontsize(13)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight("bold")
ax.set_title("Class Distribution: Benign vs Malignant", fontsize=15, fontweight="bold", pad=18)
centre_circle = plt.Circle((0, 0), 0.40, fc="white")
ax.add_patch(centre_circle)
save("class_distribution_pie.png")

# ── 3.2 Feature Distributions (first 10 features) ───────────────────────────
top10 = raw.feature_names[:10]
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes = axes.flatten()
for i, feat in enumerate(top10):
    for label, color in zip(["Benign", "Malignant"], [BLUE, RED]):
        axes[i].hist(
            df[df["diagnosis"] == label][feat],
            bins=22, alpha=0.65, label=label, color=color, edgecolor="white",
        )
    axes[i].set_title(feat, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Value", fontsize=8)
    axes[i].set_ylabel("Count", fontsize=8)
    axes[i].legend(fontsize=7)
fig.suptitle("Feature Distributions by Diagnosis (first 10 features)",
             fontsize=16, fontweight="bold")
plt.tight_layout()
save("feature_distributions.png")

# ── 3.3 Correlation Heatmap ──────────────────────────────────────────────────
corr = df[raw.feature_names].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(19, 16))
sns.heatmap(
    corr, mask=mask, cmap="RdYlBu_r", center=0,
    square=True, linewidths=0.3, annot=False, ax=ax,
    cbar_kws={"shrink": 0.75, "label": "Pearson r"},
)
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
plt.tight_layout()
save("correlation_heatmap.png")

# ── 3.4 Box Plots for 6 Key Features ─────────────────────────────────────────
key6 = ["mean radius", "mean texture", "mean perimeter",
        "mean area", "mean smoothness", "mean concavity"]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, feat in enumerate(key6):
    bp = axes[i].boxplot(
        [df[df["diagnosis"] == "Benign"][feat].values,
         df[df["diagnosis"] == "Malignant"][feat].values],
        labels=["Benign", "Malignant"],
        patch_artist=True, notch=True, widths=0.4,
        medianprops=dict(color="black", linewidth=2),
    )
    bp["boxes"][0].set_facecolor(BLUE);  bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(RED);   bp["boxes"][1].set_alpha(0.75)
    axes[i].set_title(feat.title(), fontsize=12, fontweight="bold")
    axes[i].set_ylabel("Value", fontsize=10)
fig.suptitle("Box Plots of Key Features by Diagnosis", fontsize=16, fontweight="bold")
plt.tight_layout()
save("boxplot_features.png")

# ── 3.5 Mean Feature Comparison Bar Chart ────────────────────────────────────
mean_vals = df.groupby("diagnosis")[key6].mean()
fig, ax = plt.subplots(figsize=(13, 6))
x     = np.arange(len(key6))
width = 0.35
b1 = ax.bar(x - width/2, mean_vals.loc["Benign"],    width, label="Benign",    color=BLUE, alpha=0.85, edgecolor="white")
b2 = ax.bar(x + width/2, mean_vals.loc["Malignant"], width, label="Malignant", color=RED,  alpha=0.85, edgecolor="white")
ax.set_title("Mean Feature Values by Diagnosis", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f.replace("mean ", "").title() for f in key6], rotation=15, ha="right")
ax.set_ylabel("Mean Value")
ax.legend(fontsize=12)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
save("mean_feature_comparison.png")

# ── 3.6 Violin Plots ─────────────────────────────────────────────────────────
viol_feats = ["mean radius", "mean area", "mean concavity"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, feat in zip(axes, viol_feats):
    parts = ax.violinplot(
        [df[df["diagnosis"] == "Benign"][feat].values,
         df[df["diagnosis"] == "Malignant"][feat].values],
        positions=[1, 2], showmedians=True, showextrema=True,
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor([BLUE, RED][idx])
        body.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Benign", "Malignant"], fontsize=11)
    ax.set_title(feat.title(), fontsize=12, fontweight="bold")
    ax.set_ylabel("Value")
fig.suptitle("Violin Plots of Key Features by Diagnosis", fontsize=16, fontweight="bold")
plt.tight_layout()
save("violin_plots.png")

# =============================================================================
# 4. CLASSIFICATION
# =============================================================================
print("\n=== 4. ML Classification ===")

# ── SVM (RBF) ────────────────────────────────────────────────────────────────
svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm  = svm.predict(X_test)
y_prob_svm  = svm.predict_proba(X_test)[:, 1]
svm_acc     = accuracy_score(y_test, y_pred_svm)
svm_kappa   = cohen_kappa_score(y_test, y_pred_svm)
svm_auc     = roc_auc_score(y_test, y_prob_svm)
svm_cv      = cross_val_score(svm, X_scaled, y, cv=StratifiedKFold(5), scoring="accuracy").mean()

print(f"  SVM  → Acc: {svm_acc:.4f}  Kappa: {svm_kappa:.4f}  AUC: {svm_auc:.4f}  CV-Acc: {svm_cv:.4f}")
print(f"\n{classification_report(y_test, y_pred_svm, target_names=['Malignant','Benign'])}")

# ── kNN (k=5) ────────────────────────────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_train, y_train)
y_pred_knn  = knn.predict(X_test)
y_prob_knn  = knn.predict_proba(X_test)[:, 1]
knn_acc     = accuracy_score(y_test, y_pred_knn)
knn_kappa   = cohen_kappa_score(y_test, y_pred_knn)
knn_auc     = roc_auc_score(y_test, y_prob_knn)
knn_cv      = cross_val_score(knn, X_scaled, y, cv=StratifiedKFold(5), scoring="accuracy").mean()

print(f"  kNN  → Acc: {knn_acc:.4f}  Kappa: {knn_kappa:.4f}  AUC: {knn_auc:.4f}  CV-Acc: {knn_cv:.4f}")
print(f"\n{classification_report(y_test, y_pred_knn, target_names=['Malignant','Benign'])}")

# ── 4.1 Confusion Matrices ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_pred, title in zip(
        axes,
        [y_pred_svm, y_pred_knn],
        ["SVM (RBF Kernel)", "k-Nearest Neighbours (k = 5)"]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        annot_kws={"size": 16, "weight": "bold"},
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")
plt.tight_layout()
save("confusion_matrices.png")

# ── 4.2 ROC Curves ───────────────────────────────────────────────────────────
fpr_s, tpr_s, _ = roc_curve(y_test, y_prob_svm)
fpr_k, tpr_k, _ = roc_curve(y_test, y_prob_knn)
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(fpr_s, tpr_s, color=BLUE,  lw=2.5, label=f"SVM  (AUC = {svm_auc:.3f})")
ax.plot(fpr_k, tpr_k, color=RED,   lw=2.5, label=f"kNN  (AUC = {knn_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC = 0.500)")
ax.fill_between(fpr_s, tpr_s, alpha=0.08, color=BLUE)
ax.fill_between(fpr_k, tpr_k, alpha=0.08, color=RED)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curves: SVM vs kNN", fontsize=15, fontweight="bold")
ax.legend(loc="lower right", fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
save("roc_curves.png")

# ── 4.3 Classifier Comparison Bar Chart ──────────────────────────────────────
metrics      = ["Accuracy", "Cohen's Kappa", "ROC AUC"]
svm_scores   = [svm_acc, svm_kappa, svm_auc]
knn_scores   = [knn_acc, knn_kappa, knn_auc]
x_pos        = np.arange(len(metrics))
width        = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
b_svm = ax.bar(x_pos - width/2, svm_scores, width, label="SVM",  color=BLUE, alpha=0.85, edgecolor="white")
b_knn = ax.bar(x_pos + width/2, knn_scores, width, label="kNN",  color=RED,  alpha=0.85, edgecolor="white")
ax.set_xticks(x_pos); ax.set_xticklabels(metrics, fontsize=13)
ax.set_ylim(0, 1.18); ax.set_ylabel("Score", fontsize=12)
ax.set_title("Classifier Performance Comparison: SVM vs kNN", fontsize=15, fontweight="bold")
ax.legend(fontsize=12)
for bar in list(b_svm) + list(b_knn):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
save("classifier_comparison.png")

# =============================================================================
# 5. CLUSTERING
# =============================================================================
print("\n=== 5. Clustering ===")

# PCA for visualisation
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var   = pca.explained_variance_ratio_ * 100

# ── 5.1 Elbow Plot ───────────────────────────────────────────────────────────
sse_vals = []
k_range  = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sse_vals.append(km.inertia_)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(list(k_range), sse_vals, "o-", color=BLUE, lw=2.5, markersize=8)
ax.fill_between(list(k_range), sse_vals, alpha=0.12, color=BLUE)
ax.axvline(x=2, color=RED, linestyle="--", lw=2, label="Optimal k = 2")
ax.set_xlabel("Number of Clusters (k)", fontsize=13)
ax.set_ylabel("Sum of Squared Errors (SSE)", fontsize=13)
ax.set_title("K-Means Elbow Plot", fontsize=15, fontweight="bold")
ax.set_xticks(list(k_range))
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
save("elbow_plot.png")

# ── 5.2 K-Means (k=2) ────────────────────────────────────────────────────────
kmeans    = KMeans(n_clusters=2, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_scaled)
km_sse    = kmeans.inertia_

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels,
                cmap="coolwarm", alpha=0.75, s=55, edgecolors="white", linewidth=0.4)
centers = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="X",
           s=220, zorder=6, label="Centroids", edgecolors="white", linewidth=2)
ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)", fontsize=12)
ax.set_title("K-Means Clustering (k=2) — PCA Projection", fontsize=15, fontweight="bold")
plt.colorbar(sc, ax=ax, label="Cluster ID")
ax.legend(fontsize=11)
plt.tight_layout()
save("kmeans_clusters.png")

# ── 5.3 DBSCAN ───────────────────────────────────────────────────────────────
dbscan    = DBSCAN(eps=2.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise       = int(np.sum(db_labels == -1))

fig, ax = plt.subplots(figsize=(10, 8))
unique_labels = sorted(set(db_labels))
palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(unique_labels), 1)))
for lbl, color in zip(unique_labels, palette):
    if lbl == -1:
        ax.scatter(X_pca[db_labels == -1, 0], X_pca[db_labels == -1, 1],
                   c=[[0.6, 0.6, 0.6, 0.5]], marker="x", s=35, label="Noise")
    else:
        ax.scatter(X_pca[db_labels == lbl, 0], X_pca[db_labels == lbl, 1],
                   color=color, s=55, alpha=0.75, edgecolors="white",
                   linewidth=0.4, label=f"Cluster {lbl}")
ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)", fontsize=12)
ax.set_title(
    f"DBSCAN Clustering (eps=2.5, min\\_samples=5) — PCA Projection\n"
    f"{n_clusters_db} cluster(s), {n_noise} noise points",
    fontsize=14, fontweight="bold",
)
ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
save("dbscan_clusters.png")

# ── 5.4 Entropy & SSE ────────────────────────────────────────────────────────
def cluster_entropy(true_labels, cluster_labels):
    total = 0.0
    for c in np.unique(cluster_labels):
        if c == -1:
            continue
        mask  = cluster_labels == c
        probs = np.bincount(true_labels[mask], minlength=2) / mask.sum()
        probs = probs[probs > 0]
        weight = mask.sum() / len(true_labels)
        total += weight * scipy_entropy(probs, base=2)
    return total

km_entropy = cluster_entropy(y, km_labels)
db_entropy = cluster_entropy(y, db_labels)

# SSE for DBSCAN (non-noise points)
db_sse = 0.0
for c in set(db_labels):
    if c == -1:
        continue
    mask   = db_labels == c
    center = X_scaled[mask].mean(axis=0)
    db_sse += float(np.sum((X_scaled[mask] - center) ** 2))

print(f"  K-Means → SSE: {km_sse:.2f}  Entropy: {km_entropy:.4f}")
print(f"  DBSCAN  → SSE: {db_sse:.2f}  Entropy: {db_entropy:.4f}  Clusters: {n_clusters_db}  Noise: {n_noise}")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
algos = ["K-Means", "DBSCAN"]

b1 = axes[0].bar(algos, [km_entropy, db_entropy], color=[BLUE, RED],
                  width=0.4, alpha=0.85, edgecolor="white")
axes[0].set_title("Entropy Comparison", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Entropy (bits)")
axes[0].set_ylim(0, max(km_entropy, db_entropy) * 1.35)
for bar, val in zip(b1, [km_entropy, db_entropy]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

b2 = axes[1].bar(algos, [km_sse, db_sse], color=[BLUE, RED],
                  width=0.4, alpha=0.85, edgecolor="white")
axes[1].set_title("SSE Comparison", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Sum of Squared Errors")
axes[1].set_ylim(0, max(km_sse, db_sse) * 1.2)
for bar, val in zip(b2, [km_sse, db_sse]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(km_sse, db_sse) * 0.01,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

fig.suptitle("Clustering Algorithm Comparison: K-Means vs DBSCAN",
             fontsize=15, fontweight="bold")
plt.tight_layout()
save("clustering_comparison.png")

# =============================================================================
# 6. PRINT FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY  (copy values into report.tex)")
print("=" * 60)
print(f"Dataset       : Breast Cancer Wisconsin (Diagnostic)")
print(f"Samples       : {df.shape[0]}   Features: {len(raw.feature_names)}")
print(f"Benign        : {counts['Benign']}  ({counts['Benign']/len(df)*100:.1f}%)")
print(f"Malignant     : {counts['Malignant']}  ({counts['Malignant']/len(df)*100:.1f}%)")
print(f"Train/Test    : {X_train.shape[0]} / {X_test.shape[0]}")
print()
print(f"SVM  Accuracy : {svm_acc:.4f}")
print(f"SVM  Kappa    : {svm_kappa:.4f}")
print(f"SVM  AUC      : {svm_auc:.4f}")
print(f"SVM  5-CV Acc : {svm_cv:.4f}")
print()
print(f"kNN  Accuracy : {knn_acc:.4f}")
print(f"kNN  Kappa    : {knn_kappa:.4f}")
print(f"kNN  AUC      : {knn_auc:.4f}")
print(f"kNN  5-CV Acc : {knn_cv:.4f}")
print()
print(f"K-Means SSE   : {km_sse:.2f}")
print(f"K-Means Entr. : {km_entropy:.4f}")
print(f"DBSCAN  SSE   : {db_sse:.2f}")
print(f"DBSCAN  Entr. : {db_entropy:.4f}")
print(f"DBSCAN  Clust : {n_clusters_db}")
print(f"DBSCAN  Noise : {n_noise}")
print("=" * 60)
print(f"\nAll {len(os.listdir(IMG_DIR))} plots saved to: {IMG_DIR}")
