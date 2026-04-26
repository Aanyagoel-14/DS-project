# Implementation Plan — Data Science Project
**LNMIIT | B.Tech CSE | Data Science | Even Semester 2025-26**

---

## Dataset
**Breast Cancer Wisconsin (Diagnostic)**
- Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
- Loaded via `sklearn.datasets.load_breast_cancer()` (mirrors UCI exactly)
- 569 instances, 30 numerical features, 2 classes (Malignant / Benign)
- No missing values; entirely numerical — ideal for all three project tasks

---

## File Structure

```
DS/
├── Plan.md                  ← this file
├── analysis.py              ← all Python code (preprocessing, ML, clustering, plotting)
├── report.tex               ← LaTeX report source
├── report.pdf               ← compiled PDF (run: pdflatex report.tex)
└── public/
    └── images/
        ├── class_distribution_pie.png
        ├── feature_distributions.png
        ├── correlation_heatmap.png
        ├── boxplot_features.png
        ├── mean_feature_comparison.png
        ├── violin_plots.png
        ├── confusion_matrices.png
        ├── roc_curves.png
        ├── classifier_comparison.png
        ├── elbow_plot.png
        ├── kmeans_clusters.png
        ├── dbscan_clusters.png
        └── clustering_comparison.png
```

---

## Task Breakdown

### A. Data Preprocessing & Preliminary Analysis
1. Load dataset, display shape, dtypes, class counts
2. Check for missing values (none expected in this dataset)
3. Standardise features using `StandardScaler` (zero mean, unit variance)
4. Train/test split: 80% train / 20% test, stratified
5. Statistical analysis: mean, std, min, max, quartiles (`describe()`)
6. **Plots:** class pie chart, feature histograms, correlation heatmap,
   box plots, mean-feature bar chart, violin plots

### B. ML Classification — SVM & kNN
1. **SVM:** RBF kernel, C=1.0, probability=True; fit on scaled train set
2. **kNN:** k=5 (standard default, verified via cross-validation); fit on scaled train set
3. Evaluate on test set:
   - Accuracy (`accuracy_score`)
   - Cohen's Kappa (`cohen_kappa_score`)
   - ROC AUC (`roc_auc_score`)
4. **Plots:** side-by-side confusion matrices, ROC curves (both models),
   bar chart comparing all three metrics

### C. Clustering — K-Means & DBSCAN
1. **K-Means:** run for k=2..10, record SSE; choose k=2 (matches ground truth classes)
2. **DBSCAN:** eps=2.5, min_samples=5 (tuned on scaled data)
3. PCA (2 components) for 2-D cluster visualisation
4. Compute **Entropy** and **SSE** for both algorithms
5. **Plots:** elbow plot, K-Means PCA scatter, DBSCAN PCA scatter,
   entropy & SSE comparison bar charts

---

## Report Structure (report.tex — max 20 pages)

| Section | Content |
|---------|---------|
| Title page | Project title, team members, roll numbers, date |
| Abstract | Brief overview of dataset, methods, key results |
| 1. Introduction | Motivation, objectives, questions the analysis addresses |
| 2. Dataset Description | Source, features table, class distribution (pie chart) |
| 3. Data Preprocessing | What was done, why, how; before/after comparison |
| 4. Statistical Analysis | Descriptive stats, correlation, distribution plots |
| 5. ML Classification | SVM and kNN methodology, results, confusion matrices, ROC |
| 6. Clustering | K-Means and DBSCAN methodology, results, entropy, SSE |
| 7. Discussion | Inferences, questions raised and answered by analysis |
| 8. Conclusion | Summary of findings |
| References | Dataset citation, sklearn, algorithm references |

---

## How to Run

```bash
# 1. Install dependencies (if needed)
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# 2. Generate all plots and print metrics
cd /Users/aanyagoel/Desktop/DS
python analysis.py

# 3. Compile the LaTeX report
pdflatex report.tex
pdflatex report.tex   # run twice for correct cross-references
```

---

## Team Members
*(fill in before submission)*

| Name | Roll Number |
|------|-------------|
| Member 1 | 22UXXXXX |
| Member 2 | 22UXXXXX |
| Member 3 | 22UXXXXX |
| Member 4 | 22UXXXXX |
