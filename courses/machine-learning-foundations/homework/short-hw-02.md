---
layout: jupyternotebook
title: Machine Learning Foundations – Project 02
course_home: /courses/machine-learning-foundations/
nav_section: homework
nav_order: 2
---

# Project 2: Comparing Instance-Based and Cluster-Based Learning


## *When do "neighbors" and "clusters" agree—and when do they disagree?*

Real-world data rarely tells you what the “right” categories are — sometimes we have **labels** (so we can train models), and sometimes we don’t.  
This project explores two distance-based algorithms that use very different logic to understand structure in data:

- **k-Nearest Neighbors (k-NN):** A **supervised** method that classifies new points based on how close they are to labeled examples.  
- **k-Means Clustering:** An **unsupervised** method that finds natural groups in unlabeled data by minimizing within-cluster variance.

Even though both rely on **distance**, their behavior and results can differ dramatically — especially when data overlap, scale, or class balance changes.  
Here, you’ll investigate both approaches side-by-side, visualize how they divide data, and analyze when they agree or disagree.

---

## Learning Objectives

By the end of this project, you will be able to:
- Apply **k-NN** and **k-Means** to a dataset of your choice.  
- Interpret **accuracy**, **inertia**, and **silhouette score** to assess model performance.  
- Visualize **decision boundaries** and **cluster regions** in 2D using PCA.  
- Reflect on how scaling, number of clusters/neighbors, and feature choice affect outcomes.  

---

## Instructions

1. **Choose a dataset**: Pick one dataset from the list below — all are built into `scikit-learn` or `seaborn`, so you don’t need to download anything.

   Use one of the following datasets:

   - **Penguins (multiclass, 3 classes)**  
     **Target:** species  
     ```python
     from sklearn.datasets import load_penguins
     data_original = load_penguins(as_frame=True)
     ```

   - **Wine (multiclass, 3 classes) — slightly harder**  
     **Target:** wine cultivar  
     ```python
     from sklearn.datasets import load_wine
     data_original = load_wine(as_frame=True)
     ```

   - **Breast Cancer Wisconsin (binary) — interpretable, imbalanced-ish**  
     **Target:** diagnosis (0 = malignant, 1 = benign)  
     ```python
     from sklearn.datasets import load_breast_cancer
     data_original = load_breast_cancer(as_frame=True)
     ```

   - **Digits (multiclass, 10 classes)**  
     **Target:** digit label (0–9)  
     ```python
     from sklearn.datasets import load_digits
     data_original = load_digits(as_frame=True)
     ```




2. **Update and run the code**: Use the one-cell Iris example below as your template.

   **Your goal**: 
   Modify it so that it correctly loads, processes, and analyzes your chosen dataset.

   Add comments to each block (indicated in the code below) to explain what is happening.

   **You’ll need to**:
   - Update the dataset loading section (X and y variables).
   - Check for categorical features and encode them if necessary.
   - Adjust n_clusters (for k-Means) and n_neighbors (for k-NN) as appropriate.
   - Re-run the code and confirm that all plots and metrics execute cleanly.

3. **Analyze and Report**: Write a short report (including key plots where appropriate!) interpreting the results of your analysis and what they reveal about the dataset you chose to study
    Your report should read in a **semi-professional** tone, similar to a technical summary you might provide to a customer who asked you to build this model. The goal is to clearly explain:
    - what question you were trying to answer,
    - what the model is doing (i.e. how does k-NN work, but explained to a non-professional),
    - what you found in the results/was anything odd about your results, and
    - how confident someone should be in the model’s conclusions.

    Focus on clarity and interpretation rather than technical jargon, and use plots to support your explanations when they add insight.
    
    Within the same document, but after your short report, answer the following questions:
    - How similar were the k-NN and k-Means results?
    - What effect did scaling or changing k have?
    - Did PCA make clusters more or less distinct?
    - If this were real-world data, what kind of insights could clustering reveal?
    This can be informal. I just want you to answer the questions and relfect on what you learned and what you stuggled with.

4. **(Optional Challenge)**: 
    - Get this to work for the Titanic data
        | Dataset | Type | Target Variable | Description |
        |----------|------|------------------|--------------|
        | **Titanic** | Binary | `survived` | Real-world dataset from Seaborn (requires encoding). |
        
        ```python
        import seaborn as sns  
        data_original = sns.load_dataset("titanic").dropna(subset=['survived'])
        ```

    - Try different distance metrics (Euclidean, Manhattan, Minkowski) for k-NN.
    - Compare inertia vs. silhouette score across different k values.





### What to Submit

1. **A clean, runnable Jupyter notebook** that:
    - [ ] loads your chosen dataset
    - [ ] prepares features and target variables (encoding if needed),  
    - [ ] performs a train/test split (`stratify=y` if classification),  
    - [ ] scales features appropriately,  
    - [ ] implements both **k-Nearest Neighbors (k-NN)** and **k-Means** models,  
    - [ ] evaluates k-NN with accuracy, classification report, and confusion matrix,  
    - [ ] evaluates k-Means with inertia, silhouette score, and visualizations (e.g., PCA or scatter plots),  
    - [ ] includes at least one model-comparison bar plot (e.g., k-NN vs. different k-values, or cluster count comparison),  
    - [ ] contains clear **section headers** for each major step (e.g., “Data Preparation,” “Modeling,” “Evaluation”),  
    - [ ] includes **concise, meaningful code comments** describing what each *block of code* does, and  
    - [ ] runs from start to finish **without errors**.

2. **A short written report (2–4 pages, PDF format)** that explains your project:
See What you should include in Setp 3 above. 

For further consideration, you should think about, and possibly add to your report or your reflection section (wink wink), comments about:
    - [ ] What research question or goal did you explore using your dataset?
        - I.e. What could a customer been asking for when they hired you and gave you the data?  
    - [ ] How did you prepare and analyze the data?  
        - I.e. whole row deletion if an missing data, etc.
    - [ ] What did your results show? Which model(s) performed best?  
    - [ ] Were there any unexpected findings or challenges?  
        - Is there anything you need to make you customer aware of?
    - [ ] Discuss at least one potential **source of bias** or limitation in your dataset or model.  
    - [ ] Reflect on how your workflow could be improved with more time or data.  

> **Reminder:** Your notebook should demonstrate a clear, step-by-step workflow, not just working code.  
> Use comments, titles, and plots to tell the *story* of your analysis.


## The Code

The cell below contains a complete Iris example demonstrating both k-NN and k-Means — including scaling, PCA visualization, accuracy metrics, inertia, silhouette score, and clean plots.

Use it as a template to guide your own work.


{% capture ex %}
```python
# ================================================
# One-Cell: k-NN & k-Means on the Iris Dataset
# ================================================

# --- Imports ---

# --- Core Data Science Libraries ---
import pandas as pd                     # Data manipulation & table handling (like Excel in Python)
import numpy as np                      # Numerical operations & efficient table/array computations
from scipy.stats import mode

# --- Visualization Libraries ---
import matplotlib.pyplot as plt          # Basic plotting (line plots, histograms, scatter plots)
import seaborn as sns                    # Statistical data visualization, built on top of matplotlib

# --- Scikit-learn: Core Machine Learning Toolkit ---
from sklearn.model_selection import train_test_split                # Splits data into training and testing sets
from sklearn.preprocessing import StandardScaler, LabelEncoder      # Normalize numeric data & encode categories
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,  # Model evaluation metrics
    silhouette_score
)

# --- Supervised Learning Models ---
from sklearn.neighbors import KNeighborsClassifier          # k-Nearest Neighbors (instance-based learning)
from sklearn.linear_model import LogisticRegression         # Logistic Regression (probabilistic classifier)
from sklearn.naive_bayes import GaussianNB                  # Naïve Bayes (probabilistic classifier)
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Decision Trees (rule-based learning)
from sklearn.ensemble import RandomForestClassifier         # Random Forests (ensemble of decision trees)

# --- Unsupervised Learning Models ---
from sklearn.cluster import KMeans                          # k-Means Clustering (unsupervised pattern finding)
from sklearn.decomposition import PCA                       # Principal Component Analysis (dimensionality reduction)

# --- Visualization Styling ---
sns.set(style="whitegrid", palette="muted", font_scale=1.1)  # Nice default theme for plots


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
# Add a comment explaining what the next section of code does
from sklearn.datasets import load_iris
data_original = load_iris(as_frame=True).frame
df = data_original  # includes both features and the 'target' column

# Add a comment explaining what the next section of code does
df = df.dropna()

# Add a comment explaining what the next section of code does
df_encoded = pd.get_dummies(df, drop_first=True)

# Add a comment explaining what the next section of code does
X = df_encoded.drop(columns = ['target'])
y = df_encoded['target']

# --------------------------------------------------------------------



# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Add a comment explaining what this section of code is doing. ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# --------------------------------------------------------------------



# --------------------------------------------------------------------
# --- Apply PCA for 2D visualization ---
# --------------------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
# --------------------------------------------------------------------



# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)
# --------------------------------------------------------------------



# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)



# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
inertia = kmeans.inertia_
silhouette = silhouette_score(X_train_scaled, clusters)

# --- Match cluster labels to true labels for "accuracy-like" comparison ---
cluster_labels = np.zeros_like(clusters)
for i in range(3):
    mask = (clusters == i)
    cluster_labels[mask] = mode(y_train[mask], keepdims=False).mode
cluster_acc = accuracy_score(y_train, cluster_labels)

# --- Create a mesh grid for decision boundaries ---
h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]
# --------------------------------------------------------------------



# --------------------------------------------------------------------
# --- Train models on PCA space for visualization ---
# --------------------------------------------------------------------
knn_pca = KNeighborsClassifier(n_neighbors=k).fit(X_pca, y_train)
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_pca)

# --- Predict grid points for both models ---
Z_knn = knn_pca.predict(grid_points).reshape(xx.shape)
Z_kmeans = kmeans_pca.predict(grid_points).reshape(xx.shape)

# --- Plot k-NN and k-Means Decision Regions ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# k-NN Decision Regions
axes[0].contourf(xx, yy, Z_knn, cmap="viridis", alpha=0.3)
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap="viridis", edgecolor="k", s=50)
axes[0].set_title(f"k-NN Decision Boundaries (k={k})")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")

# k-Means Cluster Regions
axes[1].contourf(xx, yy, Z_kmeans, cmap="viridis", alpha=0.3)
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", edgecolor="k", s=50)
axes[1].scatter(centers_pca[:, 0], centers_pca[:, 1],
                c="red", s=200, edgecolors="black", marker="X", label="Centroids")
axes[1].set_title("k-Means Cluster Boundaries (k=3)")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")
axes[1].legend()

# Comparison Bar Chart
bars = axes[2].bar(["k-NN Accuracy", "k-Means (matched)"], [acc_knn, cluster_acc],
                   color=["tab:blue", "tab:orange"], alpha=0.8)
axes[2].bar_label(bars, fmt="%.3f", padding=3)
axes[2].set_ylim(0, 1.05)
axes[2].set_title("Model Performance Comparison")
axes[2].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()

# --- Print Summary Metrics ---
print("=== k-NN Classification ===")
print(f"Accuracy (k={k}): {acc_knn:.3f}")
print(classification_report(y_test, y_pred_knn))

print("\n=== k-Means Clustering ===")
print(f"Inertia (within-cluster variance): {inertia:.2f}")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Approx. Accuracy (after label matching): {cluster_acc:.3f}")
```
{% endcapture %}
{% include codeinput.html content=ex %}
