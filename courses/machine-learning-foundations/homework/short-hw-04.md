---
layout: jupyternotebook
title: Machine Learning Foundations – Project 04
course_home: /courses/machine-learning-foundations/
nav_section: homework
nav_order: 4
---

# Project 4: Logistic Regression and Naive Bayes Calssification


## *Teaching computers how to draw a line.*

In this project, you will use Logistic Regression and Naïve Bayes to classify categorical outcomes.  

You’ll practice building and evaluating classifiers, interpreting model outputs (e.g., coefficients/odds ratios and class probabilities), and examining decision thresholds and assumptions that affect performance in real data.

---

## Learning Objectives

By the end of this project, you will be able to:

- **Frame a classification problem** — identify features `X` and a categorical target `y` (binary or multiclass).
- **Train and compare** Logistic Regression and Naïve Bayes variants:
  - Naïve Bayes (choose the right type for your data):
      - **Gaussian** (continuous),
      - **Multinomial** (counts/TF-IDF),
      - **Bernoulli** (binary features).
- **Interpret Logistic Regression coefficients** as **odds ratios** and reason about feature effects (direction & magnitude).
- **Evaluate classifiers** using appropriate metrics:
  - **Accuracy**, **Precision**, **Recall**, **F1-score**
  - **ROC curve** and **AUC**
  - **Confusion matrix** to diagnose error types
- **Work with probabilities & thresholds** — plot precision-recall/ROC, and **tune the decision threshold**.
- **Preprocess correctly**:
  - Scale numeric features.
  - Encode categorical features.
- **Articulate assumptions & limitations**:
  - Logistic: linear decision boundary in feature space, effect of regularization.
  - Naïve Bayes: **conditional independence** assumption and when it’s reasonable.

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

   - **Titanic (binary)**  
     **Target:** survived  
     ```python
     import seaborn as sns  
     data_original = sns.load_dataset("titanic").dropna(subset=['survived'])
     ```



2. **Update and run the code**: Use the Logistic Regression and Naive Bayes Modeling example below as your template.

**Your goal:**  
Modify it so that it correctly loads, processes, and analyzes your chosen dataset.

Add concise comments above each major code block (as indicated) explaining what that block does.

**You’ll need to:**  
- **Load your dataset** and clearly set **X (features)** and **y (target)**.
- Ensure the remaining code runs.
- (Optional) Include extra bells and whistles from lecture you think would add to your understanding of the model and helps build a better model.

3. **Analyze and Report**: Write a short report (including key plots where appropriate!) interpreting the results of your analysis and what they reveal about the dataset you chose to study
    Your report should read in a **semi-professional** tone, similar to a technical summary you might provide to a customer who asked you to build this model. The goal is to clearly explain:    
    - How well do the models perform on the test set (Accuracy, Precision, Recall, F1, ROC-AUC — use one-vs-rest for multiclass)?
    - Logistic Regression:
        - does a roughly linear decision boundary seem reasonable in feature space? Do any scaled features dominate?
        - which coefficients are largest (by absolute value)? Interpret sign and magnitude.
        - at what decision threshold do you achieve a useful balance between Precision and Recall for your task?
    - Naïve Bayes: 
        - is the conditional independence assumption plausible for your features (or clearly violated but still effective)?
        - which features are the most useful in distinguishing between the classes?
    - What does the confusion matrix reveal about common mistakes? Which classes are most often confused?
        - Does this make sense given potential class overlaps you may see in a PCA 2D plot?

    Your notebook should output:

    - **Model details & comparisons:**
    - [ ] Train a **Naïve Bayes** model appropriate to your features (Gaussian / Multinomial / Bernoulli).
    - [ ] Print a **classification report** for each model (with correct class names).

    - **Core metrics (test set):**
    - [ ] **Accuracy**
    - [ ] **Precision**, **Recall**, **F1-score** (per class)
    - [ ] **ROC-AUC** (binary; or one-vs-rest for multiclass)

    - **Diagnostics & interpretability:**
    - [ ] **Confusion matrix** (with labeled axes and classes)
    - [ ] **ROC curve** and (optionally) **Precision–Recall curve**
    - [ ] **Threshold analysis** for Logistic Regression (plot Precision, Recall, and F1 vs. threshold; report the threshold that maximizes F1 or meets a chosen precision/recall target)
    - [ ] **Top coefficients** for Logistic Regression (sorted in a logical manner; briefly interpret sign and effect)

    - **Preprocessing checks (brief):**
    - [ ] Confirm **scaling** of numeric features (especially for Logistic Regression).
    - [ ] Confirm **encoding** of categorical/text features.







### What to Submit

- A semi-professional report deailing your findings, written for a layperson.
    - Some details of what you could include are given above and must include are given below. 
- An indepth reflection about the project.
    - Suggestions on what you can reflect on are below.



1. **A clean, runnable Jupyter notebook** that:
    - [ ] loads your dataset,
    - [ ] prepares data (handle missing values, encode categoricals, scale numeric features as appropriate),
    - [ ] performs a train/test split (use `stratify=y` for classification),
    - [ ] trains Logistic Regression,
    - [ ] trains an appropriate Naïve Bayes model (Gaussian / Multinomial / Bernoulli),
    - [ ] prints a classification report (correct class names) for each model,
    - [ ] includes test-set metrics: Accuracy, Precision, Recall, F1 (per class) and ROC-AUC (one-versus-rest for multiclass),
    - [ ] shows a confusion matrix (labeled axes/classes),
    - [ ] plots ROC curve (and optionlly a Precision–Recall curve),
    - [ ] performs threshold analysis for Logistic Regression (plot Precision/Recall/F1 vs. threshold and note the chosen threshold),
    - [ ] displays top Logistic coefficients (sorted by absolute value) with a brief note on sign/effect,
    - [ ] uses clear block comments explaining each major step,
    - [ ] runs top-to-bottom without errors and sets a `random_state` where applicable for reproducibility.

2. **A short report (2–4 pages, PDF)** that:
    - [ ] states your question and briefly describes the dataset and target,
    - [ ] summarizes model performance on the test set (Accuracy, Precision/Recall/F1, ROC-AUC),
    - [ ] comments on Logistic Regression: linear decision boundary reasonableness; which features dominate after scaling,
    - [ ] comments on Naïve Bayes: whether the conditional independence assumption seems plausible,
    - [ ] interprets the confusion matrix for both models (common mistakes / confused classes),
    - [ ] reports the chosen decision threshold (and why) for Logistic Regression,
    - [ ] includes and references key figures (confusion matrix, ROC/PR curves, threshold plot, coefficient bar chart).


> **Reminder:** Your notebook should demonstrate a clear, step-by-step workflow, not just working code.  
> Use comments, titles, and plots to tell the *story* of your analysis.


## The Code

The cell below contains a complete example demonstrating both linear and multiple linear regression in action.

Use it as a template to guide your own work.

### Imports

{% capture ex %}
```python
# --- Imports ---

# --- Core Python / Math Tools ---
import numpy as np                    # numerical operations, arrays, distance computations
import pandas as pd                   # data handling and manipulation
from scipy.stats import norm

# --- Visualization Libraries ---
import matplotlib.pyplot as plt        # general plotting
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns                 # polished statistical plots
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# --- scikit-learn: Datasets ---
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, 
    load_digits, make_blobs, fetch_california_housing, 
    make_classification
)

# --- scikit-learn: Compose ---
from sklearn.compose import ColumnTransformer

# --- scikit-learn: Model Preparation ---
from sklearn.model_selection import train_test_split   # split data into train/test sets
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, # feature scaling & label encoding
    PolynomialFeatures, label_binarize,
    OneHotEncoder
)

# --- scikit-learn: Metrics ---
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, roc_auc_score, classification_report,
    mean_squared_error, r2_score,
    mean_absolute_error, mean_absolute_error
)

# --- scikit-learn: Dummy Classifier for baselines ---
from sklearn.dummy import DummyClassifier

# --- scikit-learn: Naive Bayes ---
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# --- scikit-learn: Algorithms ---
from sklearn.linear_model import ( 
    LinearRegression,   # Linear Regression Model
    LogisticRegression # Logistic Regression Model
)

# --- scikit-learn: Multiclass ---
from sklearn.multiclass import OneVsRestClassifier


# --- scikit-learn: Dimensionality Reduction ---
from sklearn.decomposition import PCA                    # reduce features for 2D visualization

# --- statsmodels: model refinement ---
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Visualization Utilities for Trees / Boundaries ---
from matplotlib.colors import ListedColormap             # color maps for decision boundaries

# --- Other Useful Tools ---
import warnings
warnings.filterwarnings("ignore")  # keep output clean for class demos

```
{% endcapture %}
{% include codeinput.html content=ex %}

### One-vs-Rest Code

{% capture ex %}
```python
# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------

# Add a comment explaining what the next section of code does
from sklearn.datasets import load_iris
data_original = load_iris(as_frame=True)

feature_names = data_original.feature_names
target_names = data_original.target_names
df = data_original.frame.copy()

## Select your target for one-versus-rest logistic analysis
target_num = 2  # choose 0=setosa, 1=versicolor, 2=virginica
                # YOU MAY NEED TO LOOK UP THE DATA SET YOUR ARE USING 
                # TO SEE WHAT THE NUMBERS IN `target` MEAN!


# Add a comment explaining what the next section of code does
df = df.dropna()

# Add a comment explaining what the next section of code does
df_encoded = pd.get_dummies(df, drop_first=True)


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
# Add a comment explaining what the next section of code does
X = df_encoded.drop(columns = ['target'])
y = df_encoded['target']
class_names = target_names[target_num]

# Convert to binary classification: one class vs. rest
y_binary = (y == target_num).astype(int)
use_labels = [f"non-{class_names}", class_names]


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
sns.pairplot(df_encoded,
             hue=y.name,
             vars=X.columns.tolist())
plt.suptitle("Feature Relationships", y=1.02)
plt.show()


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
# Add a comment explaining what the next section of code does
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, stratify=y_binary, random_state=42
)

# Add a comment explaining what the next section of code does
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000, random_state=42)
nb = GaussianNB()

models = {
    "LogReg": logreg,
    "Gaussian Naïve Bayes": nb
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
def summarize_model(name, model, X, y_true):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=use_labels, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    return cm, y_pred

cms = {}
y_pred_dict = {}
for name, model in models.items():
    cm, y_pred = summarize_model(name, model, X_test_scaled, y_test)
    cms[name] = cm
    y_pred_dict[name] = y_pred


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13,4))
for ax, (name, cm) in zip(axes, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=use_labels, yticklabels=use_labels, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
plt.figure(figsize=(6,5))
for name, model in models.items():
    probs = model.predict_proba(X_test_scaled)[:,1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"--",color="gray")
plt.title("ROC Curves (Binary Classification)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
def compute_threshold_metrics(model, X, y_true):
    """Compute precision, recall, and F1 as threshold varies."""
    probs = model.predict_proba(X)[:,1]
    thresholds = np.linspace(0, 1, 100)
    precision, recall, f1 = [], [], []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        precision.append(precision_score(y_true, y_pred, zero_division=0))
        recall.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))
    return thresholds, precision, recall, f1

thr_log, prec_log, rec_log, f1_log = compute_threshold_metrics(logreg, X_test_scaled, y_test)
thr_nb,  prec_nb,  rec_nb,  f1_nb  = compute_threshold_metrics(nb, X_test_scaled, y_test)

fig, axes = plt.subplots(1, 2, figsize=(10,4))
for ax, thr, prec, rec, f1, title in zip(
    axes, [thr_log, thr_nb], [prec_log, prec_nb], [rec_log, rec_nb], [f1_log, f1_nb],
    ["Precision, Recall, and F1 vs Threshold", "Precision, Recall, and F1 vs Threshold (Naïve Bayes)"]
):
    ax.plot(thr, prec, label="Precision", color="tab:blue")
    ax.plot(thr, rec, label="Recall", color="tab:orange")
    ax.plot(thr, f1, label="F1", color="tab:green")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------

# Add a comment explaining what is happening here
custom_threshold = 0.4  # try adjusting this to see effects

# Add a comment explaining what is happening here
logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]
y_pred_custom = (logreg_probs >= custom_threshold).astype(int)

# Add a comment explaining what is happening here
acc_custom = accuracy_score(y_test, y_pred_custom)
prec_custom = precision_score(y_test, y_pred_custom, zero_division=0)
rec_custom = recall_score(y_test, y_pred_custom, zero_division=0)
f1_custom = f1_score(y_test, y_pred_custom, zero_division=0)

print(f"\n=== Logistic Regression (Custom Threshold = {custom_threshold:.2f}) ===")
print(f"Accuracy:  {acc_custom:.3f}")
print(f"Precision: {prec_custom:.3f}")
print(f"Recall:    {rec_custom:.3f}")
print(f"F1-score:  {f1_custom:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom, target_names=use_labels, digits=3))

# Add a comment explaining what is happening here
cm_custom = confusion_matrix(y_test, y_pred_custom)
plt.figure(figsize=(4,4))
sns.heatmap(cm_custom, annot=True, fmt="d", cmap="Blues",
            xticklabels=use_labels, yticklabels=use_labels)
plt.title(f"Confusion Matrix (Threshold = {custom_threshold:.2f})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

### Full Multi-Feature Code


{% capture ex %}
```python
# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
# Add a comment explaining what the next section of code does
from sklearn.datasets import load_iris
data_original = load_iris(as_frame=True)

feature_names = data_original.feature_names
class_names = data_original.target_names
df = data_original.frame.copy()

# Add a comment explaining what the next section of code does
df = df.dropna()

# Add a comment explaining what the next section of code does
df_encoded = pd.get_dummies(df, drop_first=True)

# Add a comment explaining what the next section of code does
X = df_encoded.drop(columns = ['target'])
y = df_encoded['target']


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
# Add a comment explaining what the next section of code does
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Add a comment explaining what the next section of code does
preprocessor = StandardScaler()
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# For ROC AUC (multiclass OvR)
y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
logreg_multi = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42
)
nb_multi = GaussianNB()

models = {
    "Logistic Regression (Multinomial)": logreg_multi,
    "Gaussian Naïve Bayes": nb_multi
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
def summarize_model(name, model, X, y_true):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    return cm, y_pred

cms = {}
for name, model in models.items():
    cm, _ = summarize_model(name, model, X_test_scaled, y_test)
    cms[name] = cm


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10,4))
for ax, (name, cm) in zip(axes, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
from itertools import cycle

# For multiclass ROC, need one-vs-rest binarization
y_test_bin = label_binarize(y_test, classes=np.unique(y))

plt.figure(figsize=(7,6))
colors = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])
auc_scores = {}

for (name, model), color in zip(models.items(), colors):
    probs = model.predict_proba(X_test_scaled)
    auc = roc_auc_score(y_test_bin, probs, multi_class="ovr")
    auc_scores[name] = auc

    # Plot ROC curve for each class faintly
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, lw=1, alpha=0.3, color=color)

    # Plot average model ROC (macro-mean of all classes)
    fpr_all, tpr_all, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
    plt.plot(fpr_all, tpr_all, lw=2.2, color=color,
             label=f"{name} (OvR AUC={auc:.3f})")

# Reference line
plt.plot([0,1],[0,1],"--",color="gray")
plt.title("ROC Curves (One-vs-Rest, Multiclass)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=9, frameon=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Print AUC summary
# ----------------------------------------------------------
print("\n=== One-vs-Rest AUC Scores ===")
for name, auc in auc_scores.items():
    print(f"{name:30s} → AUC: {auc:.3f}")


# ----------------------------------------------------------
# Feature Importance Visualization (Multiclass)
# ----------------------------------------------------------
# Get feature names (update if you have categorical encoding in your pipeline)
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]

# ----------------------------------------------------------
# Logistic Regression: Feature Importance
# ----------------------------------------------------------
if hasattr(logreg_multi, "coef_"):
    coef_df = pd.DataFrame(
        logreg_multi.coef_.T,
        columns=[class_names[c] for c in np.unique(y)],
        index=feature_names
    )

    # Mean absolute importance across classes
    coef_df["Mean|Coef|"] = coef_df.abs().mean(axis=1)
    coef_df_sorted = coef_df.sort_values("Mean|Coef|", ascending=False)

    # --- Top 10 most influential features ---
    plt.figure(figsize=(8,5))
    sns.barplot(
        x="Mean|Coef|",
        y=coef_df_sorted.index[:10],
        data=coef_df_sorted.head(10),
        palette="Blues_r"
    )
    plt.title("Top 10 Most Influential Features (Logistic Regression)")
    plt.xlabel("Mean |Coefficient| across Classes")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    print("\n")

    # --- Per-class coefficient heatmap ---
    plt.figure(figsize=(8,6))
    sns.heatmap(
        coef_df.iloc[:,:-1], cmap="coolwarm", center=0,
        annot=True, fmt=".2f"
    )
    plt.title("Logistic Regression Coefficients per Class")
    plt.tight_layout()
    plt.show()
    print("\n")

# ----------------------------------------------------------
# Naïve Bayes: Mean Feature Values per Class
# ----------------------------------------------------------
if hasattr(nb_multi, "theta_"):
    nb_df = pd.DataFrame(
        nb_multi.theta_,
        columns=feature_names,
        index=[class_names[c] for c in np.unique(y)]
    )

    plt.figure(figsize=(10,6))
    sns.heatmap(
        nb_df.T, 
        cmap="YlGnBu", 
        annot=True,        # show numbers
        fmt=".2f",         # round to 2 decimals
        cbar_kws={'label': 'Mean Feature Value'},
    )
    plt.title("Naïve Bayes: Mean Feature Values per Class")
    plt.tight_layout()
    plt.show()
    print("\n")

```
{% endcapture %}
{% include codeinput.html content=ex %}

