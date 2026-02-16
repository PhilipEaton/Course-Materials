---
layout: jupyternotebook
title: Machine Learning Foundations – Project 05
course_home: /courses/machine-learning-foundations/
nav_section: homework
nav_order: 5
---

# Project 5: Decision Trees and Ensemble Methods


## *Building models one question at a time.*

In this project, you’ll build and evaluate **Decision Tree** classifiers and their more powerful ensemble counterparts, such as **Random Forests** and **Gradient-Boosted Trees**.  

You’ll explore how trees learn by splitting data, how model complexity affects overfitting, and how combining multiple trees can dramatically improve performance.

---

## Learning Objectives

By the end of this project, you will be able to:

- **Frame & prep the task:** pick features `X` and target `y`, encode categoricals; scale only if needed.
- **Train & read a Decision Tree:** understand splits via **Gini/Entropy**, and **visualize** the tree.
- **Control complexity:** tune `max_depth`, `min_samples_leaf`, and `ccp_alpha` to prevent over/underfitting.
- **Evaluate properly:** report **Accuracy/Precision/Recall/F1/ROC-AUC**, and show a **confusion matrix**.
- **Use ensembles:** train a **Random Forest** (and optionally **Gradient Boosting**); explain **bagging vs. boosting** and compare to a single tree.
- **Interpret features & communicate:** rank **feature importance**, summarize insights, and note limitations/biases.

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



2. **Update and run the code**: Use the Decision Tree and Random Forest example code below as your template.

**Your goal:**  
Modify it so that it correctly loads, processes, and analyzes your chosen dataset.

Add concise comments above each major code block (as indicated) explaining what that block does.

**You’ll need to:**  
- **Load your dataset** and clearly set **X (features)** and **y (target)**.
- Ensure the remaining code runs.
- (Optional) Include extra bells and whistles from lecture you think would add to your understanding of the model and helps build a better model.

3. **Analyze and Report**: Write a short report (including key plots where appropriate!) interpreting the results of your analysis and what they reveal about the dataset you chose to study
    Your report should read in a **semi-professional** tone, similar to a technical summary you might provide to a customer who asked you to build this model. The goal is to clearly explain:    
    - **Overall performance:** How do the Tree and Forest perform on the **test set** (Accuracy, Precision/Recall/F1, ROC-AUC)?
    - **Model complexity & pruning (Tree):** What did your **validation/pruning curve** (e.g., vs `ccp_alpha`, `max_depth`) show?
        - Justify your chosen complexity and discuss how it affected test performance.
    - **Feature importance & interpretation:** Which features rank highest in **feature importance** (Tree and Forest)? 
        - Provide at least one concrete takeaway about the data.
    - **Error analysis:** What patterns appear in the **confusion matrix**? Which classes are confused? 
        - Can you hazard a guess as to why?
    - **Ensembles vs single tree:** In what ways did the **Random Forest** improve stability/generalization over a single tree? 
        - Mention variance reduction, and robustness to noise.


    Your notebook should output:

    - **Model details & comparisons**
    - [ ] Train a **Decision Tree** (try `criterion='gini'` and/or `'entropy'`).
    - [ ] Train a ***Random Forest** (try `criterion='gini'` and/or `'entropy'`).
    - [ ] Compare the performance of each model and comment on and justify which model performed best.

    - **Core metrics (test set)**
    - [ ] **Accuracy**
    - [ ] **Precision**, **Recall**, **F1-score** (per class)
    - [ ] **ROC-AUC** (binary; or one-vs-rest for multiclass)
    - [ ] **Confusion matrix** (labeled axes/classes)

    - **Model complexity & pruning**
    - [ ] **Validation/pruning curve** for the tree (e.g., accuracy vs **`ccp_alpha`** and/or **`max_depth`**, **`min_samples_leaf`**).
    - [ ] Briefly justify your chosen complexity based on the curve(s).

    - **Interpretability**
    - [ ] **Tree visualization**.
    - [ ] **Feature importance** bar plot(s) for Tree and Forest.

    - **Preprocessing checks (brief)**
    - [ ] Confirm **encoding** of categorical features.
    - [ ] Note that **scaling is usually not required** for trees/forests.
    - [ ] Set `random_state` where applicable for reproducibility.








### What to Submit

- A semi-professional report deailing your findings, written for a layperson.
    - Some details of what you could include are given above and must include are given below. 
- An indepth reflection about the project.
    - Suggestions on what you can reflect on are below.

1. **A clean, runnable Jupyter notebook** that:
    - [ ] loads and briefly describes your dataset (**categorical target**; binary or multiclass),
    - [ ] prepares data (handle missing values; **encode** categorical features; note that **scaling is usually not required** for trees/forests),
    - [ ] performs a **train/test split** (use `stratify=y` for classification),
    - [ ] trains a **Decision Tree** (try `criterion='gini'` and/or `'entropy'`) and a **Random Forest**,
    - [ ] includes **test-set metrics**: Accuracy, Precision, Recall, F1 (per class as needed) and **ROC-AUC** (OvR for multiclass),
    - [ ] shows a **confusion matrix** (labeled axes/classes),
    - [ ] produces a **model complexity/pruning curve** for the tree (e.g., accuracy vs `ccp_alpha`, and/or vs `max_depth` / `min_samples_leaf`) and states the chosen setting,
    - [ ] includes a **tree visualization** (with feature/threshold labels),
    - [ ] shows **feature importance** bar plot(s) for the Tree and the Forest,
    - [ ] uses clear **block comments** explaining each major step,
    - [ ] runs top-to-bottom **without errors** and sets a `random_state` where applicable for reproducibility.

2. **A short report (2–4 pages, PDF)** that:
    - [ ] states your **question** and briefly describes the dataset and target,
    - [ ] summarizes **model performance** on the **test set** (Accuracy, Precision/Recall/F1, ROC-AUC).
    - [ ] explains **model complexity & pruning** choices for the tree (what the curve showed; why you chose your `ccp_alpha` / depth / leaf settings; impact on test performance),
    - [ ] interprets **feature importance** (Tree and Forest) and provides at least **one concrete takeaway** about the data,
    - [ ] performs **error analysis** using the confusion matrix (which classes are confused, and plausible reasons),
    - [ ] compares **Random Forest vs single tree** (stability/generalization, variance reduction, robustness; include OOB insight if used),
    - [ ] includes and references key **figures** (pruning/validation curve, feature importance, confusion matrix, ROC curves).


> **Reminder:** Your notebook should demonstrate a clear, step-by-step workflow, not just working code.  
> Use comments, titles, and plots to tell the *story* of your analysis.


## The Code

The cell below contains a complete example demonstrating both Decision Trees and Random Forests in action.

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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, # feature scaling & label encoding
    PolynomialFeatures, label_binarize,
    OneHotEncoder
)

# --- scikit-learn: Metrics ---
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, roc_auc_score, auc, classification_report,
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

# --- scikit-learn: Tree and Ensembles ---
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# --- scikit-learn: Multiclass ---
from sklearn.multiclass import OneVsRestClassifier

# --- scikit-learn: Inspection ---
from sklearn.inspection import PartialDependenceDisplay

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


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
X = df_encoded.drop(columns = ['target'])
y = df_encoded['target']


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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
# Add a comment explaining what the next section of code does
clf_tree = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
clf_forest = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)

clf_tree.fit(X_train, y_train)
clf_forest.fit(X_train, y_train)

# Add a comment explaining what the next section of code does
cv_tree = cross_val_score(clf_tree, X_train, y_train, cv=5, scoring="accuracy")
cv_forest = cross_val_score(clf_forest, X_train, y_train, cv=5, scoring="accuracy")

print("=== Cross-Validation Accuracy (Training Data) ===")
print(f"Decision Tree → Mean: {cv_tree.mean():.3f} ± {cv_tree.std():.3f}")
print(f"Random Forest → Mean: {cv_forest.mean():.3f} ± {cv_forest.std():.3f}")


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
cv_tree = cross_val_score(clf_tree, X_train, y_train, cv=5, scoring='accuracy')
cv_forest = cross_val_score(clf_forest, X_train, y_train, cv=5, scoring='accuracy')

print("=== Cross-Validation Results (Training Set) ===")
print(f"Decision Tree  →  Mean={cv_tree.mean():.3f} ± {cv_tree.std():.3f}")
print(f"Random Forest  →  Mean={cv_forest.mean():.3f} ± {cv_forest.std():.3f}")


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
def summarize_multiclass(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    return acc, prec, rec, f1

y_pred_tree = clf_tree.predict(X_test)
y_pred_forest = clf_forest.predict(X_test)

summarize_multiclass("Decision Tree", y_test, y_pred_tree)
summarize_multiclass("Random Forest", y_test, y_pred_forest)


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10,4))

for ax, model_name, y_pred in zip(
    axes, ["Decision Tree", "Random Forest"], [y_pred_tree, y_pred_forest]
):
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=class_names,
        cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.grid(False)

plt.tight_layout()
plt.show()


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
path = clf_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]
train_scores, test_scores = [], []

for alpha in ccp_alphas:
    tree_tmp = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    tree_tmp.fit(X_train, y_train)
    train_scores.append(tree_tmp.score(X_train, y_train))
    test_scores.append(tree_tmp.score(X_test, y_test))

plt.figure(figsize=(7,5))
plt.plot(ccp_alphas, train_scores, marker="o", label="Train Accuracy")
plt.plot(ccp_alphas, test_scores, marker="o", label="Test Accuracy")
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.title("Decision Tree Pruning Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --- Rebuild Models Using a Tuned ccp_alpha Value ---
# After inspecting the pruning curve above, choose a ccp_alpha
# (e.g., one that balances train and test accuracy or just before over-pruning).

chosen_alpha = 0.01  # 👈 Adjust this value based on your pruning curve

# --- Refit the Decision Tree with the chosen alpha ---
clf_tree_tuned = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
    ccp_alpha=chosen_alpha
)
clf_tree_tuned.fit(X_train, y_train)

# --- Optionally refit Random Forest with similar bias control ---
# While Random Forests average over trees, you can still apply a pruning alpha to each.
clf_forest_tuned = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    ccp_alpha=chosen_alpha,
    oob_score=True
)
clf_forest_tuned.fit(X_train, y_train)

# --- Evaluate tuned models on the test set ---
print("=== Tuned Decision Tree ===")
y_pred_tree_tuned = clf_tree_tuned.predict(X_test)
print(classification_report(y_test, y_pred_tree_tuned, target_names=class_names))
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree_tuned):.3f}")

print("\n=== Tuned Random Forest ===")
y_pred_forest_tuned = clf_forest_tuned.predict(X_test)
print(classification_report(y_test, y_pred_forest_tuned, target_names=class_names))
print(f"Accuracy: {accuracy_score(y_test, y_pred_forest_tuned):.3f}")

# --- Optional quick comparison summary ---
summary_df = pd.DataFrame({
    "Model": ["Tree (Original)", "Tree (Tuned)", "Forest (Original)", "Forest (Tuned)"],
    "Test Accuracy": [
        accuracy_score(y_test, y_pred_tree),
        accuracy_score(y_test, y_pred_tree_tuned),
        accuracy_score(y_test, y_pred_forest),
        accuracy_score(y_test, y_pred_forest_tuned),
    ]
})
print("\n=== Model Comparison ===")
display(summary_df)



# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Decision Tree": clf_tree_tuned.feature_importances_,
    "Random Forest": clf_forest_tuned.feature_importances_,
}).set_index("Feature")

imp_df.plot(kind="barh", figsize=(8,5))
plt.title("Feature Importances: Decision Tree vs Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------

selected_class = 0 # <- change this to change the class you are plotting for.

fig, ax = plt.subplots(2, 2, figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    clf_forest_tuned,
    X_train,
    features=[0, 1, 2, 3],
    feature_names=feature_names,
    target=2,  # Example class (Virginica)
    ax = ax
)
plt.suptitle(f"Partial Dependence (Class = {class_names[selected_class]})", y=1.02)
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
plt.figure(figsize=(12,7))
plot_tree(
    clf_tree_tuned,
    filled=True,
    feature_names=feature_names,
    class_names=class_names,
    fontsize=9
)
plt.title("Decision Tree (max_depth=3)")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
# --- ROC Curves (Multiclass One-vs-Rest) ---
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(7,6))
for model, name in zip([clf_tree_tuned, clf_forest_tuned], ["Decision Tree", "Random Forest"]):
    y_score = model.predict_proba(X_test)
    for i, cls in enumerate(model.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.8, alpha=0.7, label=f"{name}: {class_names[cls]} (AUC={roc_auc:.3f})")

plt.plot([0,1],[0,1],"--",color="gray", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Multiclass, One-vs-Rest)")
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Macro AUC Summary ---
for model, name in zip([clf_tree, clf_forest], ["Decision Tree", "Random Forest"]):
    y_score = model.predict_proba(X_test)
    auc_macro = roc_auc_score(y_test_bin, y_score, multi_class="ovr", average="macro")
    print(f"{name} OvR Macro-AUC: {auc_macro:.3f}")


```
{% endcapture %}
{% include codeinput.html content=ex %}

