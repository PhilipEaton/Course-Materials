---
layout: jupyternotebook
title: Machine Learning Foundations – Lecture 06
course_home: /courses/machine-learning-foundations/
nav_section: lectures
nav_order: 6
---

# Lecture 06: Integration, Communication, & Ethics in Machine Learning  

First, let's import the needed libraries and functions:

{% capture ex %}
```python
# ===============================================================
# === Import Libraries ===
# ===============================================================

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
    make_classification, make_moons, load_diabetes
)

# --- scikit-learn: Model Preparation ---
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, 
    cross_val_score, GridSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, # feature scaling & label encoding
    PolynomialFeatures  
)

# --- scikit-learn: Metrics ---
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, roc_auc_score, classification_report,
    mean_squared_error, r2_score,
    mean_absolute_error, mean_absolute_error
)

# --- scikit-learn: Inspection ---
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


# --- scikit-learn: Dummy Classifier for baselines ---
from sklearn.dummy import DummyClassifier

# --- scikit-learn: Naive Bayes ---
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# --- scikit-learn: Trees ---
from sklearn.tree import (
    DecisionTreeClassifier, plot_tree,
    DecisionTreeRegressor
)
from sklearn.ensemble import RandomForestClassifier

# --- scikit-learn: Algorithms ---
from sklearn.linear_model import ( 
    LinearRegression,   # Linear Regression Model
    LogisticRegression, # Logistic Regression Model
    Ridge, Lasso
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# --- scikit-learn: Dimensionality Reduction ---
from sklearn.decomposition import PCA                    # reduce features for 2D visualization

# --- statsmodels: model refinement ---
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- statsmodels: Pipelines ---
from sklearn.pipeline import Pipeline


# --- Visualization Utilities for Trees / Boundaries ---
from matplotlib.colors import ListedColormap             # color maps for decision boundaries

# --- Other Useful Tools ---
import warnings
warnings.filterwarnings("ignore")  # keep output clean for class demos
```

{% endcapture %}
{% include codeinput.html content=ex %}  


---
## Learning Objective  

By the end of this lecture, you will be able to:

- Compare models fairly using **cross-validation** and interpret what those results tell you.  
- Understand how tools like **regularization** and **PCA** improve generalization.  
- Build simple **pipelines** that make your work reproducible and shareable.  
- Communicate model results to **non-technical audiences** clearly and confidently.  
- Recognize the **ethical dimensions** of machine learning, including bias, fairness, and transparency.  
---


In our previous lectures, we explored the **core families of machine learning models** — from instance-based and probabilistic methods to linear, logistic, and tree-based models. Each one gave us a new way to model patterns, make predictions, and learn from data.  

Today, we’re going to take a step back from building new models and focus on **how to use them well** — responsibly, reproducibly, and communicatively.  




## Why This Lecture Matters  

Most of the work of a real data scientist happens *after* the model is trained. Data scientists spend their time comparing approaches, tuning parameters, documenting results, and explaining what the numbers mean — and what they *don’t* mean.  

Today’s lecture is about building that professional muscle:  
> how to validate your results, improve them responsibly, and present them with honesty and clarity.  

By the end of class, you’ll see that good machine learning isn’t about chasing the highest accuracy — it’s about building **trustworthy, interpretable, and communicable models**.  


## What We’ll Do Today  

1. **Reconnect everything** we’ve learned — where each model fits and when to use it.  
2. Learn to **compare and validate models** using cross-validation and pipelines.  
3. Explore **regularization and PCA** as tools for improving generalization.  
4. Practice **communicating results** effectively through visuals and clear summaries.  
5. Reflect on **responsible AI** — how ethics, fairness, and transparency affect real-world ML.  


> “A model’s value isn’t just in how well it predicts. It is in how well we understand, trust, and communicate it.”



## Review & Integration

Before we move forward, let’s look at the **big picture** of what we’ve covered so far in this course.  

Across the past five lectures, we explored a range of models that fall into different families of machine learning. Each one makes different assumptions about data, but all share a similar goal: **learning from examples to make predictions or uncover structure.**


### The Landscape of Machine Learning

| Model Family | Example Algorithms | What It Does | When to Use |
|:--------------|:-------------------|:--------------|:--------------|
| **Instance-Based** | k-Nearest Neighbors | Compares new data to known examples using distance | When “similar things should behave similarly” |
| **Clustering** | k-Means | Groups data based on similarity without labels | When we don’t know the categories ahead of time |
| **Probabilistic** | Naïve Bayes | Uses probability and independence assumptions to classify | When features are mostly independent and categorical |
| **Linear Models** | Linear Regression | Predicts continuous outcomes with weighted sums of features | When relationships look linear and continuous |
| **Classification (Linear Extension)** | Logistic Regression | Predicts class membership with a probability curve | When you want probabilities or binary outcomes |
| **Tree-Based Models** | Decision Trees, Random Forests | Splits data by feature thresholds into decision rules | When you want interpretable, flexible, non-linear boundaries |


### The Common Thread

Every model we’ve learned shares the same basic workflow:

1. **Prepare the data** (clean, encode, scale).  
2. **Fit the model** (learn patterns).  
3. **Evaluate the model** (accuracy, R², confusion matrix, etc.).  
4. **Interpret and communicate** the results.  

That means:  
> You don’t need to learn a *new process* for every algorithm — you just plug a new model into the same framework.



<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion Prompt:</b> 
    
- Which models gave the *clearest* or *most interpretable* results?  
- Which ones performed *best* — and why might that be?  
- What trade-offs did you notice between interpretability and accuracy?  
</div>







## Model Comparison & Validation

Once we’ve built multiple models (you should always use multiple models!), a natural question arises:  

> “How do I know which one is "*better*"?”

Evaluating and comparing models is one of the most critical — and overlooked — parts of machine learning. 


### The Goal of Model Validation  

Our goal **is not** to find the model that performs "*best*" on the training set — it’s to find the one that: 
- performs **most consistently** on new, unseen data,
- can be **easily and readily interpreted**, and
- **makes sense** given the context of the problem.








### Why Simple Train/Test Splits Generally Aren’t Enough  

When we split data into a **training set** and a **test set**, we often get a good first estimate of model performance.  But that estimate depends **heavily on how the data was split**. If we got lucky (or unlucky) with one random split, the model’s apparent performance might not reflect reality.  

To fix this, we use a more robust method called **cross-validation**.

### Cross-Validation (CV)  

**Idea:**  Instead of a single train/test split, we split the dataset into *k* parts (called *folds*). We then:
1. Train on *k-1* folds  
2. Test on the remaining fold  
3. Repeat until every fold has been used for testing once

| Fold | Training | Testing |
|:-----|:----------|:--------|
| 1 | 🔵🔵🔵🔵⚪ | ⚪ |
| 2 | 🔵🔵🔵⚪🔵 | ⚪ |
| 3 | 🔵🔵⚪🔵🔵 | ⚪ |
| ... | ... | ... |
| *k* | ⚪🔵🔵🔵🔵 | ⚪ |

This gives us *k* performance scores, which we average to estimate the model’s true performance.

**Benefits:**  
- Reduces dependence on any single split  
- Gives a more stable and fair estimate of model quality  
- Works for both regression and classification models  

**Limitations:**  
- Can't really do this with small samples. (Use bootstrapping if you have small samples.)


{% capture ex %}
```python
# --- Demonstration: Cross-Validation for Model Comparison ---

# Load a regression dataset
X, y = load_diabetes(return_X_y=True)

# Define two models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42)
}

# Define 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate each model
means, stds = [], []
labels = []

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    means.append(np.mean(scores))
    stds.append(np.std(scores))
    labels.append(name)
    print(f"=== {name} ===")
    print("Cross-Validation R² Scores:", np.round(scores, 3),"\n")
    print(f"Average R²: {means[-1]:.3f}")
    print(f"St. Dev. R²: {stds[-1]:.3f}\n")
    
# Bar plot for clarity
plt.figure(figsize=(6,4))
plt.bar(labels, means, yerr=stds, capsize=5, color=["#88CCEE","#CC6677"])
plt.ylabel("Cross-Validated R²")
plt.title("Model Comparison via Cross-Validation")
plt.ylim(0, 1)
plt.show()
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    === Linear Regression ===
    Cross-Validation R² Scores: [0.453 0.573 0.391 0.584 0.391] 
    
    Average R²: 0.478
    St. Dev. R²: 0.085
    
    === Decision Tree ===
    Cross-Validation R² Scores: [0.334 0.34  0.16  0.355 0.078] 
    
    Average R²: 0.253
    St. Dev. R²: 0.113
```
{% endcapture %}
{% include codeoutput.html content=ex %}  



    



<img
  src="{{ '/courses/machine-learning-foundations/images/lec06/output_7_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">   
    


<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion Prompt:</b> 
    
- Which model looks more stable across folds?
- Does higher average accuracy always mean the better model?
</div>






### Building Pipelines for Reproducibility  

When models require preprocessing (scaling, encoding, etc.), we often end up with separate steps scattered across cells or in different location of our script. This is fine when you are exploring your data and poking around on your own. 

Hoever, this makes it hard (if not impossible) to reproduce or share results.

To help with this consider cleaing and organizing your code before you close it for the day. Also, scikit-learn provides a **Pipeline** class — a structured way to connect preprocessing and modeling steps into one object.


#### Why Pipelines Matter  

- Prevent **data leakage** (this ensures the model never “peeks” at the test data).  
- Make experiments **repeatable** and easy to share.  
- Allow you to integrate preprocessing and model tuning in a **single workflow**.


Let’s see a simple example using a pipeline with scaling + an SVM model:








### Pipeline: Scaling + Desision Tree

{% capture ex %}
```python
# === Decision Tree Pipeline with Cross-Validation ===

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target

# ---------------------------------------------------------
# Create a pipeline (scaling + model)
# ---------------------------------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),  # ensures consistent feature scaling
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=42))
])

# ---------------------------------------------------------
# Perform 5-fold cross-validation
# ---------------------------------------------------------
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------
print("Cross-Validation Scores (Accuracy per Fold):")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.3f}")

print("\nOverall Performance:")
print(f"  Mean Accuracy: {cv_scores.mean():.3f}")
print(f"  Standard Deviation: {cv_scores.std():.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    Cross-Validation Scores (Accuracy per Fold):
      Fold 1: 0.967
      Fold 2: 0.967
      Fold 3: 0.900
      Fold 4: 0.933
      Fold 5: 1.000
    
    Overall Performance:
      Mean Accuracy: 0.953
      Standard Deviation: 0.034
```
{% endcapture %}
{% include codeoutput.html content=ex %}  










### Pipeline: Scaling + k-NN

{% capture ex %}
```python
# === K-Nearest Neighbors Pipeline with Cross-Validation ===

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target

# ---------------------------------------------------------
# Create a pipeline (scaling + KNN model)
# ---------------------------------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),          # feature scaling is critical for KNN
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# ---------------------------------------------------------
# Perform 5-fold cross-validation
# ---------------------------------------------------------
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------
print("Cross-Validation Scores (Accuracy per Fold):")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.3f}")

print("\nOverall Performance:")
print(f"  Mean Accuracy: {cv_scores.mean():.3f}")
print(f"  Standard Deviation: {cv_scores.std():.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    Cross-Validation Scores (Accuracy per Fold):
      Fold 1: 0.967
      Fold 2: 0.967
      Fold 3: 0.933
      Fold 4: 0.933
      Fold 5: 1.000
    
    Overall Performance:
      Mean Accuracy: 0.960
      Standard Deviation: 0.025
```
{% endcapture %}
{% include codeoutput.html content=ex %}  










### Pipeline: Scaling + (k-NN & Desision Tree)

{% capture ex %}
```python
# === Comparing KNN and Decision Tree Pipelines with Cross-Validation ===

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target

# ---------------------------------------------------------
# Define pipelines for both models
# ---------------------------------------------------------
pipelines = {
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),   # harmless for trees, critical for KNN
        ("tree", DecisionTreeClassifier(max_depth=4, random_state=42))
    ])
}

# ---------------------------------------------------------
# Run 5-fold cross-validation for each pipeline
# ---------------------------------------------------------
results = {}
for name, model in pipelines.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"\n=== {name} ===")
    print("Cross-Validation Scores (Accuracy per Fold):")
    for i, score in enumerate(scores, 1):
        print(f"  Fold {i}: {score:.3f}")
    print(f"Mean Accuracy: {scores.mean():.3f}")
    print(f"Standard Deviation: {scores.std():.3f}")

# ---------------------------------------------------------
# Compare Models
# ---------------------------------------------------------
print("\n=== Summary Comparison ===")
for name, scores in results.items():
    print(f"{name:20s} → Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    === K-Nearest Neighbors ===
    Cross-Validation Scores (Accuracy per Fold):
      Fold 1: 0.967
      Fold 2: 0.967
      Fold 3: 0.933
      Fold 4: 0.933
      Fold 5: 1.000
    Mean Accuracy: 0.960
    Standard Deviation: 0.025
    
    === Decision Tree ===
    Cross-Validation Scores (Accuracy per Fold):
      Fold 1: 0.967
      Fold 2: 0.967
      Fold 3: 0.900
      Fold 4: 0.933
      Fold 5: 1.000
    Mean Accuracy: 0.953
    Standard Deviation: 0.034
    
    === Summary Comparison ===
    K-Nearest Neighbors  → Mean: 0.960, Std: 0.025
    Decision Tree        → Mean: 0.953, Std: 0.034
```
{% endcapture %}
{% include codeoutput.html content=ex %}  


    








### Pipeline: Scaling + (k-NN & Desision Tree & Random Forest)

{% capture ex %}
```python
# === Comparing KNN and Decision Tree Pipelines with Cross-Validation ===

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target

# ---------------------------------------------------------
# Define pipelines for both models
# ---------------------------------------------------------
pipelines = {
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),   # harmless for trees, critical for KNN
        ("tree", DecisionTreeClassifier(max_depth=4, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),   # harmless for trees, critical for KNN
        ("tree", RandomForestClassifier(max_depth=4, random_state=42))
    ])
}

# ---------------------------------------------------------
# Run 5-fold cross-validation for each pipeline
# ---------------------------------------------------------
results = {}
for name, model in pipelines.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"\n=== {name} ===")
    print("Cross-Validation Scores (Accuracy per Fold):")
    for i, score in enumerate(scores, 1):
        print(f"  Fold {i}: {score:.3f}")
    print(f"Mean Accuracy: {scores.mean():.3f}")
    print(f"Standard Deviation: {scores.std():.3f}")

# ---------------------------------------------------------
# Compare Models
# ---------------------------------------------------------
print("\n=== Summary Comparison ===")
for name, scores in results.items():
    print(f"{name:20s} → Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    === K-Nearest Neighbors ===
    Cross-Validation Scores (Accuracy per Fold):
      Fold 1: 0.967
      Fold 2: 0.967
      Fold 3: 0.933
      Fold 4: 0.933
      Fold 5: 1.000
    Mean Accuracy: 0.960
    Standard Deviation: 0.025
    
    === Decision Tree ===
    Cross-Validation Scores (Accuracy per Fold):
      Fold 1: 0.967
      Fold 2: 0.967
      Fold 3: 0.900
      Fold 4: 0.933
      Fold 5: 1.000
    Mean Accuracy: 0.953
    Standard Deviation: 0.034
    
    === Random Forest ===
    Cross-Validation Scores (Accuracy per Fold):
      Fold 1: 0.967
      Fold 2: 0.967
      Fold 3: 0.933
      Fold 4: 0.967
      Fold 5: 1.000
    Mean Accuracy: 0.967
    Standard Deviation: 0.021
    
    === Summary Comparison ===
    K-Nearest Neighbors  → Mean: 0.960, Std: 0.025
    Decision Tree        → Mean: 0.953, Std: 0.034
    Random Forest        → Mean: 0.967, Std: 0.021
```
{% endcapture %}
{% include codeoutput.html content=ex %}  


    



<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion Prompt:</b> 
    
- What are we actually doing when we tune hyperparameters?  
- How might using a **pipeline** improve transparency and collaboration in ML projects?  
- What other stages of your earlier projects could benefit from being wrapped in a pipeline?
</div>












### Pipeline: Scaling + Feature Selection + (k-NN & Desision Tree) + Hyperparameter Tuning

{% capture ex %}
```python
# === Full Modeling Workflow with Feature Selection Reporting ===
# Includes scaling, feature selection, model tuning, and feature tracking

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target
feature_names = np.array(iris.feature_names)

# ---------------------------------------------------------
# 2. Define pipelines for two models
# ---------------------------------------------------------
pipelines = {
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=3)),
        ("knn", KNeighborsClassifier())
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),  # harmless for trees, keeps structure consistent
        ("select", SelectKBest(score_func=f_classif, k=3)),
        ("tree", DecisionTreeClassifier(random_state=42))
    ])
}

# ---------------------------------------------------------
# 3. Define hyperparameter grids
# ---------------------------------------------------------
param_grids = {
    "K-Nearest Neighbors": {
        "knn__n_neighbors": [3, 5, 7, 9],
        "knn__weights": ["uniform", "distance"]
    },
    "Decision Tree": {
        "tree__max_depth": [2, 3, 4, 5, None],
        "tree__criterion": ["gini", "entropy"]
    }
}

# ---------------------------------------------------------
# 4. Run GridSearchCV for each pipeline
# ---------------------------------------------------------
results = {}
for name, pipe in pipelines.items():
    grid = GridSearchCV(pipe, param_grids[name], cv=5, scoring="accuracy")
    grid.fit(X, y)
    results[name] = grid
    
    # Extract best pipeline and feature selection info
    best_pipeline = grid.best_estimator_
    selector = best_pipeline.named_steps["select"]
    mask = selector.get_support()
    
    selected = feature_names[mask]
    dropped = feature_names[~mask]
    
    # Display results
    print(f"\n=== {name} ===")
    print(f"Selected Features ({len(selected)}):\n {', '.join(selected)}")
    print(f"Dropped Features ({len(dropped)}):\n {', '.join(dropped)}")
    print("\n")
    print(f"Best Parameters:\n {grid.best_params_}")
    print(f"Best CV Mean Accuracy: {grid.best_score_:.3f}")
    print("\n")
    
# ---------------------------------------------------------
# 5. Summary comparison
# ---------------------------------------------------------
print("\n================================")
print("====== Summary Comparison ======")
for name, grid in results.items():
    print(f"{name:20s} → Mean CV Accuracy: {grid.best_score_:.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    === K-Nearest Neighbors ===
    Selected Features (3):
     sepal length (cm), petal length (cm), petal width (cm)
    Dropped Features (1):
     sepal width (cm)
    
    
    Best Parameters:
     {'knn__n_neighbors': 3, 'knn__weights': 'uniform'}
    Best CV Mean Accuracy: 0.960
    
    
    
    === Decision Tree ===
    Selected Features (3):
     sepal length (cm), petal length (cm), petal width (cm)
    Dropped Features (1):
     sepal width (cm)
    
    
    Best Parameters:
     {'tree__criterion': 'gini', 'tree__max_depth': 3}
    Best CV Mean Accuracy: 0.973
    
    
    
    ================================
    ====== Summary Comparison ======
    K-Nearest Neighbors  → Mean CV Accuracy: 0.960
    Decision Tree        → Mean CV Accuracy: 0.973
```
{% endcapture %}
{% include codeoutput.html content=ex %}  


    







### Reproducibility: Why Random Seeds Matter

If we train the same model twice and get *different results*, we can’t trust the comparison.  

**Many ML algorithms** (e.g., Random Forests, train/test splits) **use randomness** internally. Setting a **random seed** makes those random operations repeatable — a key part of trustworthy data science.

{% capture ex %}
```python
# Load a regression dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

for i in range(6):
    print(f"Batch {i+1}")
    
    # Train twice without fixing random_state
    acc1 = RandomForestClassifier().fit(X_train, y_train).score(X_test, y_test)
    acc2 = RandomForestClassifier().fit(X_train, y_train).score(X_test, y_test)
    
    print(f"Without fixed seed: Run 1 = {acc1:.3f}, Run 2 = {acc2:.3f}")
    
    # Now fix the seed
    acc3 = RandomForestClassifier(random_state=42).fit(X_train, y_train).score(X_test, y_test)
    acc4 = RandomForestClassifier(random_state=42).fit(X_train, y_train).score(X_test, y_test)
    
    print(f"With fixed seed:    Run 1 = {acc3:.3f}, Run 2 = {acc4:.3f}")
    print("\n")
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    Batch 1
    Without fixed seed: Run 1 = 0.978, Run 2 = 0.978
    With fixed seed:    Run 1 = 0.978, Run 2 = 0.978
    
    
    Batch 2
    Without fixed seed: Run 1 = 0.978, Run 2 = 0.956
    With fixed seed:    Run 1 = 0.978, Run 2 = 0.978
    
    
    Batch 3
    Without fixed seed: Run 1 = 0.956, Run 2 = 0.978
    With fixed seed:    Run 1 = 0.978, Run 2 = 0.978
    
    
    Batch 4
    Without fixed seed: Run 1 = 0.956, Run 2 = 0.978
    With fixed seed:    Run 1 = 0.978, Run 2 = 0.978
    
    
    Batch 5
    Without fixed seed: Run 1 = 0.978, Run 2 = 0.978
    With fixed seed:    Run 1 = 0.978, Run 2 = 0.978
    
    
    Batch 6
    Without fixed seed: Run 1 = 0.978, Run 2 = 0.978
    With fixed seed:    Run 1 = 0.978, Run 2 = 0.978
```
{% endcapture %}
{% include codeoutput.html content=ex %}  



    
    


Cross-validation can be used to not only splits data into multiple folds, but also apply different random seeds to avoid random seed effects.

It provides a **more reliable estimate** than a single train/test split and demonstrates why reproducibility is essential.

{% capture ex %}
```python
# === Demonstrating Random State in Cross-Validation ===

# ----------------------------------------------------------
# Load a simple dataset
# ----------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target

# Just so we can see the structure
print(f"Dataset size: {len(X)} samples, {len(np.unique(y))} classes\n")

# ----------------------------------------------------------
# Create two cross-validators: one with a fixed seed, one without
# ----------------------------------------------------------
cv_fixed = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_random = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)

# ----------------------------------------------------------
# Show fold assignments
# ----------------------------------------------------------
print("=== Using random_state = 42 (reproducible) ===")
for fold, (train_idx, test_idx) in enumerate(cv_fixed.split(X, y), 1):
    print(f"Fold {fold}: Test indices = {test_idx[:10]}...")  # show first 10

print("\n=== Using random_state = None (randomized each run) ===")
for fold, (train_idx, test_idx) in enumerate(cv_random.split(X, y), 1):
    print(f"Fold {fold}: Test indices = {test_idx[:10]}...")

# ----------------------------------------------------------
# Rerun to show difference
# ----------------------------------------------------------
print("\nRun this cell again — the top set will stay identical, "
      "but the bottom set will change because its random_state=None.")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    Dataset size: 150 samples, 3 classes
    
    === Using random_state = 42 (reproducible) ===
    Fold 1: Test indices = [ 0 10 11 13 14 15 19 20 21 25]...
    Fold 2: Test indices = [ 2  4  6  7  8  9 22 24 27 28]...
    Fold 3: Test indices = [ 1  3  5 12 16 17 18 23 26 32]...
    
    === Using random_state = None (randomized each run) ===
    Fold 1: Test indices = [ 0 10 11 14 17 19 21 26 32 34]...
    Fold 2: Test indices = [ 1  6  8  9 12 13 18 20 22 28]...
    Fold 3: Test indices = [ 2  3  4  5  7 15 16 23 24 25]...
    
    Run this cell again — the top set will stay identical, but the bottom set will change because its random_state=None.
```
{% endcapture %}
{% include codeoutput.html content=ex %}  





Notice that with the **same** `random_state`, the model produces identical results. But, if we do not set the random state, then we can occationally get different results.  

Always record:
- Data set being used,
- model(s) being fit,
- your **random state/seed** numbers,
    - key **model parameters**, and  
- **software versions** (possibily).

Even a simple markdown cell at the top of your notebook works:

> Experiment Log
> - Dataset: penguins.csv
> - Model: Random Forest (100 trees)
> - Random Seed: 42
> - Date: 2025-10-07


<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b> 

- **Cross-validation** gives us a stable, fair estimate of model performance.  
- **Pipelines** make preprocessing and modeling reproducible.  
- **Hyperparameter tuning** (via GridSearchCV) helps find the best configuration — but it must be validated fairly.  
- Our goal is not just *accuracy*, but **consistency and explainability**.
</div>







## Improving Generalization

Even when two models perform similarly under cross-validation, one might still be more **trustworthy** in practice.  That difference comes from *generalization* — the ability of a model to perform well on unseen data.

### What Does "Generalization" Mean?

Reacll, when a model memorizes noise in the training set it will look great during training but fail miserably when faced with new data. We call that **overfitting**.  

On the other hand, a model that is too simple may miss real relationships — remember this is called **underfitting**.

| Behavior | Description | Visualization |
|:----------|:-------------|:---------------|
| **Underfitting** | Model is too simple to capture the pattern | Almost a straight line through curved data |
| **Good Fit** | Model captures trend but ignores random noise | Smooth curve that follows general pattern |
| **Overfitting** | Model learns every tiny fluctuation (noise) | Wiggly curve that fits all points perfectly |


### Bias–Variance Tradeoff

- **High bias** → Model assumptions are too strong (underfit).  
- **High variance** → Model is too sensitive to training data (overfit).  
- **Goal:** Find the sweet spot between the two.


### Tools to Improve Generalization

Two of the most powerful ways to help models generalize are:

1. **Regularization** → Prevents models from becoming too complex.
    - We saw this before when discussing the logistic model.  
3. **Dimensionality Reduction (PCA)** → Simplifies data by removing redundant features.

We’ll explore both next.








### Regularization

Regularization is a way to combat model complexity by adding a penalty to large coefficients, effectively constraining the model’s complexity. The more large coefficients you have, the fewer terms you need in your model. 

The two most common regularization methods for regression models like linear and logistic regression are **Ridge** (aka L2) and **Lasso** (aka L1).

| Method | Penalty Term | Effect |
|:--------|:----------------|:---------|
| **Ridge** | Adds $\lambda \sum w^2$ | Shrinks coefficients smoothly |
| **Lasso** | Adds $\lambda \sum \vert w \vert$ | Pushes some coefficients exactly to zero |
| **Elastic Net** | Mix of Ridge and Lasso | Balances both effects |

**λ (lambda)** (called `C` in `sklearn`) is a tuning parameter that controls how strong the penalty is.  
- High λ → simpler model (less variance, more bias)  
- Low λ → more flexible model (more variance, less bias)  

Regularization doesn’t make your model “better” — it makes it **more stable and generalizable.**

{% capture ex %}
```python
# Load and prepare data
X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit models
ols = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

# Collect coefficients
coef_data = np.vstack([ols.coef_, ridge.coef_, lasso.coef_])
labels = ["OLS", "Ridge", "Lasso"]

# Plot coefficients
plt.figure(figsize=(8,5))
plt.plot(coef_data.T, marker="o")
plt.xticks(range(X.shape[1]), [f"X{i+1}" for i in range(X.shape[1])])
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.title("Effect of Regularization on Model Coefficients")
plt.legend(labels)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Display R² scores
print("=== Model Performance ===")
print(f"OLS R² (Test):   {ols.score(X_test, y_test):.3f}")
print(f"Ridge R² (Test): {ridge.score(X_test, y_test):.3f}")
print(f"Lasso R² (Test): {lasso.score(X_test, y_test):.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec06/output_29_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">  
    

```python
    === Model Performance ===
    OLS R² (Test):   0.477
    Ridge R² (Test): 0.478
    Lasso R² (Test): 0.478
```
{% endcapture %}
{% include codeoutput.html content=ex %}  






### Regularization Beyond Linear Models

Lasso (L1) and Ridge (L2) regularization works by penalizing large coefficients in linear models, shrinking them toward zero. However, models like k-NN and Decision Trees don’t learn explicit weights. These models are regularized by controlling complexity through their hyperparameters.

| Model Type                             | How It Regularizes                                             | Key Parameters                                | Analogy to L1/L2                                                        |
| :------------------------------------- | :------------------------------------------------------------- | :-------------------------------------------- | :---------------------------------------------------------------------- |
| **Linear Models (OLS, Ridge, Lasso)**  | Adds penalty to large coefficient magnitudes                   | `alpha`, `lambda`                             | Direct L1/L2 penalties                                                  |
| **K-Nearest Neighbors (KNN)**          | Controls how local or global the model is                      | `n_neighbors`                                 | Smaller *k* → more flexible (overfit); larger *k* → smoother (underfit) |
| **Decision Trees**                     | Limits depth and leaf size to prevent overfitting              | `max_depth`, `min_samples_split`, `ccp_alpha` | Prunes complexity instead of shrinking weights                          |
| **Random Forests / Gradient Boosting** | Regularize by averaging (bagging) or shrinkage (learning rate) | `n_estimators`, `max_depth`, `learning_rate`  | Reduces variance and overfitting through ensemble averaging             |







### Principal Component Analysis

PCA (Principal Component Analysis) is a technique used to find the best way to combine multiple features using their mutual correlations. This is useful when:

- You have **many correlated features**
- You want to **reduce dimensionality** while preserving most of the information
- You need to **visualize** high-dimensional data


#### PCA vs. Regularization

| Method | Purpose | Typical Use Case |
|:--------|:----------|:----------------|
| **PCA** | **Simplifies the *data*** by reducing redundant dimensions | **Preprocessing step** for visualization or clustering |
| **Regularization/Hyperparameter Tuning** | **Simplifies the *model*** by penalizing large coefficients or tuning model construction parameters | **Model fitting** step for improving generalization in linear models |


Both aim to **reduce overfitting**, but they act on different parts of the pipeline.  


<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion Prompt:</b> 
    
- How would you decide between regularization and PCA for your dataset?  
- Could you combine both in a pipeline?  
- What trade-offs do you see between interpretability and performance?
</div>


{% capture ex %}
```python
# Load dataset
X, y = load_wine(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# Fit PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance
plt.figure(figsize=(6,4))
plt.bar(range(1, 4), pca.explained_variance_ratio_ * 100, color="#88CCEE")
plt.ylabel("Variance Explained (%)")
plt.xlabel("Principal Component")
plt.title("PCA: Variance Captured by Each Component")
plt.show()

# Scatter plot of first two components
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection of Wine Data")
plt.colorbar(label="Class")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec06/output_32_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">  
    



<img
  src="{{ '/courses/machine-learning-foundations/images/lec06/output_32_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">      
    
{% endcapture %}
{% include codeoutput.html content=ex %}  










## Communicating Model Results

We’ve learned the basics of how to train, validate, and improve models; none of that matters if we can’t **communicate what the model found**.

> Machine learning isn’t just about prediction; **it’s about explanation**.

This section focuses on three key communication skills:

1. **Visualization:** Using plots to highlight important results.  
2. **Interpretation:** Turning numbers into plain-language insights.  
3. **Transparency:** Explaining uncertainty, assumptions, and limitations.

### Why Communication Matters

A ***good*** model answers a *question.*  

A ***great*** model tells a *story.*

Whether you’re presenting to other data scientists, scientists in another field, or non-technical audiences, the goal is the same: help them understand what the model learned and what it means.







### Interpreting Feature Importance

Feature importance tells us *which variables had the most influence* on the model’s predictions.

In the following example, certain chemical components of wine strongly determine which class (type of wine) it belongs to.

But be careful: **importance ≠ causation.**

A high importance score means a feature was ***useful to the model***, not necessarily ***the cause of an outcome***. 

Let's take a look at a Random Forest model for wine types:

{% capture ex %}
```python
# --- Train a random forest ---
data = load_wine()
X, y = data.data, data.target
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# --- Feature importances ---
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = np.array(data.feature_names)[indices]

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.barh(features, importances[indices], color="#88CCEE")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(False)
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec06/output_35_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">  
    

{% endcapture %}
{% include codeoutput.html content=ex %}  





#### Good Practices for Explaining Importance (to Non-Experts)

- Use **rankings or visual bars** instead of raw numbers.
    - Visuals are almost always better than displaying raw numbers.
    - If you have to give the numbers, do so as parft of a visual when possible.
- Describe what each feature represents in real-world terms.
    > Flavonoids are colorful, protective chemicals made by plants.
- Connect findings back to the original question.
    > Flavonoids are quite efective in differentiating between types of wine.
- Acknowledge uncertainty: “Feature X appears most predictive, but correlation is not causation.”
    > Flavonoids appear to be the most predicitive, but this this is only a correlation. The flavonoids do not cause the wines to be of different types.

As another example, we can considing the diabetes data set where the target is does the patient have diabetes or not. 

Let's build a multiple linear regression model for this data set:

{% capture ex %}
```python
# --- Load and prepare data ---
X, y = load_diabetes(return_X_y=True)
feature_names = load_diabetes().feature_names
X_scaled = StandardScaler().fit_transform(X)

# --- Fit model ---
model = LinearRegression().fit(X_scaled, y)
coefs = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_,
    "Impact": ["Positive" if c > 0 else "Negative" for c in model.coef_]
})

# --- Display ---
coefs_sorted = coefs.sort_values("Coefficient", ascending=False)
display(coefs_sorted)

# --- Plot ---
plt.figure(figsize=(8, 5))
colors = ["#88CCEE" if c > 0 else "#CC6677" for c in coefs_sorted["Coefficient"]]
plt.barh(coefs_sorted["Feature"], coefs_sorted["Coefficient"], color=colors)
plt.title("Linear Model Coefficients and Direction of Effect")
plt.xlabel("Coefficient Value (Standardized Scale)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

| Feature | Coefficient  | Impact   |
|----------|-------------|----------|
| s5       | 35.734446   | Positive |
| bmi      | 24.726549   | Positive |
| s2       | 22.676163   | Positive |
| bp       | 15.429404   | Positive |
| s4       | 8.422039    | Positive |
| s3       | 4.806138    | Positive |
| s6       | 3.216674    | Positive |
| age      | -0.476121   | Negative |
| sex      | -11.406867  | Negative |
| s1       | -37.679953  | Negative |



<img
  src="{{ '/courses/machine-learning-foundations/images/lec06/output_38_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">      

{% endcapture %}
{% include codeoutput.html content=ex %}  








    


### Explaining Coefficients (to Non-Experts)

Each coefficient represents how much the target changes when that feature changes, *holding everything else constant.*

- **Positive values** → higher feature values increase the predicted value.  
- **Negative values** → higher feature values decrease the predicted value.  
- **Larger *magnitudes*** → stronger influence.


#### Example Translation

Instead of saying:
> “BMI has a coefficient of 24.7.”

Say:
> “Increasing BMI is associated with higher predicted values, holding other factors constant.”

That translation turns abstract math into interpretable insight.


#### Common Pitfalls

- Forgetting to scale features before comparing coefficient magnitudes.  
- Interpreting coefficients as causal relationships.  
- Ignoring uncertainty or correlation between features.







### Communicating Limitations and Uncertainty

Transparent communication builds **trust** in data science.

Always include:
- **What the model can do** (strengths)  
- **What it cannot do** (limitations)  
- **Where it might be biased** (data imbalance, missing variables)  
- **How confident we are** (validation results, error margins)


#### Example Summary Statement

> “Our decision tree correctly identifies wine types about 93% of the time. The biggest clues it uses are the wine’s flavonoids, alcohol level, and color intensity. When we try it on wines from new regions it hasn’t seen before, accuracy drops to 85%, which suggests the model learned the training examples a bit too closely (some overfitting).”

That’s a *complete, transparent* statement — it reports success while acknowledging uncertainty.


<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion Prompt:</b> 
    
- How would you explain your model’s key result to someone in marketing or healthcare?  
- What are the risks of over-simplifying your findings?  
- Which visualizations do you find most effective for telling your model’s story?
</div>







## Responsible AI: Bias, Fairness, and Transparency

Machine learning models are **not neutral**.  They learn from data, and data always reflect the world as it *was*, not necessarily as it *should be.*

If our training data contain bias, our models will too. As a result, if we don’t communicate uncertainty clearly, our models can be misused or misunderstood.

In this section, we’ll discuss:

1. **Where bias comes from**  
2. **How to detect and mitigate it**  
3. **Why fairness and transparency matter in practice**







### Where Does Bias Come From?

Bias can creep into a machine learning system at many points:

| Stage | Description | Example |
|:------|:-------------|:---------|
| **Data Collection** | Who or what was included in the dataset | A medical dataset overrepresents one demographic group |
| **Labeling** | How outcomes were defined or labeled | Past hiring decisions labeled as "successful" may encode bias |
| **Model Design** | Choice of algorithm or features | Ignoring important context features (e.g., socioeconomic status) |
| **Evaluation** | How performance is measured | Accuracy hides group-level disparities |

Bias is not always intentional, but it’s always impactful.


For example, in am imbalanced data set, where one group has much more data than the other(s), the larger group can potentially cominate the models being generated. 

**For example**:
Suppose you have a data set with two groups:
- Group 1 (80% of the data set): Mean = 0, St.Dev. = 1
- Group 2 (20% of the data set): Mean = 2, St.Dev. = 1 

{% capture ex %}
```python
# --- Create biased data ---
np.random.seed(42)
n_samples = 500
group_A = np.random.normal(0, 1, (int(0.8 * n_samples), 2))
group_B = np.random.normal(2, 1, (int(0.2 * n_samples), 2))

X = np.vstack([group_A, group_B])
y = np.array([0] * len(group_A) + [1] * len(group_B))

# --- Split and train ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
plt.title("Confusion Matrix on Biased Dataset")
plt.show()

print(classification_report(y_test, y_pred))

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec06/output_45_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">      
    

```python
                  precision    recall  f1-score   support
    
               0       0.97      0.97      0.97       119
               1       0.90      0.90      0.90        31
    
        accuracy                           0.96       150
       macro avg       0.94      0.94      0.94       150
    weighted avg       0.96      0.96      0.96       150
```
{% endcapture %}
{% include codeoutput.html content=ex %}  



    


### Understanding the Consequences

When one group dominates the dataset, the model learns to prioritize performance for that group. This leads to **unequal accuracy**, which can have serious consequences in high-stakes settings like healthcare, hiring, or criminal justice.


#### Key Lesson

A model that performs well *on average* may still perform *poorly for some groups.*

That’s why fairness must be measured explicitly, not assumed.


{% capture ex %}
```python
# --- Simulate data ---
np.random.seed(42)
n = 600

# Group A: large, well-behaved
X_A = np.random.normal(loc=[2, 2], scale=[1, 1], size=(int(0.8*n), 2))
y_A = (X_A[:, 0] + X_A[:, 1] > 4).astype(int)

# Group B: small, noisy, offset
X_B = np.random.normal(loc=[0.5, 1], scale=[1.3, 1.3], size=(int(0.2*n), 2))
y_B = (X_B[:, 0] + X_B[:, 1] > 3.5).astype(int)

# Combine
X = np.vstack([X_A, X_B])
y = np.concatenate([y_A, y_B])
groups = np.array(["Group A"] * len(X_A) + ["Group B"] * len(X_B))

# --- Split data ---
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.3, random_state=42, stratify=groups
)

# --- Visualize imbalance and distributions ---
fig, axes = plt.subplots(2, 1, figsize=(5, 8))

# Left: Feature space
for g, color in zip(["Group A", "Group B"], ["#88CCEE", "#CC6677"]):
    subset = (groups == g)
    axes[0].scatter(X[subset, 0], X[subset, 1], alpha=0.6, label=g, color=color)

axes[0].set_title("Feature Distribution by Group")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.4)

# Right: Group counts
counts = pd.Series(groups).value_counts().reset_index()
counts.columns = ["Group", "Count"]
axes[1].bar(counts["Group"], counts["Count"]/sum(counts["Count"]), color=["#88CCEE", "#CC6677"])
axes[1].set_title("Group Representation in Data")
axes[1].set_ylabel("Portion")
axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

print(counts["Count"]/sum(counts["Count"]),"\n")

# --- Train model ---
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# --- Subgroup accuracy ---
results = pd.DataFrame({"True": y_test, "Pred": y_pred, "Group": groups_test})
group_acc = (
    results.groupby("Group")[["True", "Pred"]]
    .apply(lambda g: np.mean(g["True"] == g["Pred"]))
    .reset_index(name="Accuracy")
)

plt.bar(group_acc["Group"], group_acc["Accuracy"], color=["#88CCEE", "#CC6677"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy by Subgroup (Illustrating Group Bias)")
plt.show()

print(group_acc,"\n")
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec06/output_47_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">      
    

```python
    0    0.8
    1    0.2
    Name: Count, dtype: float64 
```



<img
  src="{{ '/courses/machine-learning-foundations/images/lec06/output_47_2.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">      
    

```python
         Group  Accuracy
    0  Group A  1.000000
    1  Group B  0.944444 
```
{% endcapture %}
{% include codeoutput.html content=ex %}  




    


## Group Imbalance and Fairness

These visuals show how **unequal group representation** and **different data distributions** can lead to unequal model performance:

1. **Feature Distribution Plot:**  
   Group A’s data is denser and more uniform, while Group B’s is smaller and noisier.  
   The model learns more easily from Group A.

2. **Group Representation Bar Chart:**  
   The imbalance (80% vs 20%) reinforces this bias — Group A dominates training.

3. **Subgroup Accuracy Plot:**  
   When evaluated separately, Group A has much higher accuracy. This reflects a real-world issue: models often perform best on the majority group.

**The result**: good overall accuracy but poor subgroup fairness.

This is why evaluating performance by subgroup — not just overall — is critical in applied machine learning, particularly in health, finance, and hiring contexts.



### Possible fix - Bootstrapping

What if we tried to fix this by **resampling the data**, so that each group contributes equally to the training set?

We can do that with a simple resampling technique called **bootstrapping** — it means *sampling with replacement* to build multiple training sets of equal size. Then, instead of relying on one biased model, we can average results across multiple bootstrapped models to get a fairer estimate.

{% capture ex %}
```python
# Reuse previous dataset
data = pd.DataFrame({
    "Feature_1": X[:, 0],
    "Feature_2": X[:, 1],
    "Target": y,
    "Group": groups
})

n_bootstrap = 30
boot_results = []

# --- Perform bootstrapping ---
for i in range(n_bootstrap):
    # ✅ Explicitly select columns inside apply to avoid deprecation warning
    boot_sample = (
        data.groupby("Group")[["Feature_1", "Feature_2", "Target"]]
        .apply(lambda g: g.sample(
            n=min(len(data[data.Group == "Group B"]), len(g)),
            replace=True
        ))
        .reset_index(drop=True)
    )
    
    # Train/test split
    Xb = boot_sample[["Feature_1", "Feature_2"]].values
    yb = boot_sample["Target"].values
    model = LogisticRegression()
    model.fit(Xb, yb)
    
    # Evaluate on the full test data
    y_pred = model.predict(X_test)
    results = pd.DataFrame({"True": y_test, "Pred": y_pred, "Group": groups_test})
    group_acc = (
        results.groupby("Group")[["True", "Pred"]]
        .apply(lambda g: np.mean(g["True"] == g["Pred"]))
        .reset_index(name="Accuracy")
    )
    group_acc["Iteration"] = i
    boot_results.append(group_acc)

boot_df = pd.concat(boot_results)

# --- Average accuracy across bootstraps ---
boot_summary = boot_df.groupby("Group")["Accuracy"].agg(["mean", "std"]).reset_index()

# --- Plot ---
plt.bar(
    boot_summary["Group"], boot_summary["mean"], 
    yerr=boot_summary["std"], color=["#88CCEE", "#CC6677"], capsize=5
)
plt.ylim(0, 1)
plt.ylabel("Average Accuracy (±SD)")
plt.title(f"Bootstrapped Group Accuracy Across {n_bootstrap} Models")
plt.show()

boot_summary

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec06/output_50_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">      
    
    
| Group   | Mean      | Std       |
|----------|-----------|-----------|
| Group A  | 0.963426  | 0.017486  |
| Group B  | 0.957407  | 0.014095  |

{% endcapture %}
{% include codeoutput.html content=ex %}  











### Bootstrapping to Improve Fairness and Stability

Bootstrapping is a simple yet powerful idea:
- We create multiple training datasets by sampling **with replacement**.
- Each dataset contains a balanced number of examples from each group.
- We train a separate model on each dataset and average the results.

This reduces bias caused by uneven group sizes and makes our results more stable.



### Fairness Metrics and Mitigation

There’s no single definition of “fairness,” but common metrics include:

| Metric | Meaning | Example Goal |
|:--------|:----------|:-------------|
| **Demographic Parity** | Equal positive prediction rate across groups | “Both groups should receive similar acceptance rates.” |
| **Equal Opportunity** | Equal true positive rate across groups | “If qualified, both groups should have equal chance of acceptance.” |
| **Predictive Parity** | Equal precision across groups | “Predictions should be equally trustworthy for all groups.” |


#### How to Mitigate Bias

- **Collect more representative data**
- **Rebalance classes** (e.g., oversampling underrepresented groups via bootstrapping)
- **Audit models** regularly for subgroup performance
- **Engage domain experts** to interpret disparities



### Transparency and Documentation

Responsible AI means being able to answer these questions:

1. What data was used, and how was it collected?  
2. What assumptions were made?  
3. Who benefits from the model’s predictions?  
4. Who might be harmed if it fails?

To ensure accountability, data scientists create **Model Cards** and **Datasheets** — structured summaries that document:

- Purpose and limitations  
- Intended audience  
- Data sources and biases  
- Performance metrics  
- Ethical considerations


#### Example Model Card Summary

> **Model:** Logistic Regression for Disease Prediction   
> **Training Data:** 10,000 patient records (65% female, 35% male)  
> **Performance:** 87% accuracy overall; 91% female, 80% male  
> **Limitations:** Reduced accuracy for smaller demographic groups  
> **Intended Use:** Screening aid, not diagnostic tool  
> **Ethical Note:** Should not be used without clinician oversight





### Communicating Results to Non-Technical Audiences

Numbers are powerful — stories are *memorable*.  As data scientists, we must translate technical results into actionable insights.

Example 1 – Confusion Matrix (technical)

|        | Pred 0 | Pred 1 |
| ------ | ------ | ------ |
| True 0 | 90     | 10     |
| True 1 | 8      | 92     |


Example 2 – Plain-Language Summary
> “Our model correctly identifies 92  of positive cases but misses 8%.  It also incorrectly flags 10% of negative cases.  In practice, this means roughly 1 in 10 people would be told they have the condition when they don’t.”

**Every figure in your reports should have a one-sentence interpretation underneath it.**








### Section Summary — Responsible AI

| Concept | Description | Example |
|:---------|:-------------|:---------|
| **Bias** | Systematic error favoring certain outcomes | Model underpredicts for minority groups |
| **Fairness** | Equal treatment and performance across groups | Demographic parity, equal opportunity |
| **Transparency** | Clear documentation of methods and limits | Model cards, datasheets |
| **Accountability** | Human responsibility for AI decisions | Audit trails, review boards |




### Responsible AI in Practice: Hiring Bias Example
Imagine a company builds an ML model to screen résumés for interviews.  
The training data come from 10 years of historical hiring — mostly male applicants.

**Potential Bias Sources**
1. Training Data Bias – past hiring decisions reflect existing inequality.  
2. Feature Selection Bias – words like “leadership” or “captain” may correlate with male applicants.  
3. Outcome Bias – the model learns “successful” = male.

<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion Prompt:</b> 
    
- Where did bias enter?  
- How could you detect it (metrics, subgroup performance)?  
- What mitigation strategies would you use (rebalancing, removing sensitive features, fairness constraints)?
</div>


<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion Prompt:</b> 
    
- How could bias appear in one of your earlier project datasets?  
- What’s one step you could take to check for fairness in your own models?  
- How can we communicate model limitations without undermining trust?
- What habit will make you a *trustworthy* data scientist (not just a technically good one)?
- How will you communicate model results to a non-technical stakeholder in your field?
</div>


<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaway:</b> 

Responsible AI isn’t an optional extra — it’s an integral part of building credible, ethical, and sustainable machine learning systems.
</div>

