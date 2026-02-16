---
layout: jupyternotebook
title: Machine Learning Foundations – Lecture 05
course_home: /courses/machine-learning-foundations/
nav_section: lectures
nav_order: 5
---

# Lecture 05: Decision Trees & Ensemble Methods

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
    make_classification, make_moons
)

# --- scikit-learn: Model Preparation ---
from sklearn.model_selection import train_test_split   # split data into train/test sets
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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# --- scikit-learn: Algorithms ---
from sklearn.linear_model import (
    LinearRegression,   # Linear Regression Model
    LogisticRegression # Logistic Regression Model
)

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


---

## Learning Objectives
- Understand how decision trees learn to make choices by splitting data based on feature values.  
- Explain the concepts of **Gini impurity** and **entropy**, and how they guide tree construction.  
- Recognize when a tree is **overfitting** or **underfitting**, and know how to control model complexity.  
- Understand the motivation behind **Random Forests** and **Ensemble Learning**, and how combining many “weak” models can create a strong one.  
- Interpret **feature importance** and explain how trees tell us which features matter most.

---

Decision Trees — and their ensemble extensions like Random Forests, Gradient Boosting, XGBoost, and LightGBM — are workhorses of modern data science. They combine simplicity of concept with interpretability, flexibility, and power to make them some of the most widely used and effective machine-learning algorithms today.

Decision Trees are:
- **Easy to interpret**: Their structure is literally a tree of “if–then” decisions that you can read and visualize.
- **Flexible**: They handle numeric (linear and non-linear) and categorical data with minimal preprocessing.
- **Powerful**: Ensembles of trees can outperform linear models, even on messy, real-world datasets.

When you understand Decision Trees, you gain a lens for seeing how **machine-learning models make structured choices** — step by step, branch by branch — and how combining many simple models can create something remarkably strong.









## From Decisions to Trees - Conceptual Foundation

Let’s start with a question:

> Have you ever played *20 Questions*?

You, for example, begin with a broad question (“Is it an animal?”) and, based on the answer, ask a narrower one (“Does it live in water?”). Each question **splits the possibilities** until you’ve isolated the right answer.

That’s ***exactly*** what a **Decision Tree** does.

At its core, a Decision Tree is a series of *yes/no questions* about the data. Each question divides the data into smaller, purer groups — that is, groups that are more uniform in their outcomes.


### The Intuition

Think of the tree as a **flowchart of decisions**:
- Each **node** asks a question about a feature.  
- Each **branch** represents the answer (“yes” or “no”).  
- Each **leaf node** (the end of a branch tree) holds the final prediction (a class label or numeric value).  

The model’s job is to find **which questions** to ask and **in what order** so that it can make a simple model and accurate predictions.


### In Machine Learning Terms

Decision Trees are built automatically using algorithms (like **CART (Classification and Regression Trees)** or **ID3**):
1. The algorithm looks for the feature and threshold that best separate the data (most reduces impurity).  
2. It recursively repeats this process on each subset until stopping conditions are met (e.g., max depth, min samples per leaf).  







#### Visualization

Let's take a peek at an example that builds and visualizes a tiny Decision Tree on the Iris dataset.

Try changing the `max_depth` to see how the tree becomes more detailed — and how quickly it can grow complex.  


{% capture ex %}
```python
# Load sample data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train a shallow tree for easy visualization
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Plot the tree
plt.figure(figsize=(10, 6))
plot_tree(tree, feature_names=feature_names, class_names=target_names,
          filled=True, rounded=True, fontsize=10)
plt.title("Simple Decision Tree on Iris Dataset")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_4_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

{% endcapture %}
{% include codeoutput.html content=ex %}  












### Splitting Criteria and Tree Growth

Now that we understand *what* a Decision Tree is, let’s talk about *how* they are built.

A Decision Tree builds itself by asking:  

> “Which yes/no question divides the data into the cleanest groups?”

This process is called **splitting**, and the “cleanliness” of a split is measured using something called **impurity**.





#### What Is Impurity?

At the start, our dataset is messy, containing contains a mix of different classes (like several species of flowers, for example).  

- A **pure** group contains only one class (for example, all `setosa`).  
- An **impure** group contains a mix of classes (e.g., `setosa` and `versicolor`).

We want to find a feature and threshold that separate those classes as much as possible, creating the most pure groups possible. A tree algorithm tries to maximize purity (i.e., minimize impurity) after each split.






#### Common Measures of Impurity

When building a Decision Tree, the algorithm needs a way to decide how “mixed” or “pure” a node is.
- If **all samples in a node belong to the same class**, we will say the **node is pure (impurity = 0)**.
- If **samples are *evenly* split among several classes**, the node is ***maximally* impure**.

To quantify this, we define impurity using one of several mathematical measures.

| Measure                  | Formula                          | Conceptual Meaning                                                                                                                                                                                                                                                                                                                                                                 |
| :----------------------- | :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Gini Impurity**        | $$ G = 1 - \sum_i p_i^2 $$       | Think of ( p_i ) as the fraction of samples in this node that belong to class ( i ). Gini measures how often you would mislabel a randomly chosen sample if you assigned its class *at random* according to these class probabilities. <br><br>**Low Gini** means one class dominates (the node is “clean”); **high Gini** means the node is mixed.                                |
| **Entropy**              | $$ H = -\sum_i p_i \log_2 p_i $$ | Borrowed from information theory, entropy measures **uncertainty** or “information disorder.” <br><br>When one class dominates (say ( p_i = 1 ) for one class and 0 for others), entropy = 0 because there’s no uncertainty — you know exactly what the class is. When classes are perfectly balanced (e.g., 50/50), entropy is highest because each outcome is equally uncertain. |
| **Classification Error** | $$ E = 1 - \max(p_i) $$          | This is the simplest impurity measure — it only cares about the most common class. It tells you the fraction of samples *not* in the majority class. <br><br>However, because it ignores smaller changes in class balance, it’s less sensitive and rarely used for training (though sometimes used for quick summaries).                                                           |


#### **What is $p_i$?**

$p_i$ represents the **proportion** (or probability) of samples in the node that belong to class $i$:

$$
p_i = \frac{\text{number of samples of class $i$}}{\text{total number of samples in the node}}
$$

So for a binary classification node with 80 samples of Class A and 20 samples of Class B:

$$
p_A = 0.8 \qquad\qquad p_B = 0.2
$$

The impurity measure would give:  


| Measure                  | Equation                          |  Result |
| :----------------------- | :-----------------------------------  | :---------- |
| **Gini Impurity**        | $$ 1 - \big( (0.8)^2 + (0.2)^2 \big)$$      |  = 0.32 |
| **Entropy**              | $$ - \big( (0.8) \log_2 (0.8) + (0.2) \log_2 (0.2) \big)$$ | = 0.72 |
| **Classification Error** | $$ 1 - \text{max}\big(0.8,\, 0.2\big)$$          |          = 0.2 |


The **CART algorithm** (used in `sklearn`) defaults to **Gini Impurity** because it’s fast and effective.



#### Visualize Measures of Impurity

Below, we’ll visualize how impurity changes as a function of $p_i$ for a binary classification. For instance, imagine we have flowers labeled `Red` or `Blue`, and we’re considering splitting on **Petal Length**.


{% capture ex %}
```python
# Define possible "purity" conditions
p = np.linspace(0, 1, 100)
gini = 1 - (p**2 + (1 - p)**2)
entropy = -p*np.log2(p + 1e-9) - (1 - p)*np.log2(1 - p + 1e-9)
error = 1 - np.maximum(p, 1 - p)

plt.figure(figsize=(7,5))
plt.plot(p, gini, label="Gini Impurity", linewidth=2)
plt.plot(p, entropy, label="Entropy", linewidth=2)
plt.plot(p, error, label="Classification Error", linestyle="--")
plt.title("Impurity vs Class Proportion")
plt.xlabel("Proportion of Class 1 (p)")
plt.ylabel("Impurity")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_7_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
 
{% endcapture %}
{% include codeoutput.html content=ex %}  



Notice how impurity is **highest** when classes are evenly mixed (p = 0.5), and **lowest** when one class dominates (p = 0 or 1).

That’s exactly what the algorithm looks for:  

> it tests different feature thresholds to find the split that produces the *largest drop in impurity*.


#### How a Tree Chooses Its Split

At each step, the algorithm:
1. **Tries every feature and every possible cut** (e.g., `Petal Length < 2.5` or `Petal Length < 1.5`).
2. **Calculates how much impurity would drop if it used that cut**.
3. Chooses the feature and threshold that produce the **biggest decrease in impurity**.
4. **Repeats** the process for the new subsets until the stopping criteria are met.

This is called **recursive binary splitting** — because the tree keeps splitting until it’s told to stop.

##### Visualizing a Split

Let’s watch a simple 2D dataset get split by a Decision Tree.


{% capture ex %}
```python
# Visualizes impurity reduction (information gain) vs Feature 1 threshold

# Create a simple 2D dataset (two features, two classes)
X, y = make_classification(
    n_samples=150, n_features=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)

# Train a shallow tree (only one split)
tree = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=42)
tree.fit(X, y)

# Visualize the decision boundary
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
    np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200)
)
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7,5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k")
plt.title("Single Split (max_depth=1): Choosing the Best Threshold")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Highlight the chosen split position (vertical line)
split_threshold = tree.tree_.threshold[0]
plt.axvline(split_threshold, color="k", linestyle="--", lw=2, label=f"Chosen split = {split_threshold:.2f}")
plt.legend()
plt.show()

# -------------------------------------------------------
# Compute impurity drop (information gain) for each possible threshold
def gini_impurity(y_subset):
    """Compute Gini impurity for a given set of labels."""
    if len(y_subset) == 0:
        return 0
    p = np.bincount(y_subset) / len(y_subset)
    return 1 - np.sum(p**2)

# Sort samples by Feature 1 for threshold scanning
sorted_idx = np.argsort(X[:,0])
X_sorted, y_sorted = X[sorted_idx], y[sorted_idx]

thresholds = []
impurity_drop = []

parent_impurity = gini_impurity(y)
n = len(y)

for i in range(1, n):
    thr = (X_sorted[i-1,0] + X_sorted[i,0]) / 2
    left, right = y_sorted[:i], y_sorted[i:]
    n_left, n_right = len(left), len(right)
    child_impurity = (n_left/n) * gini_impurity(left) + (n_right/n) * gini_impurity(right)
    gain = parent_impurity - child_impurity
    thresholds.append(thr)
    impurity_drop.append(gain)

# Plot impurity reduction vs threshold
plt.figure(figsize=(7,4))
plt.plot(thresholds, impurity_drop, color="purple", lw=2)
plt.axvline(split_threshold, color="k", linestyle="--", lw=2, label=f"Chosen split = {split_threshold:.2f}")
plt.title("Impurity Reduction (Information Gain) vs. Feature 1 Threshold")
plt.xlabel("Feature 1 Threshold")
plt.ylabel("Reduction in Gini Impurity")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_10_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    


<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_10_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

{% endcapture %}
{% include codeoutput.html content=ex %}  




The tree made a single split along the feature (Feature 1) that gave the **cleanest separation** between classes.  








### Overfitting and Pruning

Decision Trees are powerful — but also dangerously flexible.

If you let a tree keep growing without limits, it will keep finding smaller and smaller splits… eventually memorizing every little noise or outlier in the training data.

This leads to **overfitting** — when the model fits the training data *too well* and fails to generalize to new data.

As a reminder we have:

| Stage | Description | Typical Behavior |
|:------|:-------------|:----------------|
| **Underfitting** | The tree is too shallow and fails to capture patterns. | **Low** training accuracy, **low** testing accuracy. |
| **Good Fit** | The tree captures the real structure of the data. | **High** training accuracy and **high** testing accuracy. |
| **Overfitting** | The tree is too deep and memorizes noise. | ***Very* high** training accuracy, **low** testing accuracy. |

Each split reduces impurity *just a little*, but even tiny gains add up. A fully grown tree can create one leaf for each training point — giving 100% training accuracy but poor generalization. The goal isn’t to make a perfect training tree — it’s to make a **useful** one.


###  How We Control Tree Growth

We use *regularization parameters* to prevent overfitting:
- `max_depth`: limits how deep the tree can go  
- `min_samples_split`: the minimum number of samples needed to make a split  
- `min_samples_leaf`: the smallest size allowed for a leaf  
- `max_leaf_nodes`: limits the total number of leaves

We can also **prune** the tree after it’s been built — removing unnecessary branches that don’t improve performance.







### Underfit vs Overfit Trees:

{% capture ex %}
```python
# --- Create nonlinear data ---
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# --- Train three trees of different depths ---
models = {
    "Underfit (max_depth=1)": DecisionTreeClassifier(max_depth=1, random_state=42),
    "Good Fit (max_depth=4)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Overfit (max_depth=None)": DecisionTreeClassifier(random_state=42)
}

# --- Plot decision boundaries ---
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 300),
    np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 300)
)

fig, axes = plt.subplots(3, 1, figsize=(6,12))

for ax, (title, model) in zip(axes, models.items()):
    model.fit(X, y)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)
    ax.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k", s=30)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.suptitle("Underfitting vs Good Fit vs Overfitting in Decision Trees", fontsize=14)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_13_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

{% endcapture %}
{% include codeoutput.html content=ex %}  





Notice how:
- The **underfit** model barely separates the classes.  
- The **well-fit** model captures the general pattern while keeping boundaries smooth-ish.  
- The **overfit** model draws jagged, overly complex boundaries that cling to noise.







### Check Training vs Testing Accuracy:

{% capture ex %}
```python
from sklearn.model_selection import train_test_split

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"{'Model':30s} | Train Acc | Test Acc")
print("-" * 50)

for title, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{title:30s} | {train_acc:9.2f} | {test_acc:8.2f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    Model                          | Train Acc | Test Acc
    --------------------------------------------------
    Underfit (max_depth=1)         |      0.82 |     0.78
    Good Fit (max_depth=4)         |      0.90 |     0.90
    Overfit (max_depth=None)       |      1.00 |     0.92
```
{% endcapture %}
{% include codeoutput.html content=ex %}  




Notice the **overfit** model gets perfect training accuracy — but loses quite a bit on the test set. The **well-fit** model achieves a balance possessing good training accuracy *and* good test accuracy.

This is why we use **pruning** and **regularization**. We want a tree that’s deep enough to capture structure, but not so deep that it memorizes noise.







### Pruning in Practice

Scikit-learn allows pruning via:
- `ccp_alpha` → Cost Complexity Pruning (ccp) parameter

A **higher** `ccp_alpha` means **more pruning** (simpler trees). You can tune this hyperparameter using cross-validation to find the sweet spot where validation accuracy peaks.


{% capture ex %}
```python
# === Pruning in Practice: Cost Complexity Pruning (ccp_alpha) ===


# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train an unpruned (fully grown) tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
print(f"Full Tree Depth: {clf.get_depth()}, Leaves: {clf.get_n_leaves()}")

# Get effective alphas (complexity parameters)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train a series of pruned trees using different alphas
clfs = []
for ccp in ccp_alphas:
    model = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp)
    model.fit(X_train, y_train)
    clfs.append(model)

# Remove last tree (often trivial, with only one node)
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# Evaluate performance
train_scores = [m.score(X_train, y_train) for m in clfs]
test_scores  = [m.score(X_test,  y_test)  for m in clfs]
depths       = [m.get_depth() for m in clfs]

# Plot results
plt.figure(figsize=(7,4))
plt.plot(ccp_alphas, train_scores, marker='o', label="Train Accuracy", color='royalblue')
plt.plot(ccp_alphas, test_scores, marker='o', label="Test Accuracy", color='orange')
plt.xscale("log")
plt.xlabel("ccp_alpha (Cost Complexity Parameter)")
plt.ylabel("Accuracy")
plt.title("Effect of Cost Complexity Pruning (ccp_alpha)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Show best alpha
best_alpha = ccp_alphas[np.argmax(test_scores)]
best_model = clfs[np.argmax(test_scores)]
print(f"Best alpha = {best_alpha:.5f}")
print(f"Optimal Depth = {best_model.get_depth()}, Test Accuracy = {max(test_scores):.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

Full Tree Depth: 6, Leaves: 16


<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_17_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    


```python
    Best alpha = 0.00295
    Optimal Depth = 4, Test Accuracy = 0.936
```
{% endcapture %}
{% include codeoutput.html content=ex %}  


    







{% capture ex %}
```python
# --- Load data and split ---
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# --- Overfitted vs. Pruned Trees ---
deep_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
shallow_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
deep_tree.fit(X_train, y_train)
shallow_tree.fit(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, model, title in zip(axes, [deep_tree, shallow_tree],
                            ["Overfitted Tree", "Pruned Tree (max_depth=3)"]):
    plot_tree(model, filled=True, ax=ax, feature_names=data.feature_names,
              class_names=data.target_names, fontsize=6)
    ax.set_title(title)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_18_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
  
{% endcapture %}
{% include codeoutput.html content=ex %}  



  











## Random Forests — Combining Many Trees

By now, we’ve seen the potential that a Decision Tree can have. The only problem is that these trees are pretty unstable; change a few data points, and the tree might split differently and make different predictions. This means a single tree isn't ver generalizable in nature.

How do we make trees **more stable**, **accurate**, and **generalizable**?

The answer: **build a forest**!

A **Random Forest** is simply a collection of Decision Trees that work together to make predictions. This is done by creating a multiple trees, each getting slightly different data and features, and their votes are combined for a final, more reliable answer.

### The Wisdom of the Crowd
Imagine you’re trying to predict tomorrow’s weather.

- One person (a single tree) might be confident but wrong.
- A group of 100 people (a forest), each with different perspectives, can average their predictions.
- The collective judgment is often *more accurate*.

That’s the idea behind **ensemble learning**:

> Combining many weak models can produce a strong overall model.






#### How a Random Forest Works

1. **Bootstrap Sampling (aka Bagging):**  
   Each tree gets a **random sample** of the training data (*with replacement*).

2. **Random Feature Selection:**  
   At each split, the tree only considers a **random subset of features**.

3. **Independent Trees:**  
   Each tree...
   - learns its own patterns independently — no two trees are identical.
   - likely has mediocer performace as a predictive model.

5. **Voting (Classification) or Averaging (Regression):**  
   The forest combines all trees’ predictions for a more stable, accurate result.

Because no single tree sees all the data or features, their errors tend to balance out.

{% capture ex %}
```python
# --- Create data ---
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# --- Train models ---
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
forest = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

tree.fit(X, y)
forest.fit(X, y)

# --- Create decision grid ---
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
)
Z_tree = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_forest = forest.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(5, 8))
for ax, Z, title in zip(axes, [Z_tree, Z_forest],
                        ["Single Decision Tree", "Random Forest (100 Trees)"]):
    ax.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=30)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3)

plt.suptitle("Single Decision Tree vs Random Forest", fontsize=14)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_21_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">     

    
{% endcapture %}
{% include codeoutput.html content=ex %}  






The Random Forest’s decision boundary fits the data much better than a single tree becasue it combines the opinions of many slightly different trees.








### Comparing Accuracy:

{% capture ex %}
```python
# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree.fit(X_train, y_train)
forest.fit(X_train, y_train)

train_acc_tree = accuracy_score(y_train, tree.predict(X_train))
test_acc_tree = accuracy_score(y_test, tree.predict(X_test))
train_acc_forest = accuracy_score(y_train, forest.predict(X_train))
test_acc_forest = accuracy_score(y_test, forest.predict(X_test))

print("Model".ljust(25), "Train Acc", "Test Acc")
print("-" * 45)
print(f"Decision Tree:".ljust(25), f"{train_acc_tree:.2f}", f"{test_acc_tree:.2f}")
print(f"Random Forest:".ljust(25), f"{train_acc_forest:.2f}", f"{test_acc_forest:.2f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    Model                     Train Acc Test Acc
    ---------------------------------------------
    Decision Tree:            0.90 0.90
    Random Forest:            0.91 0.90
```
{% endcapture %}
{% include codeoutput.html content=ex %}  




You’ll ***typically*** see that the Random Forest:
- Performs *almost* as well as a single Decision Tree on the **training data**
- Performs **better** on **test data** due to less overfitting!







### Feature Importance — Understanding What the Model Learned

One of the best parts about Decision Trees and Random Forests is that they don’t just make predictions. They can tell us **why** they made them.

This is done through something called **feature importance**.






### What Is Feature Importance?

Each time a Decision Tree splits, it chooses the feature that reduces impurity (Gini or Entropy) the most. We can track **how much impurity each feature reduces** across all splits in all trees. This gives us a sense of which features matter most for making predictions.

In a Random Forest, we average this importance across all trees, producing a ranking of features by their **predictive power**.






### How Feature Importance Is Calculated (Conceptually)

| Method | Description | Pros | Cons |
|:--------|:-------------|:------|:------|
| **Mean Decrease in Impurity (MDI)** | Sum of impurity reduction from all splits using a feature. | Fast, built into training. | Can be biased toward features with many categories. |
| **Permutation Importance** | Randomly shuffle one feature and measure drop in model accuracy. | More accurate reflection of predictive power. | Slower; requires retraining or repeated predictions. |

Both methods tell us which features matter, but *permutation importance tends to be more robust*, especially when features are correlated.







### Why It Matters

Feature importance helps us:
- **Interpret** what the model has learned  
- **Identify the key drivers** of predictions  
- **Simplify models** by allowing us to remove uninformative features  
- **Guide domain understanding** — e.g., which factors matter most for predicting species, disease, or customer churn  








### Example: Feature Importance on the Iris Dataset

{% capture ex %}
```python
# --- Load dataset ---
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# --- Train Random Forest ---
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X, y)

# --- Get feature importances ---
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# --- Display results ---
plt.figure(figsize=(7,5))
plt.barh(range(len(importances)), importances[indices][::-1], align="center", color="teal")
plt.yticks(range(len(importances)), [feature_names[i] for i in indices][::-1])
plt.xlabel("Importance Score")
plt.title("Feature Importance in Random Forest (Iris Dataset)")
plt.grid(axis="x", alpha=0.4)
plt.show()

# Print table
print(pd.DataFrame({
    "Feature": [feature_names[i] for i in indices],
    "Importance": importances[indices]
}))

print("\n")

# --- Compute permutation importance ---
result = permutation_importance(forest, X, y, n_repeats=10, random_state=42)

# --- Plot ---
sorted_idx = result.importances_mean.argsort()
plt.figure(figsize=(7,5))
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(feature_names)[sorted_idx])
plt.title("Permutation Importance (Random Forest)")
plt.xlabel("Decrease in Model Accuracy When Shuffled")
plt.grid(alpha=0.3)
plt.show()

print(pd.DataFrame({
    "Feature": [feature_names[i] for i in indices],
    "Importance": result.importances_mean[indices]
}))
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_26_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

```python
                 Feature  Importance
    0  petal length (cm)    0.436130
    1   petal width (cm)    0.436065
    2  sepal length (cm)    0.106128
    3   sepal width (cm)    0.021678
```
    



<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_26_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">     
    

```python
                 Feature  Importance
    0  petal length (cm)    0.222667
    1   petal width (cm)    0.180667
    2  sepal length (cm)    0.014667
    3   sepal width (cm)    0.012667
```
{% endcapture %}
{% include codeoutput.html content=ex %}  




and we can compute permutation importance like this:

Here we can see that **petal length** and **petal width** were most useful for classifying flowers.




<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b>

- Feature importance quantifies **how much each variable contributes** to model accuracy.  
- Tree-based models are not “black boxes” — they can reveal which patterns drive decisions.  
- Random Forests average importance across trees, making results more **stable and reliable** than from a single tree.  
</div>










### Ensemble Variants — Bagging, Boosting, and the Bias–Variance Tradeoff

At this point, we’ve seen that combining multiple trees can dramatically improve performance and stability. But there’s more than one way to build an ensemble.

Random Forests use a method called **Bootstrapping (aka Bagging)**, which reduces variance.  Other methods, like **Boosting**, work differently.

Let’s unpack both ideas and see where they fit in the broader landscape of ensemble methods.





#### The Two Major Ensemble Strategies

| **Method** | **Core Idea** | **Goal** | **Example Algorithms** |
|:------------|:--------------|:---------|:-----------------------|
| **Bootstrapping (Bagging)** | Train many models independently on random subsets of data and average their predictions. | Reduce **variance** (stabilize predictions). | Random Forest, Bagged Trees |
| **Boosting** | Train models *sequentially*, each correcting the errors of the last one. | Reduce **bias** (improve weak models). | AdaBoost, Gradient Boosting, XGBoost |





#### Bagging: Reducing Variance with Randomization

Recall, in Bootstrapping (Bagging):
1. We draw many random samples (with replacement) from the training data.
2. We train a model — like a Decision Tree — on each sample.
3. We combine (average or vote) their results.

The randomn data samples helps prevent overfitting and smooths out noisy patterns. Additionally, Random Forests add an extra twist: they also randomize **which features** each tree can use.




#### Boosting: Reducing Bias with Sequential Learning

Boosting takes a totally different approach:
1. Start with a simple, weak model (e.g., a shallow tree).
2. Identify where it makes mistakes.
3. Train the next model to focus on those mistakes.
4. Repeat — each new model improves upon the previous ones.

The result is a “team” of models, each one correcting the last.

Boosting tends to produce *highly accurate* models, though it’s slower and more prone to overfitting.

{% capture ex %}
```python
# --- Simulate training complexity vs. error ---
complexity = np.linspace(1, 10, 100)
bias = (10 - complexity) ** 2 / 25
variance = complexity ** 2 / 50
total_error = bias + variance + 0.2

plt.figure(figsize=(8,5))
plt.plot(complexity, bias, label="Bias² (Underfitting)", linewidth=2)
plt.plot(complexity, variance, label="Variance (Overfitting)", linewidth=2)
plt.plot(complexity, total_error, label="Total Error", color="black", linestyle="--", linewidth=2)
plt.title("Bias–Variance Tradeoff")
plt.xlabel("Model Complexity →")
plt.ylabel("Error")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_31_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

{% endcapture %}
{% include codeoutput.html content=ex %}  





This curve summarizes an essential truth in machine learning:

- **Bias**: Error from overly simple models (e.g., shallow trees).  
- **Variance**: Error from overly complex models (e.g., deep, overfit trees).  
- **Total Error**: The combination — minimized at just the right level of complexity.

**Bagging** (like Random Forests) reduces variance.  
**Boosting** reduces bias.  
Together, these ideas cover most of modern ensemble learning.


Let's look at an exmple of a couple of Boost options:

{% capture ex %}
```python
# === Boosting in Practice: Decision Trees as Weak Learners ===
# Demonstrates AdaBoost and Gradient Boosting using shallow trees

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Create a noisy, non-linear binary dataset
X, y = make_classification(
    n_samples=600, n_features=2, n_redundant=0, n_clusters_per_class=1,
    class_sep=1.2, flip_y=0.15, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# Train a single shallow decision tree
base_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
base_tree.fit(X_train, y_train)
y_pred_tree = base_tree.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)


# Train boosted trees (AdaBoost and Gradient Boosting)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100, learning_rate=0.5, random_state=42
)
ada.fit(X_train, y_train)

gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42
)
gb.fit(X_train, y_train)

acc_ada = accuracy_score(y_test, ada.predict(X_test))
acc_gb = accuracy_score(y_test, gb.predict(X_test))

print(f"Single Shallow Tree Accuracy:  {acc_tree:.3f}")
print(f"AdaBoost Accuracy:            {acc_ada:.3f}")
print(f"Gradient Boosting Accuracy:   {acc_gb:.3f}")


# Visualize decision boundaries
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 300),
    np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 300)
)

def plot_boundary(ax, model, title):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap="coolwarm", edgecolor="k", s=25)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

fig, axes = plt.subplots(1, 3, figsize=(13,4))
plot_boundary(axes[0], base_tree, f"Single Tree (max_depth=1)\nAcc = {acc_tree:.2f}")
plot_boundary(axes[1], ada, f"AdaBoost (100 weak trees)\nAcc = {acc_ada:.2f}")
plot_boundary(axes[2], gb, f"Gradient Boosting (100 weak trees)\nAcc = {acc_gb:.2f}")

plt.tight_layout()
plt.show()


# visualize accuracy improvement with # of estimators
ada_scores, gb_scores = [], []
for n in range(1, 101):
    ada_temp = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n, learning_rate=0.5, random_state=42
    ).fit(X_train, y_train)
    gb_temp = GradientBoostingClassifier(
        n_estimators=n, learning_rate=0.1, max_depth=1, random_state=42
    ).fit(X_train, y_train)
    ada_scores.append(accuracy_score(y_test, ada_temp.predict(X_test)))
    gb_scores.append(accuracy_score(y_test, gb_temp.predict(X_test)))

plt.figure(figsize=(7,4))
plt.plot(range(1,101), ada_scores, label="AdaBoost", color="royalblue")
plt.plot(range(1,101), gb_scores, label="Gradient Boosting", color="orange")
plt.axhline(acc_tree, color="gray", linestyle="--", label="Single Tree")
plt.xlabel("Number of Weak Trees (n_estimators)")
plt.ylabel("Test Accuracy")
plt.title("Boosting: Accuracy vs Number of Weak Learners")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    Single Shallow Tree Accuracy:  0.867
    AdaBoost Accuracy:            0.883
    Gradient Boosting Accuracy:   0.906
```


<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_33_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">        


<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_33_2.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">        
    

{% endcapture %}
{% include codeoutput.html content=ex %}  






### Choosing the Right Tool

| **Scenario** | **Recommended Method** | **Why** |
|:--------------|:----------------------|:--------|
| High variance / noisy data | **Random Forest (Bagging)** | Averages out instability from individual trees. |
| High bias / underfitting | **Boosting (e.g., AdaBoost, Gradient Boosting)** | Sequentially improves simple models. |
| Fast, interpretable baseline | **Single Decision Tree** | Easy to visualize and explain results. |





#### Time and Complexity

- **Decision Trees:** Fast to train and easy to interpret.
- **Random Forests:** More accurate and stable, but slower and harder to explain.
- **Boosted Trees:** Often the best-performing models, but computationally heavier.

Always balance *accuracy* against *speed* and *interpretability* for your project.

<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b>

- Ensembles combine many weak models to create a strong learner.  
- **Bagging** (Random Forests) combats overfitting by averaging uncorrelated models.  
- **Boosting** builds models sequentially to correct previous errors.  
- The **bias–variance tradeoff** explains why we need ensembles in the first place.  
- Understanding both allows you to pick the right tool for your data and constraints.
</div>







#### Evaluation and Visualization

Let’s **evaluate** how well our Decision Trees and Random Forests perform and **visualize** what they’ve actually learned from the data.

Remember, we can’t rely on a single “accuracy” number alone. We want to understand *what kinds* of mistakes the model makes.

We’ll use out standard array of descriptive statisitcs:
- **Accuracy:** % of correctly predicted samples  
- **Precision:** Of those predicted positive, how many were correct?  
- **Recall:** Of the actual positives, how many did we catch?  
- **F1 Score:** Harmonic mean of precision and recall — balances both.

### Side Note: When Features Overlap

Highly correlated features can confuse models like trees and forests. Both Features may get similar importance scores, or the importance may appear “split” between them. One way to fix this is to use techniques like **PCA** or **Regularization** to combine correlated Features into single measurements.

{% capture ex %}
```python
# --- Load data ---
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=2
)

# --- Train Random Forest ---
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

# --- Compute metrics ---
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("=== Model Performance Metrics ===")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=iris.target_names)
disp.plot(cmap="Blues", values_format="d")
plt.grid(False)
plt.title("Confusion Matrix — Random Forest (Iris Dataset)")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    === Model Performance Metrics ===
    Accuracy:  0.978
    Precision: 0.976
    Recall:    0.978
    F1 Score:  0.976
    
    === Classification Report ===
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        17
      versicolor       1.00      0.93      0.97        15
       virginica       0.93      1.00      0.96        13
    
        accuracy                           0.98        45
       macro avg       0.98      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45
``` 



<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_36_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">        

{% endcapture %}
{% include codeoutput.html content=ex %}  







The confusion matrix shows exactly where the model succeeds and fails.






#### Visualizing Decision Boundaries

Let’s visualize how a tree or forest divides the feature space.

We’ll use two features from the Iris dataset so we can plot them directly.

{% capture ex %}
```python
# --- Select two features for visualization ---
X = iris.data[:, [0, 2]]   # sepal length, petal length
y = iris.target

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# --- Fit models ---
clf_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_forest = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
clf_tree.fit(X_train, y_train)
clf_forest.fit(X_train, y_train)

# --- Mesh grid ---
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min() - 0.5, X[:,0].max() + 0.5, 300),
    np.linspace(X[:,1].min() - 0.5, X[:,1].max() + 0.5, 300)
)

# --- Get decision regions ---
Z_tree = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_forest = clf_forest.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12,5))
for ax, Z, title in zip(
    axes,
    [Z_tree, Z_forest],
    ["Decision Tree Boundary", "Random Forest Boundary"]
):
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:,0], X[:,1], c=y, edgecolor="k", cmap="coolwarm", s=30)
    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Petal Length")
    ax.set_title(title)
plt.tight_layout()
plt.show()


print("\n")

# --- Get importances ---
importances = clf_forest.feature_importances_
feature_names = ["Sepal Length", "Petal Length"]

# --- Plot ---
plt.barh(feature_names, importances, color="teal")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.grid(axis="x", alpha=0.3)
plt.show()

# --- Display data frame ---
pd.DataFrame({"Feature": feature_names, "Importance": importances})



```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_38_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    


<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_38_2.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    



| Index | Feature        | Importance |
|------:|----------------|-----------:|
| 0     | Sepal Length  | 0.369273   |
| 1     | Petal Length  | 0.630727   |


{% endcapture %}
{% include codeoutput.html content=ex %}  









#### Partial Dependence Plots

Partial Dependence Plots (PDPs) help us see *how* a feature affects predictions, holding all other features constant. They’re especially useful for explaining Random Forests.

Each PDP shows how predicted probability changes as one feature varies. This can reveal trends and nonlinear relationships that might not be obvious from coefficients alone.

In multiclass problems, each class has its own predicted probability curve. We must tell scikit-learn *which class* we want to visualize using the `target` argument.

For example:
- `target=0` → setosa  
- `target=1` → versicolor  
- `target=2` → virginica  

This allows us to see how the probability of belonging to one class changes as features like petal length and sepal length vary.

{% capture ex %}
```python
# Load data and train Random Forest
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf_forest = RandomForestClassifier(
    n_estimators=200, random_state=42, max_depth=5
)
clf_forest.fit(X_train, y_train)


# Generate separate PDPs for each flower class
for class_idx, class_name in enumerate(iris.target_names):
    fig, ax = plt.subplots(figsize=(8, 5))
    display = PartialDependenceDisplay.from_estimator(
        clf_forest,
        X_train,
        features=[0, 2],  # Example: Sepal Length (0) and Petal Length (2)
        feature_names=iris.feature_names,
        target=class_idx,  # Probability surface for this class
        ax=ax
    )
    plt.suptitle(f"Partial Dependence Plot — Class: {class_name.capitalize()}", fontsize=14)
    plt.tight_layout()
    plt.show()
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_40_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_40_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_40_2.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
   
{% endcapture %}
{% include codeoutput.html content=ex %}  


 












## Summary: Decision Trees & Ensemble Methods

By now, you’ve learned that **Decision Trees** are both intuitive and powerful—models that split data into smaller and smaller regions to make predictions.  
But you’ve also seen how easily they can **overfit** when allowed to grow too deep.

**Key Ideas**
- **Splitting Criteria:** Trees use impurity measures like *Gini* or *Entropy* to choose the best split.
- **Overfitting & Pruning:** Unpruned trees memorize noise; limiting depth or leaf size improves generalization.
- **Ensembles:**
  - **Random Forests** build many randomized trees and average their predictions to reduce variance.  
  - **Bagging** draws multiple bootstrap samples and trains separate models.  
  - **Boosting** builds trees sequentially, focusing on correcting prior errors.
- **Feature Importance:** Random Forests reveal which variables most influence the prediction.
- **Partial Dependence:** We can visualize how specific features impact predicted probabilities.
- **Evaluation:** We assess performance using accuracy, precision, recall, F₁ score, and ROC/AUC.

**When to Use**
- When interpretability and feature importance matter.
- When you expect nonlinear relationships or mixed data types.
- When you need a strong, general-purpose baseline model.

**Limitations**
- Single trees overfit easily without pruning.
- Random Forests can be slow and less interpretable.
- Correlated features can “split” importance scores.

In practice, trees and ensembles often form the foundation for modern ML—  
they’re powerful on their own and serve as building blocks for advanced methods like Gradient Boosting and XGBoost.








## One-Cell Does it all code

{% capture ex %}
```python
# === Lecture 05: Decision Trees & Random Forest ===

# --- Imports ---
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# --- Load Data ---
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Fit Models ---
clf_tree = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
clf_forest = RandomForestClassifier(n_estimators=100, random_state=42)
clf_tree.fit(X_train, y_train)
clf_forest.fit(X_train, y_train)

# --- Predict & Evaluate ---
y_pred_tree = clf_tree.predict(X_test)
y_pred_forest = clf_forest.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

evaluate_model("Decision Tree", y_test, y_pred_tree)
evaluate_model("Random Forest", y_test, y_pred_forest)

# --- Confusion Matrix Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(10,4))

for ax, model_name, y_pred in zip(
    axes,
    ["Decision Tree", "Random Forest"],
    [y_pred_tree, y_pred_forest]
):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.grid(False)
    ax.set_title(f"{model_name} Confusion Matrix")

plt.tight_layout()
plt.show()

# --- Feature Importances (Seaborn ≥ 0.14 compatible) ---
importances = clf_forest.feature_importances_
imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True)

plt.figure(figsize=(7, 4))
sns.barplot(
    data=imp_df,
    x="Importance",
    y="Feature",
    hue="Feature",
    dodge=False,
    legend=False,
    palette="viridis"
)
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --- Partial Dependence Plot (Class = 2) ---
fig, ax = plt.subplots(figsize=(8,5))
PartialDependenceDisplay.from_estimator(
    clf_forest, X_train, features=[0,2],
    feature_names=feature_names, target=2, ax=ax
)
plt.title(f"Partial Dependence (Class = {class_names[2]})")
plt.tight_layout()
plt.show()

# --- Visualize Decision Tree Structure ---
plt.figure(figsize=(10,6))
plot_tree(clf_tree, filled=True, feature_names=feature_names, class_names=class_names, fontsize=8)
plt.title("Decision Tree (max_depth=3)")
plt.show()

# --- ROC Curve (One-vs-Rest) ---
y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_score = clf_forest.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr')

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.3f})", color="darkorange")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest, Multiclass OVR)")
plt.legend()
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    === Decision Tree ===
    Accuracy: 0.978
    Classification Report:
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        15
      versicolor       1.00      0.93      0.97        15
       virginica       0.94      1.00      0.97        15
    
        accuracy                           0.98        45
       macro avg       0.98      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45
    
    
    === Random Forest ===
    Accuracy: 0.889
    Classification Report:
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        15
      versicolor       0.78      0.93      0.85        15
       virginica       0.92      0.73      0.81        15
    
        accuracy                           0.89        45
       macro avg       0.90      0.89      0.89        45
    weighted avg       0.90      0.89      0.89        45
```  


<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_43_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_43_2.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_43_3.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_43_4.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    

    

<img
  src="{{ '/courses/machine-learning-foundations/images/lec05/output_43_5.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
   
{% endcapture %}
{% include codeoutput.html content=ex %}  


 
