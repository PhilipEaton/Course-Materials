---
layout: jupyternotebook
title: Machine Learning Foundations – Lecture 04
course_home: /courses/machine-learning-foundations/
nav_section: lectures
nav_order: 4
---

# Lecture 04: Logistic Regression and Naive Bayes Calssification

## Setup: Import Libraries

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
    make_classification
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





## Logistic Regression

We’ve learned how to use **linear regression** to predict *continuous* outcomes, like housing prices:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots
$$

Here, $ y $ might represent something like the price of a house or the number of cats a person owns (yes, crazy cat people are absolutely valid data points).

But now suppose we want to answer a *different* kind of question.

What if we’re not interested in predicting a number, but instead want to know *what kind* of thing we’re looking at?

In simpler terms, what if we want to predict a **category**, such as:
- Will a student **pass or fail**? (0 or 1)
- Will a customer **buy the product or not**? (yes / no)
- Is a penguin **male or female**?

If we try to use **linear regression** for this type of problem, we quickly run into trouble:

- can predict values less than 0 or greater than 1, which don’t make sense for categories.
- doesn’t naturally give us probabilities or a clear sense of how confident it is in a classification.
- relationships between the features $ x $ and a categorical outcome $ y $ are often nonlinear for categorization problems.

This is where **classification models**, models designed specifically to predict *categories* rather than continuous values, come into play.

Suppose we have a single feature and data points that belong to one of two categories. In that case, there should exist some point along that feature where we can make a **decision cut** and say:

- everything **below** this cut is **Category A**, and  
- everything **above** this cut is **Category B**.

This cut defines a **decision boundary** in feature space.  

Let’s look at a plot of this idea for reference:

{% capture ex %}
```python
# Generate fake binary data
np.random.seed(42)
X = np.linspace(0, 10, 50)
y = (X > 5).astype(int)  # 0 for X <= 5, 1 for X > 5

# Plot
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='gray', alpha=0.6, label='True Classes (0 or 1)')
plt.axhline(0, color='black', linestyle='--', alpha=0.6)
plt.axhline(1, color='black', linestyle='--', alpha=0.6)
plt.ylim(-0.5, 1.5)
plt.title("Decision Cut Plot")
plt.xlabel("Feature X")
plt.ylabel("Predicted y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_4_0.png' | relative_url }}"
  alt="A scatter plot with all points t the left of x equals 5 sit along y equals 0 and all points to the right of x equals 5 sit along the y equals 1 line. y euqlas 0 is one choice and y equals 1 is the other."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  





We need a model that:

1. Still uses a linear combination of features (so we can easily interpret coefficients), but  
2. “Squashes” the predictions into the range $[0, 1]$, colser to 0 is one option and closer to 1 is the other.

That’s where **Logistic Regression** comes in. It takes the linear output and wraps it in a *sigmoid* (or logistic) function:

$$
P(y = 1 | x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots)}}
$$

This function predicts the **probability** that $y = 1$, where $1$ could mean: 
- "Yes, the student will fail."
- "Yes, a customer will but the product."
- "The pedguin is female."



{% capture ex %}
```python
# === Define Sigmoid (Logistic) Function ===
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# === Plot Sigmoid (Logistic) Function ===
# Create a range of z values (the linear model output)
z = np.linspace(-10, 10, 200)
p = sigmoid(z)

# Plot the sigmoid curve
plt.figure(figsize=(7,5))
plt.plot(z, p, color="blue", linewidth=2)
plt.axvline(0, color="gray", linestyle="--", alpha=0.8)
plt.axhline(0.5, color="gray", linestyle="--", alpha=0.8)
plt.title("The Sigmoid Function")
plt.xlabel("Linear Model Output (z = β₀ + β₁x)")
plt.ylabel("Predicted Probability P(y=1|x)")
plt.text(6, 0.925, "P ≈ 1 → ", fontsize=11)
plt.text(-9, 0.055, "← P ≈ 0 ", fontsize=11)
plt.text(-1, 0.55, "Decision boundary (P=0.5)", fontsize=10, color="gray", rotation = 90)
plt.text(-9.5, 0.45, "P = 0.5 line", fontsize=10, color="gray")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
 
 <img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_6_0.png' | relative_url }}"
  alt="A plot of a logistic curve."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  




   







### Fitting a Logistic Model

Let’s see how logistic regression actually *works in practice*.

We’ll use a simple example where the outcome is binary: 0 or 1.

Our goals:
1. Fit a logistic regression model to data.
    - This is done by minmizing the sum of the squared error (SSE) or, more commonly, maximizing the likelihood of the resulting model.
    - We are not going to talk about this in detail since it is a bit technical and not needed to be able to use the tool effectively.
3. Interpret what the coefficients mean.  
4. Visualize the decision boundary.  
5. Evaluate the model’s performance.


{% capture ex %}
```python
# Synthetic binary classification data
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, flip_y=0.05, class_sep=1.5, random_state=42)

df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
df["Target"] = y

# Visualize the dataset
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="Feature_1", y="Feature_2", hue="Target", palette="coolwarm", s=60)
plt.title("Synthetic Binary Classification Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Target")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_8_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  





{% capture ex %}
```python
# Separate Features and Target
X_features = df[["Feature_1", "Feature_2"]]
y_target = df["Target"]

# Fit Logistic Regression 
log_reg = LogisticRegression()
log_reg.fit(X_features, y_target)

# Coefficients and intercept
print("Intercept (β₀):", log_reg.intercept_)
print("Coefficients (β):", log_reg.coef_)

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
Intercept ($\beta_0$): [0.72487538]
Coefficients ($\beta_i$): [[-0.48899191  2.033306  ]]
{% endcapture %}
{% include codeoutput.html content=ex %}  





###  Interpreting Coefficients

If we take the logistic equation, simplify a little by calling the probability of getting a 1 the letter $p$:

$$ 
P(y = 1 | x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots)}} \quad\longrightarrow\quad p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots)}}
$$

and solve for the feature section of the equation, $\beta_0 + \beta_1 x$, we get:

$$
\ln\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots 
$$

which gives the definition

$$
\text{logit}(p) = \ln\!\left(\frac{p}{1-p}\right)
$$

The function $\ln\left(\frac{p}{1-p}\right)$ is called the **log-odds**, since it is:
- the probability of it being 1 (which we called $p$), divided by 
- the prabability it is not 1 (which would be $1-p$). 

The "log" in the name comes from taking the natrual logarithm of the odds.

From this we can see that the logistic regression coefficients represent how each feature affects the **log-odds** of belonging to class 1 (whatever we determine that classification to be).

$$
\ln\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots
$$

- A **positive** coefficient means increasing that feature increases the probability of class 1.  
- A **negative** coefficient means increasing that feature decreases that probability.  

We can exponentiate each coefficient, eliminating the natural log as a result, to get:

$$
\left(\frac{p}{1-p}\right) = e^{\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots} = e^{\beta_0} \, e^{\beta_1 x_1}\, e^{\beta_2 x_2} \cdots
$$

where we call the exponentials of just the coefficients are called the **odds ratio**:

$$
\text{Odds Ratio} = e^{\beta_i}
$$

- An odds ratio **greater than 1** means increasing that feature increases the probability of class 1.  
- An odds ratio **less than 1** means increasing that feature decreases that probability.  

Either way you decide to go, the interpretation will be the same. Pick your poison in this case.

{% capture ex %}
```python
# Display odds ratios for easier interpretation
odds_ratios = np.exp(log_reg.coef_[0])
for feature, coef, odds in zip(X_features.columns, log_reg.coef_[0], odds_ratios):
    print(f"{feature}: coefficient = {coef:.3f}, odds ratio = {odds:.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
Feature_1: coefficient = -0.489, odds ratio = 0.613  
Feature_2: coefficient = 2.033, odds ratio = 7.639
{% endcapture %}
{% include codeoutput.html content=ex %}  








### Decision Boundary

Recall, a **decision boundary** is the line (in 2D) or surface (in higher dimensions) which divides the sapce between the options points will take if placed in each region. In the case of a logistic classifier, this line is drawn where the classifier is **indifferent** between classes.

For **logistic regression**, this is where the predicted probability is exactly **0.5** (for binary classification) — on one side, $P(y=1 \mid \mathbf{x})>0.5$; on the other, $P(y=1 \mid \mathbf{x})<0.5$.

To build intuition, we can overlay a **probability heatmap** on top of our scatter plot: darker regions indicate lower $P(y=1)$, lighter regions indicate higher $P(y=1)$. The **decision boundary** appears as the contour where $P(y=1)=0.5$.


{% capture ex %}
```python
# === Decision Boundary Visualization ===

# Create a mesh grid over the feature space
x_min, x_max = X_features["Feature_1"].min() - 1, X_features["Feature_1"].max() + 1
y_min, y_max = X_features["Feature_2"].min() - 1, X_features["Feature_2"].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict probabilities for each grid point
Z = log_reg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, levels=20, cmap="viridis", alpha=0.6)
sns.scatterplot(data=df, x="Feature_1", y="Feature_2", hue="Target",
                palette="coolwarm", s=60, edgecolor="black")
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Predicted P(y=1)")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_13_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  



    

<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b> 

- Logistic regression is linear in the features but nonlinear in the output.  
- The model outputs probabilities, not raw scores.  
- Coefficients describe how features change the **log-odds** of the target class.  
- The decision boundary occurs where $P(y=1)=0.5$, i.e. where $\beta_0 + \beta_1x_1 + \beta_2x_2 = 0$.  
- Visualizing both the probabilities and the boundary helps you see how the model separates the classes.
</div>

Next: we’ll evaluate how *well* our logistic model performs — accuracy, precision, recall, and ROC curves.










### Evaluating Logistic Regression Models

With linear regression, we evaluated how well our model fit the data using metrics like **MSE** and **$R^2$**.

For **logistic regression**, our goal is not to predict a *continuous* number, but to correctly *classify* observations into categories (e.g., 0 vs. 1).

We’ll use our old classification metrics from k-NN and k-Means:
- How many predictions are **correct** (Accuracy)
- How well we identify **positives** (Precision, Recall)
- How balanced our performance is (F1 Score)  

and a new metric:
- How well our model ranks cases (ROC curve & AUC)


{% capture ex %}
```python

# --- Predict classes and probabilities ---
y_pred = log_reg.predict(X_features)
y_proba = log_reg.predict_proba(X_features)[:, 1]

# --- Compute metrics ---
acc = accuracy_score(y_target, y_pred)
prec = precision_score(y_target, y_pred)
rec = recall_score(y_target, y_pred)
f1 = f1_score(y_target, y_pred)
auc = roc_auc_score(y_target, y_proba)

print("=== Logistic Regression Model Performance ===")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"AUC:       {auc:.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    === Logistic Regression Model Performance ===
    Accuracy:  0.915
    Precision: 0.930
    Recall:    0.903
    F1 Score:  0.916
    AUC:       0.964
```
{% endcapture %}
{% include codeoutput.html content=ex %}  



We can also still use the confusion matrix, since this is a classification model:


{% capture ex %}
```python
# --- Confusion Matrix Visualization ---
cm = confusion_matrix(y_target, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_)

# Plot the confusion matrix
disp.plot(cmap="Blues", values_format='d')

# Remove gridlines
plt.grid(False)

# Optional: clean up tick marks and frame
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_18_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

{% endcapture %}
{% include codeoutput.html content=ex %}  









#### Exploring the Threshold


When a logistic regression model makes predictions, it doesn’t actually say “Class 0” or “Class 1” directly. Instead, it outputs a **probability** — for example, “there’s a 0.72 (72%) chance this sample belongs to Class 1.”  

By default, we classify a data point as **Class 1** if its probability ≥ 0.5.  

But that 0.5 cutoff is completely arbitrary — and changing it changes how the model behaves.


##### **Why Explore the Threshold?**

Adjusting the classification threshold helps balance **precision** and **recall**:

- Lower thresholds (e.g., 0.3) classify more points as positive → **higher recall**, **lower precision**  
- Higher thresholds (e.g., 0.7) classify fewer points as positive → **higher precision**, **lower recall**

Depending on the problem, one may matter more than the other:
- In **medical screening**, we lower the threshold to catch every possible case (maximize recall).
- In **fraud detection**, we raise the threshold to avoid flagging legitimate transactions (maximize precision).

##### **How to Explore the Threshold**

You can visualize how model performance changes as you move the threshold using either:
- The **ROC curve**, which shows the trade-off between True Positive Rate and False Positive Rate.
- The **Precision–Recall curve**, which shows the trade-off between precision and recall directly.

Both help you choose the best operating point for your model — not just rely on the default 0.5 cutoff.


{% capture ex %}
```python
# --- Exploring the Threshold ---
thresholds = np.linspace(0, 1, 101)
precisions, recalls, f1s = [], [], []

for t in thresholds:
    preds = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_target, preds))
    recalls.append(recall_score(y_target, preds))
    f1s.append(f1_score(y_target, preds))

# --- Plot Precision, Recall, and F1 vs. Threshold ---
plt.figure(figsize=(8,6))
plt.plot(thresholds, precisions, label="Precision", linewidth=2)
plt.plot(thresholds, recalls, label="Recall", linewidth=2)
plt.plot(thresholds, f1s, label="F1 Score", linewidth=2, linestyle="--", color="purple")
plt.title("Effect of Decision Threshold on Precision, Recall, and F1 Score")
plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
  
<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_20_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  



  

A good choice would be the threshold that maximizes the F1-score. However, this really depends on what your objectives are! 

Though, some good rules of thumb would be:

| Goal                                        | Best Threshold Choice                                 |
| ------------------------------------------- | ----------------------------------------------------- |
| Maximize balance between precision & recall | Threshold where **F1** is highest                     |
| Minimize false negatives                    | Lower threshold (shift toward higher recall)          |
| Minimize false positives                    | Higher threshold (shift toward higher precision)      |
| You’re not sure                             | Start where precision ≈ recall, then adjust as needed |


{% capture ex %}
```python
# --- Exploring Decision Threshold Effects (Self-contained Demo) ---

# --- Generate synthetic 2D classification data ---
X, y_target = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    flip_y=0.05,
    class_sep=1.5,
    random_state=42
)

# --- Train a logistic regression model ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_target, test_size=0.3, random_state=42
)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# --- Get predicted probabilities for the positive class ---
y_proba = log_reg.predict_proba(X_test)[:, 1]

# --- Explore thresholds ---
example_thresholds = [0.1, 0.5, 0.9]

# --- Create subplots ---
fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True, sharey=True)

for ax, t in zip(axes, example_thresholds):
    preds = (y_proba >= t).astype(int)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"Threshold = {t:.2f} | Precision = {prec:.2f} | Recall = {rec:.2f} | F1 = {f1:.2f}")
    
    # --- Scatter plot of predictions ---
    ax.scatter(X_test[:, 0], X_test[:, 1], c=preds, cmap="coolwarm", alpha=0.7, edgecolor="k")
    ax.set_title(f"Threshold = {t:.1f}\nPrec={prec:.2f}, Rec={rec:.2f}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.suptitle("Effect of Changing Decision Threshold (Logistic Regression)", fontsize=14)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

Threshold = 0.10 | Precision = 0.76 | Recall = 0.94 | F1 = 0.84
Threshold = 0.50 | Precision = 0.96 | Recall = 0.87 | F1 = 0.92
Threshold = 0.90 | Precision = 1.00 | Recall = 0.74 | F1 = 0.85


<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_22_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  







<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #8EC9DC;
    padding: 14px;
    border-radius: 6px;
">
<b style="color:#1b4965;">Professional Practice</b>  
<br>

- Never accept the default threshold blindly — **choose it based on your use-case**.  
- When presenting results, **report the chosen threshold** and justify why.  
- Use domain-specific metrics (like F1 score or cost-weighted accuracy) to select the optimal cutoff.  
- For critical applications, perform a **threshold sweep** and present how precision, recall, and F1 change with threshold.

> A good data scientist doesn’t just train a model — they decide **how it will be used** in the real world.
</div>







#### Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)

When evaluating **classification models**, especially those that output **probabilities** (like logistic regression and Naive Bayes), it’s not enough to just look at accuracy.  
Accuracy depends on the **classification threshold** (often 0.5 by default), but what if we want to see how our model performs **across all possible thresholds**?

That’s where the **ROC curve** and **AUC** come in.


##### **1. What is the ROC Curve?**

ROC stands for **Receiver Operating Characteristic**.  It’s a graphical representation of a model’s performance as the decision threshold varies.

The ROC curve plots:

- **True Positive Rate (TPR)** — also known as **Recall** or **Sensitivity**  
  
  $$
  \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$
  
  where TP = True Positive and FN = False Negative. This just tells us of all the actual positives given, how many did we correctly identify.

- **False Positive Rate (FPR)** — the proportion of negatives incorrectly labeled as positives  
  
  $$
  \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
  $$
  
  where FP = False Positive and TN = True Negative. This just tells us of all the actual negatives given, how many did we mistakenly call positive.


###### For example, consider the following confusion matrix:

{% capture ex %}
```python
# --- Confusion Matrix Visualization ---
cm = confusion_matrix(y_target, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_)

# Plot the confusion matrix
disp.plot(cmap="Blues", values_format='d')

# Remove gridlines
plt.grid(False)

# Optional: clean up tick marks and frame
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_25_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  



    


We can collect the following data:  
- TP = 90
- TN = 93
- FP = 7
- FN = 10

In this case the TPR and FPR would be:

$$
\text{TPR} = \frac{90}{90 + 10} = 0.90 \qquad\qquad \text{FPR} = \frac{7}{7 + 93} = 0.07 
$$

This would be plotted in a graph with the TRP on ther vertical and the FPR on the horizontal. 

Each point on the ROC curve corresponds to a different **classification threshold**.

- A **lower threshold** → more positive predictions → higher TPR *and* higher FPR  
- A **higher threshold** → fewer positive predictions → lower TPR *and* lower FPR


{% capture ex %}
```python
# Make synthetic classified data
X, y = make_classification(
    n_samples=2000, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, flip_y=0.05, class_sep=1.5, random_state=42
)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train logistic regression model ---
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

# --- Predict probabilities for the positive class ---
y_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

# --- Compute ROC curve and AUC ---
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# --- Plot ROC Curve ---
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve for Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_27_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    
{% endcapture %}
{% include codeoutput.html content=ex %}  



    



##### **2. Interpreting the ROC Curve**

- The **closer the curve is to the top-left corner**, the better the model is at distinguishing between classes.  
- A **diagonal line (y = x)** represents random guessing, our baseline — a model with **no discriminative power**.  

**Visually**:
- Random Guess → diagonal line
- Better Model → bowing toward the upper left
- Perfect Model → reaches top-left corner

##### **3. Area Under the Curve (AUC)**

The **Area Under the Curve (AUC)** quantifies the ROC curve into a single number:

$$
0 \leq \text{AUC} \leq 1
$$

by claculating the area under the ROC curve. A perfect model would make a right angle in the top left courner giveit it an area of 1 (horizontal length = 1, vertical length = 1; area = 1$\times$1 = 1). Non-perfect models will deviate below this and give a smaller area under the curve. 

The following is a good breakdown of how the AUC can be interpreted:
- **AUC = 1.0** → Perfect separation (likely overfitting!)
- **AUC > 0.9** → Excellent
- **AUC = 0.8–0.9** → Good
- **AUC = 0.7–0.8** → Fair    
- **AUC = 0.5** → Model performs no better than random chance  

Intuitively, the AUC represents the probability a randomly chosen example of class 1 is given a higher prababilty than a randomly chosen class 0 data point.

The AUC of the ROC is useful because:

- It’s **independent of the classification threshold**.  
- It works well for **imbalanced datasets**, where accuracy can be misleading.  
- It focuses on the model’s ability to **rank predictions correctly**, not just label them.


<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #8EC9DC;
    padding: 14px;
    border-radius: 6px;
">
<b style="color:#1b4965;">Professional Practice</b>  
<br><br>

| Metric | Description | Ideal Value | When to Focus |
|:--|:--|:--:|:--|
| **Accuracy** | % of total predictions that are correct | 1.0 | Classes are balanced and misclassification costs are similar |
| **Precision** | % of predicted positives that are truly positive | 1.0 | False positives are costly (e.g., flagging legit emails as spam) |
| **Recall** | % of true positives correctly identified | 1.0 | False negatives are costly (e.g., missing a disease case) |
| **F1 Score** | Harmonic mean of precision and recall | 1.0 | Need a single number that balances precision & recall |
| **ROC Curve** | TPR vs. FPR across thresholds | N/A | Comparing rank-ordering quality across models |
| **AUC** | Probability the model ranks a random positive above a random negative | 1.0 | Comparing overall discriminative ability, threshold-free |

**Key idea:** No single metric tells the whole story. Always interpret metrics **together** and in the **business context**.

---

###### Practical habits that scale

- **Always set a baseline.** Compare against `DummyClassifier` (e.g., most-frequent) to ensure you’re beating a trivial strategy.
- **Pick a threshold on purpose.** Default 0.5 is rarely optimal. Choose it using:
  - **Precision–Recall trade-off** (maximize F1, or meet a minimum precision/recall)
  - **Cost-based thresholding** (pick the threshold that minimizes expected cost given FP/FN costs)
- **Report both threshold-dependent _and_ threshold-free metrics.**  
  Example: Accuracy/F1 **and** ROC–AUC (or PR–AUC for imbalanced data).
- **Mind class imbalance.** Accuracy can look great while recall is terrible. Prefer **Precision-Recall–AUC**, **recall**, or **F1** in rare-event problems.
- **Calibrate probabilities** when actions depend on scores (e.g., risk). Use **Platt scaling** or **isotonic regression** (`CalibratedClassifierCV`) and check **calibration curves**. (Look into this on your own.)
- **Use cross-validation** for reliable estimates; report mean ± std over folds.
- **Add uncertainty.** Where stakes are high, include **bootstrap CIs** for AUC, precision, recall, etc.
- **Show the confusion matrix** at your chosen threshold; it makes trade-offs concrete.
- **Document your operating point.** Record the chosen threshold, rationale, and the expected FP/FN trade-offs for stakeholders.

---

###### Quick “how-to” checklist (scikit-learn)
- Predicted probabilities: `y_proba = model.predict_proba(X)[:,1]`
- ROC–AUC: `roc_auc_score(y_true, y_proba)`
- PR–AUC: `average_precision_score(y_true, y_proba)`
- Choose threshold `t`: `y_pred = (y_proba >= t).astype(int)`
- Confusion matrix & report: `confusion_matrix(...)`, `classification_report(...)`
- Probability calibration: `CalibratedClassifierCV(model, method="isotonic" or "sigmoid")`

</div>


<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b> 

- Logistic regression outputs **probabilities**, not hard classifications.  
- By default, sklearn classifies as class 1 if $ P(y=1) > 0.5 $, but you can, and generally should, adjust and justify the threshold.  
- Evaluate models using **multiple metrics** — especially when classes are imbalanced.  
- **ROC curves** and **AUC** summarize the model’s overall ranking ability, independent of any threshold.
</div>

Next, we’ll move into *multi-class logistic regression* and *regularization (L1/L2)* — powerful tools for larger, more complex datasets.











## Multi-Class Logistic Regression and Regularization

So far, we’ve used logistic regression for **binary classification** — deciding between two outcomes (e.g., 0 vs. 1).

But what if we have **more than two classes**?

For example:
- Classifying **iris flowers** into *setosa*, *versicolor*, or *virginica*  
- Categorizing emails into *spam*, *promotional*, or *primary*

Logistic regression can easily handle these cases using one of two strategies:

1. **One-vs-Rest (OvR)**: Train one model per class. Each model predicts whether an observation belongs to *that* class or not.  
2. **Multinomial (Softmax)**: A single model learns probabilities for *all* classes simultaneously.

Scikit-learn automatically uses **multinomial** when appropriate, which is what we’ll explore next.


{% capture ex %}
```python
# --- Load data ---
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Split and scale ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train multinomial logistic regression ---
multi_log = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
multi_log.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_pred = multi_log.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues", values_format='d')

# Remove gridlines
plt.grid(False)

# Set title
plt.title("Confusion Matrix: Multinomial Logistic Regression on Iris Data")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
                precision    recall  f1-score   support

        setosa       1.00      1.00      1.00        15
    versicolor       0.82      0.93      0.88        15
    virginica       0.92      0.80      0.86        15

    accuracy                           0.91        45
    macro avg       0.92      0.91      0.91        45
weighted avg       0.92      0.91      0.91        45
```    

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_32_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    


{% endcapture %}
{% include codeoutput.html content=ex %}  





    

### Regularization: Avoiding Overfitting


When a model becomes too flexible or learns too precisely from the training data, it can start to **overfit** — meaning it memorizes random noise and quirks of the dataset instead of learning the true underlying pattern.  As a result, it will perform well on the training data but poorly on new, unseen data.

**Regularization** is a technique to prevent overfitting.  

It works by **discouraging overly large model weights**, which are often a sign that the model is relying too heavily on a few specific features or data points. We do this by adding a *penalty term* to the model’s loss function. 

We will not get into the specifics of this; suffice to say, we add a penalty for adding complexity to our model. Think of it like telling the model:  

> “Yes, fit the data — but don’t work too hard at it.”


#### Common Types of Regularization

| Type | Penalty Term | Effect | Intuition |
|:-----|:--------------|:-------|:-----------|
| **L1 (Lasso)** | $$ C \sum |w_i| $$ | Encourages **sparsity** — some weights are driven exactly to 0. | This effectively performs **feature selection** by removing unimportant predictors. |
| **L2 (Ridge)** | $$ C \sum w_i^2 $$ | Keeps all weights **small but nonzero**, leading to a smoother model. | The model spreads its “attention” across all features instead of focusing too heavily on any single one. |

The parameter **`C`** (more generally called lambda, $\lambda$) controls how strong the penalty is. The `C` parameter is the *inverse* of regularization strength:

- Large `C` → weak regularization → the model can overfit (focuses too much on the training data)  
- Small `C` → strong regularization → the model may underfit (too simple, misses patterns)

Goal: Find the sweet spot.

#### Example: Logistic Regression
In `LogisticRegression` from scikit-learn:

{% capture ex %}
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, penalty='l2')  # L2 regularization by default
```
{% endcapture %}
{% include codeinput.html content=ex %}  


If your model overfits, try:
- decreasing C (stronger regularization), or
- switching from 'l2' to 'l1' (if your solver supports it) to encourage sparsity.

Regularization helps keep your model balanced — complex enough to learn real relationships, but simple enough to generalize well to new data.


{% capture ex %}
```python
# --- Compare models with different regularization strengths ---
C_values = [0.01, 0.1, 1, 10]
train_scores = []
test_scores = []

for C in C_values:
    log_reg_reg = LogisticRegression(C=C, multi_class="multinomial", solver="lbfgs", max_iter=1000)
    log_reg_reg.fit(X_train_scaled, y_train)
    train_scores.append(log_reg_reg.score(X_train_scaled, y_train))
    test_scores.append(log_reg_reg.score(X_test_scaled, y_test))

# --- Plot performance vs regularization strength ---
plt.figure(figsize=(7,5))
plt.plot(C_values, train_scores, marker="o", label="Training Accuracy")
plt.plot(C_values, test_scores, marker="o", label="Test Accuracy")
plt.xscale("log")
plt.xlabel("C (Inverse Regularization Strength)")
plt.ylabel("Accuracy")
plt.title("Effect of Regularization Strength on Model Performance")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_34_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

{% endcapture %}
{% include codeoutput.html content=ex %}  




<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b> 

- Logistic regression extends naturally to **multi-class problems** via *One-vs-Rest* or *Softmax*.
- **Regularization** keeps weights small and prevents overfitting.
- **C** controls the strength of regularization — find the sweet spot where the model generalizes best.
- Regularization is one of the most powerful tools for improving **model robustness** and **interpretability**.
</div>













## Naïve Bayes - Classification via Probabilities

We have now explored three classification methods: **k-Nearest Neighbors**, **k-Means Clustering**, and **Regression models**. Each of those algorithms gave us different *ways of thinking* about how to make predictions:

- **k-NN** looked at the *geometry* of the data (how close a point is to its neighbors).  
- **k-Means** looked for *groups* hidden in unlabeled data (what groupings are potentially hidded in the data).  
- **Logistic Regression** learned a *boundary* that separates classes.  

Let's take yet another perspective: **probability**.  

Specifically, let's ask questions like:
> “Given this new data point, what’s the probability it belongs to each class?”

> “If we assume each feature gives us a small clue, how do we combine them into a full prediction?”

That’s where **Naïve Bayes Classification** comes in. This has less to do with modeling lines and what not and are more interested in **how confident** we are about a prediction.

For example, imagine we’re building an email spam detector:
- Some words (like *“lottery”*) are strong spam signals.
- Others (like *“meeting”*) are more typical of normal mail.
- Each word gives us a hint — a *probability*.

Naïve Bayes takes these small clues from each feature and combines them into a big, simple prediction:
> “What’s the probability this email is spam given all the words inside it?”

This kind of reasoning is built on **Bayes’ Theorem**, which connects *what we already know* with *what we just observed.*


Let's begin by learning about prabability and Bayes' Theorem.











### Bayes’ Theorem — The Foundation of Probabilistic Reasoning


Before we jump right into using a Naive Bayes classifier, we need to understand **Bayes’ Theorem**,  which is the mathematical foundation of probabilistic classification.

At its heart, Bayes’ Theorem answers this question:

> “Given that we observed some evidence, what is the probability that a certain hypothesis is true?”

Written mathematically, this can be sxpressed as

$$
P(H\vert E) = \frac{P(E|H) \cdot P(H)}{P(E)}
$$

where:
- $ P(H\vert E) $ = *Posterior Probability*
    - How likely the hypothesis is given the evidence.  
- $ P(E\vert H) $ = *Likelihood*
    - How likely we are to see this evidence if the hypothesis were true.  
- $ P(H) $ = *Prior Probability*
    - How likely the hypothesis was before seeing the evidence.  
- $ P(E) $ = *Evidence*
    - How likely this evidence is under all possible hypotheses.

#### Notational Comment

The vertical bar "$\vert$" means "given". 

So, $H\vert E$ reads as "$H$ given $E$" and it taken to mean "$H$ occures given $E$ happened."


#### Example


Suppose:
- 1% of emails are spam.
    >  This directly tells us: $ P(\text{Spam}) = 0.01 $
- 90% of spam emails contain the word “lottery”
    > This is the chance of seeing the word "lottery" in the email given that is is spam:  → $ P(\text{Lottery}\vert \text{Spam}) = 0.9 $
- 5% of non-spam emails contain “lottery”
  > This is the chance of seeing the word "lottery" in the email given that is is NOT spam: → $ P(\text{Lottery}\vert \text{Not Spam}) = 0.05 $

This means the probability to seeing the word "lottery" in an email will be:

$$
P(\text{Lottery}) = P(\text{Lottery}\vert \text{Spam}) P(\text{Spam}) + P(\text{Lottery}\vert \text{Not Spam}) P(\text{Not Spam}) = (0.9) (0.01) + (0.05) (0.99) = 0.0585
$$

Now we can ask what is the chance an email is spam given the word "lottery" is included ($P(\text{Spam}\vert\text{Lottery})$). According to Bayes' Theorem:

$$
P(\text{Spam}\vert \text{Lottery}) = 
\frac{P(\text{Lottery}\vert \text{Spam}) P(\text{Spam})}{P(\text{Lottery})} = \frac{(0.9)(0.01)}{0.0585} = 0.154
$$

This gives us a *probability* that the email is spam as 15.4%, not a simple yes/no answer. Even though “lottery” appears very often in spam, the word alone gives only a 15.4% chance that the email is spam — far from a sure thing.


{% capture ex %}
```python
# === Visualizing Bayes' Theorem Concept ===

# Create a range for the "evidence" variable
x = np.linspace(-6, 10, 400)

# Define two hypotheses (classes) with normal distributions
prior_A = 0.5
prior_B = 0.5

# Likelihoods (how evidence is distributed given each class)
likelihood_A = norm.pdf(x, loc=2, scale=1.2)   # Class A centered near 2
likelihood_B = norm.pdf(x, loc=6, scale=1.5)   # Class B centered near 6

# Combine into overall evidence
evidence = likelihood_A * prior_A + likelihood_B * prior_B

# Compute posteriors using Bayes' theorem
posterior_A = (likelihood_A * prior_A) / evidence
posterior_B = (likelihood_B * prior_B) / evidence

# Plot
fig, axes = plt.subplots(2, 1, figsize=(8,8), sharex=True)

# Top: Likelihoods
axes[0].plot(x, likelihood_A, 'b--', label="P(E|A): Likelihood of Evidence if Class A")
axes[0].plot(x, likelihood_B, 'r--', label="P(E|B): Likelihood of Evidence if Class B")
axes[0].set_title("Likelihoods (Evidence Distributions by Class)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Bottom: Posterior Probabilities
axes[1].plot(x, posterior_A, 'b', linewidth=2, label="Posterior: P(A|E)")
axes[1].plot(x, posterior_B, 'r', linewidth=2, label="Posterior: P(B|E)")
axes[1].axvline(4, color='k', linestyle=':', label="Decision Boundary")
axes[1].set_title("Posterior Probabilities (After Seeing Evidence)")
axes[1].set_xlabel("Evidence (x)")
axes[1].set_ylabel("Probability")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_38_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
   
{% endcapture %}
{% include codeoutput.html content=ex %}  


 


#### Understanding the Visualization

In the **top plot**, we see two overlapping curves:
- The **blue dashed** curve shows how likely the evidence is if the data came from *Class A*.
- The **red dashed** curve shows the same for *Class B*.

This is our *likelihood* — how each class “expects” the evidence to look.


In the **bottom plot**, we see how Bayes’ theorem combines those likelihoods 
with our *priors* to produce *posterior probabilities*:

- On the left side (where evidence looks like Class A), $ P(A|E) $ is high.  
- On the right side (where evidence looks like Class B), $ P(B|E) $ is high.  
- Around the center, both probabilities are similar — this is our **decision boundary**.

This is what all probabilistic classifiers (including Naïve Bayes and Logistic Regression) do behind the scenes:
> They compute the likelihood of the data under each class and then choose the class with the highest posterior probability.


<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion Prompt:</b> 
<br>

1. **Priors:**  
   - Priors represent what we believe *before* seeing any evidence.  
   - Changing them shifts our conclusions — even if the data doesn’t change.


2. **Likelihood:**  
   - Describes how each class "expects" evidence to appear.  
   - For example, if penguins of one species tend to have longer bills, then "long bill" has a high likelihood for that class.


3. **Posterior:**  
   - Combines priors and likelihoods to give us updated belief after seeing the data.
</div>



#### Key Idea
All of machine learning is, in some sense, a form of **informed updating**:
> Start with what you believe (priors),  
> see new data (evidence),  
> and update your belief (posterior).  

This is literally the wy humans build confidence in ideas and beliefs!

Naïve Bayes simply applies this process **for every feature**, assuming each feature contributes independently.








#### Example: Bayesian Updating with a Weighted Coin


This example demonstrates **Bayesian inference** in action using a simple coin-flip experiment. Our goal is to decide whether a coin is **fair** (50/50) or **weighted** (biased toward heads), based on observed flips.


##### The Scenario

We flip a coin multiple times and keep track of how many heads (H) and tails (T) we observe. At the start, we don’t know if the coin is fair or weighted, so we assign a **prior belief**:

$$
P(\text{weighted}) = 0.5 \quad\text{and}\quad P(\text{fair}) = 0.5
$$


##### The Bayesian Update Process

For each coin flip:

1. Compute how likely the flip is under both models:
   - Fair coin: $ P(x\vert\text{fair}) = 0.5 $
   - Weighted coin: $ P(x\vert\text{weighted}) = p_{\text{predictive}} $
       - $ p_{\text{predictive}} $ comes from some underlying probability distribution.
       - In this case it will follow a **Beta distribution**, but we shouldn't get caught up in the details here.   

2. Update the **Bayes factor (evidence ratio)** comparing how much more likely the data are under the weighted model than the fair model.

3. Use this to update the **posterior model probability**:
   $$
   P(\text{weighted} \vert \text{data}) = \frac{P(\text{weighted}) \times \text{Bayes Factor}}{1 + P(\text{weighted}) \times \text{Bayes Factor}}
   $$

4. Update the Beta posterior for $p$ given the data at hand.


{% capture ex %}
```python
# === Bayesian coin demo with automatic interval-based progress printouts ===
np.random.seed(0)

# --- Settings you can tweak ---
n_flips       = 300
p_true        = 0.60          # True coin bias (unknown to the model)
pi_weighted   = 0.5           # Prior P(weighted)
a0, b0        = 1.0, 1.0      # Beta prior for weighted model (uninformative)
print_every   = 25            # Print results every N flips
# --------------------------------

# --- Simulate flips ---
flips = (np.random.rand(n_flips) < p_true).astype(int)  # 1=heads, 0=tails

# --- Initialize values ---
H = T = 0
a, b = a0, b0
log_BF = 0.0
log_prior_odds = np.log(pi_weighted) - np.log(1 - pi_weighted)

log_BF_path, post_W_path, p_mean_path, p_lo_path, p_hi_path = [], [], [], [], []

print(f"=== Bayesian Updating: Fair vs Weighted Coin ===")
print(f"True p = {p_true:.3f} | Prior P(weighted) = {pi_weighted:.2f} | Beta prior (a,b)=({a0},{b0})\n")

for i, x in enumerate(flips, start=1):
    # Predictive probabilities
    denom = a + b + H + T
    p_head_weighted = (a + H) / denom
    p_tail_weighted = (b + T) / denom
    p_x_weighted = p_head_weighted if x == 1 else p_tail_weighted
    p_x_fair = 0.5

    # Update Bayes factor (log space)
    log_BF += np.log(p_x_weighted) - np.log(p_x_fair)

    # Update counts
    if x == 1:
        H += 1
    else:
        T += 1

    # Posterior parameters
    a_post, b_post = a0 + H, b0 + T
    p_mean = a_post / (a_post + b_post)
    
    # Credible interval (grid approximation)
    grid = np.linspace(0, 1, 2001)
    eps = 1e-12
    log_pdf = (a_post - 1) * np.log(np.clip(grid, eps, 1.0)) + (b_post - 1) * np.log(np.clip(1 - grid, eps, 1.0))
    pdf = np.exp(log_pdf - log_pdf.max())
    pdf /= np.trapz(pdf, grid)
    cdf = np.cumsum(pdf) / np.sum(pdf)
    lo = grid[np.searchsorted(cdf, 0.025)]
    hi = grid[np.searchsorted(cdf, 0.975)]

    # Model posterior
    log_posterior_odds = log_prior_odds + log_BF
    odds = np.exp(log_posterior_odds)
    post_weighted = odds / (1 + odds)

    # Store results
    log_BF_path.append(log_BF)
    post_W_path.append(post_weighted)
    p_mean_path.append(p_mean)
    p_lo_path.append(lo)
    p_hi_path.append(hi)

    # Print progress every `print_every` flips
    if i % print_every == 0 or i == n_flips:
        print(f"After {i:4d} flips  |  H={H:3d}  T={T:3d}  |  "
              f"Post P(weighted) = {post_weighted:0.3f}  |  "
              f"p̄ = {p_mean:0.3f}  |  95% CI p = [{lo:0.3f}, {hi:0.3f}]")

# --- Plot results ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: model posterior
axes[0].plot(range(1, n_flips+1), post_W_path, lw=2, color='tab:blue', label="P(Weighted | data)")
axes[0].axhline(pi_weighted, color='gray', ls='--', label='Prior P(Weighted)')
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("Number of flips")
axes[0].set_ylabel("Posterior P(Weighted)")
axes[0].set_title("Model Posterior: Weighted vs Fair")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: posterior mean for p
axes[1].plot(range(1, n_flips+1), p_mean_path, lw=2, color='tab:green', label="Posterior mean p")
axes[1].fill_between(range(1, n_flips+1), p_lo_path, p_hi_path, color='tab:green', alpha=0.15)
axes[1].axhline(0.5, color='red', ls='--', alpha=0.7, label="Fair p=0.5")
axes[1].axhline(p_true, color='black', ls='--', alpha=0.7, label=f"True p={p_true:.2f}")
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("Number of flips")
axes[1].set_ylabel("p")
axes[1].set_title("Parameter Posterior for p (Weighted Model)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("Bayesian Coin: Posterior Updates Over Time", fontsize=13, weight='bold')
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    === Bayesian Updating: Fair vs Weighted Coin ===
    True p = 0.600 | Prior P(weighted) = 0.50 | Beta prior (a,b)=(1.0,1.0)
    
    After   25 flips  |  H= 12  T= 13  |  Post P(weighted) = 0.199  |  p̄ = 0.481  |  95% CI p = [0.299, 0.666]
    After   50 flips  |  H= 26  T= 24  |  Post P(weighted) = 0.154  |  p̄ = 0.519  |  95% CI p = [0.385, 0.652]
    After   75 flips  |  H= 42  T= 33  |  Post P(weighted) = 0.196  |  p̄ = 0.558  |  95% CI p = [0.447, 0.667]
    After  100 flips  |  H= 62  T= 38  |  Post P(weighted) = 0.689  |  p̄ = 0.618  |  95% CI p = [0.522, 0.709]
    After  125 flips  |  H= 74  T= 51  |  Post P(weighted) = 0.479  |  p̄ = 0.591  |  95% CI p = [0.504, 0.674]
    After  150 flips  |  H= 87  T= 63  |  Post P(weighted) = 0.409  |  p̄ = 0.579  |  95% CI p = [0.500, 0.656]
    After  175 flips  |  H=101  T= 74  |  Post P(weighted) = 0.430  |  p̄ = 0.576  |  95% CI p = [0.503, 0.648]
    After  200 flips  |  H=118  T= 82  |  Post P(weighted) = 0.693  |  p̄ = 0.589  |  95% CI p = [0.520, 0.656]
    After  225 flips  |  H=133  T= 92  |  Post P(weighted) = 0.778  |  p̄ = 0.590  |  95% CI p = [0.526, 0.653]
    After  250 flips  |  H=152  T= 98  |  Post P(weighted) = 0.965  |  p̄ = 0.607  |  95% CI p = [0.546, 0.666]
    After  275 flips  |  H=166  T=109  |  Post P(weighted) = 0.966  |  p̄ = 0.603  |  95% CI p = [0.544, 0.659]
    After  300 flips  |  H=178  T=122  |  Post P(weighted) = 0.932  |  p̄ = 0.593  |  95% CI p = [0.537, 0.647]
```

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_42_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    

{% endcapture %}
{% include codeoutput.html content=ex %}  





##### What the Plots Show

**Left Plot:**  
Tracks the *posterior probability* that the coin is **weighted** after each flip. If the coin truly is biased, this value will climb toward 1 as evidence accumulates. If it’s actually fair, the probability will fall toward 0.

**Right Plot:**  
Shows how our estimate of the coin’s **bias parameter $p$** evolves over time. The green curve is the **posterior mean**, and the shaded region is the **95% credible interval**.

As we observe more flips:
- The credible interval narrows (our uncertainty decreases).
- The mean approaches the true value of $p_{\text{true}}$.
- The model’s confidence (posterior probability of “weighted”) becomes decisive.






#### Example: Bayesian Updating with Two Features

In this demonstration, we extend the logic of our coin-flip example into two dimensions.
We now have two measurable features — for instance, these could represent observable traits like temperature and humidity, or height and weight — and we’re trying to decide which of two possible models (H₀ or H₁) best explains our data.

Each model assumes that the data come from a 2D Gaussian distribution (a bell curve in two dimensions). As we collect more samples, we update our belief about which hypothesis is more likely using Bayes’ theorem.

The left plot shows how our posterior probability for H₁ evolves as we gather more evidence.
The right plot shows the feature space, with a decision boundary separating regions that are more likely under H₀ versus H₁. Points are drawn one by one, and as more are observed, the model becomes increasingly confident about which hypothesis generated the data.

**Key Ideas**
- Features can work together: A single variable may not be enough to classify accurately, but two (or more) features combined can reveal strong patterns.
- Overlap causes uncertainty: When distributions overlap (high noise or small separation), the model updates its confidence more slowly.
- Bayesian inference accumulates evidence: Early observations may fluctuate, but with enough data, the posterior tends to stabilize near the true generating process.

**Try Adjusting**:
- separation — smaller values make classes harder to distinguish.
- sigma — higher noise makes learning slower.
- tau — dampens how strongly each observation influences the posterior (useful for visualizing gradual convergence).
- true_class — flip between "H0" and "H1" to see how the posterior moves in opposite directions.


{% capture ex %}
```python
# === Two-Feature Bayesian Model Selection (Gradual Convergence) ===
# Compare H0 vs H1 for 2D Gaussian data; update posterior sequentially.
# Tunable overlap (separation, sigma) and optional likelihood tempering (tau).

#np.random.seed(7)

# -------------------------
# Tunable settings
# -------------------------
n_obs      = 100
prior_H1   = 0.5          # prior P(H1)
true_class = "H1"         # "H0" or "H1"

separation = 0.6          # smaller = harder (means closer)
sigma      = 1.2          # larger = harder (more noise)
tau        = 1.0          # 1.0 = standard Bayes; 0.3–0.7 slows convergence for demos
print_every = 25          # print progress every X obs

# Means (H0 around (0,0), H1 shifted by 'separation' on both axes)
muA1, muA2 = 0.0, 0.0
muB1, muB2 = separation, 0.7 * separation

# Same variance on both axes for both models
sigma1 = sigma2 = sigma

# -------------------------
# Generate data
# -------------------------
if true_class == "H0":
    x1 = np.random.normal(muA1, sigma1, n_obs)
    x2 = np.random.normal(muA2, sigma2, n_obs)
else:
    x1 = np.random.normal(muB1, sigma1, n_obs)
    x2 = np.random.normal(muB2, sigma2, n_obs)

X = np.column_stack([x1, x2])

# -------------------------
# Sequential Bayes update (with optional tempering)
# -------------------------
log_prior_odds = np.log(prior_H1) - np.log(1 - prior_H1)
log_BF_path, post_H1_path = [], []
log_BF = 0.0

def point_llr(x1, x2):
    ll1 = norm.logpdf(x1, muB1, sigma1) + norm.logpdf(x2, muB2, sigma2)
    ll0 = norm.logpdf(x1, muA1, sigma1) + norm.logpdf(x2, muA2, sigma2)
    return ll1 - ll0  # log likelihood ratio

print("=== Two-Feature Bayesian Updating: H0 vs H1 ===")
print(f"True class = {true_class} | Prior P(H1) = {prior_H1:.2f}")
print(f"H0 means = ({muA1:.2f}, {muA2:.2f}), H1 means = ({muB1:.2f}, {muB2:.2f}), sigma = {sigma:.2f}")
print(f"Separation = {separation:.2f} | Tempering tau = {tau:.2f}\n")

for i, (xi1, xi2) in enumerate(X, start=1):
    # tempered per-point contribution (tau<1 slows updates for pedagogy)
    log_BF += tau * point_llr(xi1, xi2)

    # posterior from odds
    log_post_odds = log_prior_odds + log_BF
    odds = np.exp(log_post_odds)
    post_H1 = odds / (1 + odds)

    log_BF_path.append(log_BF)
    post_H1_path.append(post_H1)

    if i % print_every == 0 or i == n_obs:
        print(f"After {i:4d} obs | Posterior P(H1) = {post_H1:0.3f} | log BF = {log_BF:0.2f}")

# -------------------------
# Plots
# -------------------------
fig, axes = plt.subplots(2, 1, figsize=(6, 8))

# (Left) posterior over time
axes[0].plot(range(1, n_obs+1), post_H1_path, lw=2, color='tab:blue', label="P(H1 | data)")
axes[0].axhline(prior_H1, color='gray', ls='--', label='Prior P(H1)')
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("Number of observations")
axes[0].set_ylabel("Posterior P(H1)")
axes[0].set_title(f"Posterior Convergence (true={true_class}, sep={separation}, σ={sigma}, τ={tau})")
axes[0].grid(alpha=0.3)
axes[0].legend()

# (Right) feature space with decision boundary (LLR=0)
x1_min, x1_max = X[:,0].min() - 2*sigma1, X[:,0].max() + 2*sigma1
x2_min, x2_max = X[:,1].min() - 2*sigma2, X[:,1].max() + 2*sigma2
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                     np.linspace(x2_min, x2_max, 300))

llr_grid = (
    norm.logpdf(xx, muB1, sigma1) + norm.logpdf(yy, muB2, sigma2)
  - norm.logpdf(xx, muA1, sigma1) - norm.logpdf(yy, muA2, sigma2)
)
# color regions by which hypothesis is favored
axes[1].contourf(xx, yy, (llr_grid > 0).astype(int), alpha=0.20, cmap="coolwarm")
axes[1].contour(xx, yy, llr_grid, levels=[0], colors='k', linewidths=2)

# observed points
axes[1].scatter(X[:,0], X[:,1], s=35, c='k', alpha=0.55, label="observations")
# class means
axes[1].scatter([muA1, muB1], [muA2, muB2], s=180, c=["tab:blue","tab:red"], edgecolor="k", label="class means")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].set_title("Feature Space + Bayes Decision Boundary")
axes[1].legend(loc="best")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
   === Two-Feature Bayesian Updating: H0 vs H1 ===
    True class = H1 | Prior P(H1) = 0.50
    H0 means = (0.00, 0.00), H1 means = (0.60, 0.42), sigma = 1.20
    Separation = 0.60 | Tempering tau = 1.00
    
    After   25 obs | Posterior P(H1) = 0.724 | log BF = 0.97
    After   50 obs | Posterior P(H1) = 0.684 | log BF = 0.77
    After   75 obs | Posterior P(H1) = 0.998 | log BF = 6.15
    After  100 obs | Posterior P(H1) = 1.000 | log BF = 8.65
```


<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_45_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    


{% endcapture %}
{% include codeoutput.html content=ex %}  


 








### From Bayes’ Theorem to Naïve Bayes Classification


Now that we understand how Bayes’ Theorem works for a single piece of evidence, let’s see how we can extend it to work for *many features at once*.

In classification, we often want to know:

> “Given a set of features — say, bill length, flipper length, and body mass — what is the probability that this penguin belongs to a certain species?”

We can write that as:

$$
P(\text{Species} \vert \text{Features}) = 
\frac{P(\text{Features} \vert \text{Species}) \, P(\text{Species})}
{P(\text{Features})}
$$

That’s great in theory, but there’s a problem:
- The number of combinations of features explodes.
- Estimating $ P(\text{Features}\vert\text{Species}) $ directly would require *tons* of data.

So how do we make this tractable?  
We make one bold — and very helpful — **assumption**.







#### The “Naïve” Assumption

Naïve Bayes assumes that all features are **conditionally independent** given the class.

In math terms:

$$
P(x_1, x_2, ..., x_n \very \text{Class}) = P(x_1 \vert \text{Class}) P(x_2 \vert \text{Class}) \cdots =  \prod_i P(x_i \vert \text{Class})
$$

That means each feature contributes its own independent piece of evidence for or against a class. This assumption allows us to write the probability of a class given the features in the following manner:

$$
P(\text{Class} \vert x_1, x_2, ..., x_n) \;\propto\; P(\text{Class}) \prod_i P(x_i \vert C)
$$

This assumption is almost never *literally* true. For example, in the Iris dataset, petal length and petal width are clearly correlated. But, it still works remarkably well in practice because it captures *most* of the structure in the data.

##### Intuition
Each feature acts like a separate “vote.” If multiple features all point toward the same class, the combined probability for that class increases rapidly.

We’re not saying the features don’t interact — they likely do — we’re just saying the model *assumes they don’t* in order to make calculations fast and stable.

##### Example of the Intuition
Imagine trying to classify animals using just two features: “has fur” and “lays eggs.”

- If “has fur” → likely a mammal.  
- If “lays eggs” → likely not a mammal.
- These features are not conditionally independent - platypus. 
- However, we can assume both features are conditionally independent, since the cross over is tiny compared to the number of cases where they do not cross over.


{% capture ex %}
```python
# === Simple 2-Feature Numerical Example ===

# Suppose we’re classifying an animal as “Mammal” or “Not Mammal”
# Using two binary features: Has Fur (x1) and Lays Eggs (x2)

# Priors
P_Mammal = 0.5
P_NotMammal = (1 - P_Mammal)

# Likelihoods for each feature given class
P_fur_given_Mammal = 0.9
P_fur_given_NotMammal = 0.1

P_eggs_given_Mammal = 0.05
P_eggs_given_NotMammal = 0.8

# Observation: animal has fur AND lays eggs
x1, x2 = 1, 1

# Compute joint likelihoods (naïve assumption)
if (x1,x2) == (1,1):
    P_obs_given_Mammal = P_fur_given_Mammal * P_eggs_given_Mammal
    P_obs_given_NotMammal = P_fur_given_NotMammal * P_eggs_given_NotMammal
elif (x1,x2) == (1,0): 
    P_obs_given_Mammal = P_fur_given_Mammal * (1-P_eggs_given_Mammal)
    P_obs_given_NotMammal = P_fur_given_NotMammal * (1-P_eggs_given_NotMammal)
elif (x1,x2) == (0,1): 
    P_obs_given_Mammal = (1-P_fur_given_Mammal) * P_eggs_given_Mammal
    P_obs_given_NotMammal = (1-P_fur_given_NotMammal) * P_eggs_given_NotMammal
else: 
    P_obs_given_Mammal = (1-P_fur_given_Mammal) * (1 - P_eggs_given_Mammal)
    P_obs_given_NotMammal = (1-P_fur_given_NotMammal) * (1 - P_eggs_given_NotMammal)

# Compute posteriors (normalized so they make sense)
posterior_Mammal = (P_obs_given_Mammal * P_Mammal) / (
    P_obs_given_Mammal * P_Mammal + P_obs_given_NotMammal * P_NotMammal
)
posterior_NotMammal = 1 - posterior_Mammal

print(f"======== Prior Beleifs ========")
print(f"P(Mammal | evidence) = {P_Mammal:.2f}")
print(f"P(Not Mammal | evidence) = {P_NotMammal:.2f}")
print("")

print(f"======== Likelihoods ========")
print(f"P(Fur | Mammal) = {P_fur_given_Mammal:.2f}")
print(f"P(Fur | Not Mammal) = {P_fur_given_NotMammal:.2f}")
print(f"P(Lays Eggs | Mammal) = {P_eggs_given_Mammal:.2f}")
print(f"P(Lays Eggs | Not Mammal) = {P_eggs_given_NotMammal:.2f}")
print("")

print(f"======== Observations ========")
print(f"Has Fur? (1 = Yes, 0 = No): {x1:.0f}")
print(f"Lays Eggs? (1 = Yes, 0 = No): {x2:.0f}")
print("")

print(f"======== Posterior Beliefs ========")
print(f"P(Mammal | evidence) = {posterior_Mammal:.3f}")
print(f"P(Not Mammal | evidence) = {posterior_NotMammal:.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    ======== Prior Beleifs ========
    P(Mammal | evidence) = 0.50
    P(Not Mammal | evidence) = 0.50
    
    ======== Likelihoods ========
    P(Fur | Mammal) = 0.90
    P(Fur | Not Mammal) = 0.10
    P(Lays Eggs | Mammal) = 0.05
    P(Lays Eggs | Not Mammal) = 0.80
    
    ======== Observations ========
    Has Fur? (1 = Yes, 0 = No): 1
    Lays Eggs? (1 = Yes, 0 = No): 1
    
    ======== Posterior Beliefs ========
    P(Mammal | evidence) = 0.360
    P(Not Mammal | evidence) = 0.640
```
{% endcapture %}
{% include codeoutput.html content=ex %}  




Even though “has fur” strongly favors mammal, the evidence “lays eggs” overwhelms it — the combined probability now favors not mammal.

<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b> 

- **Naïve Bayes** applies Bayes’ Theorem to every feature, assuming they’re independent.  
- This assumption simplifies math, making it fast and scalable — ideal for text or categorical data.  
- Each feature contributes an independent “vote” to the overall probability.  
- Despite its simplicity, Naïve Bayes performs surprisingly well in practice, especially when features provide complementary information.
</div>

#### Technical Note:

Most systems will actually calculate the log-probability since it is easier to process and is less prone to overflow errors in computer archatecture. By this we mean numbers get too large or small for the computer to keep track of. Taking the log of the prabability turns the product above into a summation:

$$
\ln\Big(P(\text{Class} \vert x_1, x_2, ..., x_n)\Big) \;\propto\; \ln\Big(P(\text{Class}) \prod_i P(x_i \vert C)\Big) = \ln\Big(P(\text{Class})\Big) + \ln\Big(\prod_i P(x_i \vert C)\Big) = \ln\Big(P(\text{Class})\Big) + \sum_i \ln\Big(P(x_i \vert C)\Big)
$$

You do not need to worry about this technical detial. It is being mentioned here for those who are interested. 







### Two-Feature Naïve Bayes Example: Flower Classification

Here we simulate two types of flowers — **Setosa** and **Not Setosa** — based on their 
petal **length** and **width**.

Each feature *on its own* only partially separates the species:
- A flower with long petals is *probably* not Setosa.
- A flower with narrow petals is *probably* Setosa.
  
But neither feature is perfect in isolation.

When we **combine both features**, however, their joint probability distributions 
make the classes easily separable.

Naïve Bayes models this joint probability as the product of the individual ones:

$$
P(\text{Class}\vert x_1, x_2) \propto P(\text{Class}) \times P(x_1\vert \text{Class}) \times P(x_2\vert \text{Class})
$$

Assuming independence between petal length and width within each class, the model uses simple Gaussian likelihoods (bell curves) for each feature.

The **decision boundary** (shown by the shaded regions) is where the model assigns equal probability to each class. Notice that the combination of both features yields excellent separation — far better than using either feature alone.


{% capture ex %}
```python
# === Two-Feature Naïve Bayes Demonstration (Fixed) ===
# Classifies two flower species using two continuous features

np.random.seed(42)

# --- Generate two overlapping Gaussian clusters ---
n = 150

# Class 0 (Setosa-like)
x0 = np.random.normal(1.5, 0.3, n)
y0 = np.random.normal(0.3, 0.1, n)
X0 = np.column_stack((x0, y0))

# Class 1 (Not Setosa-like)
x1 = np.random.normal(4.5, 0.7, n)
y1 = np.random.normal(1.5, 0.4, n)
X1 = np.column_stack((x1, y1))

# Combine into one dataset
X = np.vstack((X0, X1))
y = np.array([0]*n + [1]*n)

# --- Train Naïve Bayes classifier ---
model = GaussianNB()
model.fit(X, y)
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

# --- Decision region grid ---
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# --- Plot ---
plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
plt.scatter(X0[:, 0], X0[:, 1], color="tab:blue", label="Setosa", edgecolor="k", s=60)
plt.scatter(X1[:, 0], X1[:, 1], color="tab:red", label="Not Setosa", edgecolor="k", s=60)
plt.xlabel("Feature 1: Petal Length (cm)")
plt.ylabel("Feature 2: Petal Width (cm)")
plt.title(f"Naïve Bayes Classification (Accuracy = {acc*100:.1f}%)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}


<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_52_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    

    


{% endcapture %}
{% include codeoutput.html content=ex %}  










### Gaussian Naïve Bayes on the Iris Dataset


Now that we understand how Naïve Bayes *thinks*, let’s put it into action.

We’ll use the **Iris dataset**, which has:
- 4 continuous features: sepal length, sepal width, petal length, petal width
- 3 species (classes)

Because the features are continuous, we’ll use <strong>Gaussian Naïve Bayes</strong> (GNB). This assumes that for each class, each feature is normally distributed (bell-shaped) and that features are conditionally independent given the class (Naive Bayes).

**What the model learns**  
- For *each class* and *each feature*, it estimates the mean and variance from the training data (simple sample averages within the class).

| Class | sepal length (Mean) | sepal length (Var) | sepal width (Mean) | sepal width (Var) | petal length (Mean) | petal length (Var) | petal width (Mean) | petal width (Var) |
|:------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| setosa | 5.006 | 0.124 | 3.418 | 0.144 | 1.464 | 0.030 | 0.244 | 0.011 |
| versicolor | 5.936 | 0.266 | 2.770 | 0.098 | 4.260 | 0.220 | 1.326 | 0.039 |
| virginica | 6.588 | 0.404 | 2.974 | 0.104 | 5.552 | 0.304 | 2.026 | 0.075 |

All measured in centimeters.

- It also uses a class prior, typically the class’s relative frequency in the training set (unless you set priors explicitly).

| Class | Prior Probability |
|:------|:----------------:|
| setosa | 0.333 |
| versicolor | 0.333 |
| virginica | 0.333 |


**How prediction works**  
1. For a new flower, compute the Gaussian likelihood of each feature under each class using that class’s mean and variance.
2. Multiply these likelihoods across features (the Naïve independence step) and combine with the class prior to get a score proportional to the posterior.
3. Pick the class with the highest posterior score.


> In short: GNB fits a mean and variance per feature per class, assumes independence across features within a class, and chooses the class with the largest resulting posterior.




{% capture ex %}
```python
# === Gaussian Naïve Bayes Demo on Iris Dataset ===
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# --- Load dataset ---
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# --- Split and scale ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train Naïve Bayes ---
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred = nb.predict(X_test_scaled)

# --- Evaluate ---
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

Accuracy: 0.911

```python
Classification Report:
                precision    recall  f1-score   support

        setosa       1.00      1.00      1.00        15
    versicolor       0.82      0.93      0.88        15
    virginica       0.92      0.80      0.86        15

    accuracy                           0.91        45
    macro avg       0.92      0.91      0.91        45
weighted avg       0.92      0.91      0.91        45
```
{% endcapture %}
{% include codeoutput.html content=ex %}  








### Confusion Matrix

We can get a feel for how well this model did and the kinds of errors it made by looking at the confusion matrix:

{% capture ex %}
```python
# --- Confusion Matrix ---
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"Gaussian Naïve Bayes Confusion Matrix (Accuracy = {acc:.2f})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_56_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  




<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion Prompt:</b> 
<br>
The confusion matrix shows how many samples each class was correctly or incorrectly classified.

Gaussian Naïve Bayes tends to perform quite well on the Iris dataset — 
even though its “independence” assumption isn’t strictly true.

Why?  
Because:
- Each feature still provides strong independent clues.  
- The Gaussian distributions fit the feature data fairly well.  
- It doesn’t overfit — it stays simple and interpretable.
</div>





### Decision Boundaries

We’ll use PCA to project the data to 2D and visualize the model’s decision boundaries. This gives an intuitive sense of how well the classes separate.

Keep in mind: this is only a 2D view of a higher-dimensional dataset. Apparent overlaps in the plot may disappear in the full feature space.


{% capture ex %}
```python
# --- Reduce to 2 principal components for visualization ---
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Refit model on PCA-reduced data (for visualization only)
nb_pca = GaussianNB()
nb_pca.fit(X_train_pca, y_train)

# Create a meshgrid
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = nb_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(7,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set2')
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1],
                hue=[target_names[i] for i in y_train],
                palette="Set2", edgecolor="k", s=60)
plt.title("Gaussian Naïve Bayes Decision Boundaries (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_59_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    


{% endcapture %}
{% include codeoutput.html content=ex %}  



  
#### Interpreting the Boundaries

Each color region represents the area where the Naïve Bayes model 
believes one class is *most probable*.

Notice that:
- Boundaries are **soft** and somewhat **curved**, unlike the sharp linear lines of logistic regression.  
- Naïve Bayes naturally handles **uncertainty** — points near the edges have more ambiguous probabilities.  
- It performs especially well when the classes have **distinct statistical distributions** (like different means and variances).










### Types of Naïve Bayes Classifiers

"Naïve Bayes" isn't a single algorithm — it's a family of models that all use **Bayes’ Theorem** under the assumption that features are **conditionally independent** given the class.

Each version of Naïve Bayes makes a different assumption about the **distribution** of the features.

| Type | Data Type | Distributional Assumption | Example Applications |
|:------|:-----------|:--------------------------|:----------------------|
| **GaussianNB** | Continuous numeric | Features are normally distributed (follow a bell curve) | Iris data, sensor readings |
| **MultinomialNB** | Discrete counts | Features represent frequency or counts | Text classification, spam filters |
| **BernoulliNB** | Binary | Features are True/False (0/1) | Document term presence, yes/no surveys |

These variants all compute class probabilities using Bayes’ theorem — they just calculate likelihoods differently.


{% capture ex %}
```python
# --- Demonstration: Gaussian vs Multinomial Naïve Bayes ---

# --- Load Iris dataset ---
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
class_names = iris.target_names

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# --- Gaussian Naïve Bayes (for continuous features) --- 
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# Metrics
acc_gnb = accuracy_score(y_test, y_pred_gnb)
report_gnb = classification_report(
    y_test, y_pred_gnb, target_names=class_names, digits=3
)
cm_gnb = confusion_matrix(y_test, y_pred_gnb)


# --- Multinomial Naïve Bayes (for count data) --- 
# Convert continuous features to non-negative integer "counts"
X_train_counts = np.clip((X_train * 10).astype(int), a_min=0, a_max=None)
X_test_counts = np.clip((X_test * 10).astype(int), a_min=0, a_max=None)

mnb = MultinomialNB()
mnb.fit(X_train_counts, y_train)
y_pred_mnb = mnb.predict(X_test_counts)

# Metrics
acc_mnb = accuracy_score(y_test, y_pred_mnb)
report_mnb = classification_report(
    y_test, y_pred_mnb, target_names=class_names, digits=3
)
cm_mnb = confusion_matrix(y_test, y_pred_mnb)


# --- Print Results --- 
print("=== Gaussian Naïve Bayes ===")
print(f"Accuracy: {acc_gnb:.3f}\n")
print("Classification Report:")
print(report_gnb)
print("Confusion Matrix:")
print(pd.DataFrame(cm_gnb, index=class_names, columns=class_names))
print("\n")

# --- Confusion Matrix ---
plt.figure(figsize=(5,4))
sns.heatmap(cm_gnb, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"Gaussian Naïve Bayes Confusion Matrix (Accuracy = {acc_gnb:.2f})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

print("\n" + "="*60 + "\n")

print("=== Multinomial Naïve Bayes ===")
print(f"Accuracy: {acc_mnb:.3f}\n")
print("Classification Report:")
print(report_mnb)
print("Confusion Matrix:")
print(pd.DataFrame(cm_mnb, index=class_names, columns=class_names))
print("\n")

# --- Confusion Matrix ---
plt.figure(figsize=(5,4))
sns.heatmap(cm_mnb, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"Multinomial Naïve Bayes Confusion Matrix (Accuracy = {acc_mnb:.2f})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
    === Gaussian Naïve Bayes ===
    Accuracy: 0.911
    
    Classification Report:
                  precision    recall  f1-score   support
    
          setosa      1.000     1.000     1.000        15
      versicolor      0.824     0.933     0.875        15
       virginica      0.923     0.800     0.857        15
    
        accuracy                          0.911        45
       macro avg      0.916     0.911     0.911        45
    weighted avg      0.916     0.911     0.911        45
    
    Confusion Matrix:
                setosa  versicolor  virginica
    setosa          15           0          0
    versicolor       0          14          1
    virginica        0           3         12
```
    



<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_62_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">

    


```python 
    === Multinomial Naïve Bayes ===
    Accuracy: 0.978
    
    Classification Report:
                  precision    recall  f1-score   support
    
          setosa      1.000     1.000     1.000        15
      versicolor      0.938     1.000     0.968        15
       virginica      1.000     0.933     0.966        15
    
        accuracy                          0.978        45
       macro avg      0.979     0.978     0.978        45
    weighted avg      0.979     0.978     0.978        45
    
    Confusion Matrix:
                setosa  versicolor  virginica
    setosa          15           0          0
    versicolor       0          15          0
    virginica        0           1         14
```
    


<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_62_3.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  



### Example: Naïve Bayes for Spam Detection (Toy SMS Demo)

This short example shows how a Naïve Bayes text classifier can distinguish between spam and normal (“ham”) messages using only a few lines of Python.

We’ll build a miniature SMS spam detector from scratch, using a tiny in-memory dataset for demonstration.

No downloads or external data files are needed — the goal is simply to illustrate the full machine learning workflow:

1. Create a small labeled dataset of spam and ham messages
2. Split the data into training and testing sets
3. Vectorize text into numerical features (using TF-IDF or bag-of-words)
4. Train a Naïve Bayes classifier (MultinomialNB)
5. Evaluate performance with accuracy, precision, recall, F1, and a confusion matrix
6. Inspect the most “spammy” words learned by the model
7. Test it on new example messages

This pipeline mirrors what you’d do with a full real-world dataset (like the SMS Spam Collection), but scaled down for easy experimentation.

It’s also a great example of how probabilistic models like Naïve Bayes can work surprisingly well for real text classification tasks — even with minimal data and tuning.

{% capture ex %}
```python
# === Naïve Bayes for Spam Detection (Toy SMS Demo) ===
# No downloads needed. This uses a tiny in-memory dataset for teaching purposes.
# Swap in a real dataset later (e.g., the SMS Spam Collection) without changing the pipeline.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# 1) Tiny SMS-style dataset (balanced for demo)
#    Label: 1 = spam, 0 = ham (not spam)
# --------------------------------------------------
spam_texts = [
    "WINNER! claim your prize now click the link",
    "Congratulations you have won a free gift card reply YES",
    "Urgent! your account is selected for a reward call now",
    "Get rich quick!!! limited time crypto investment offer",
    "You have been selected for a $1000 Walmart gift card",
    "FREE entry in a weekly contest! text WIN to 55555",
    "Lowest interest loan pre-approved apply today",
    "Claim your exclusive offer now limited seats available",
    "This is not a scam click here to verify your details",
    "Act now! limited time deal reply STOP to opt out",
]

ham_texts = [
    "Are we still meeting for lunch at noon?",
    "Don’t forget the team standup at 9am tomorrow.",
    "Can you send me the slides from class?",
    "Running 10 minutes late sorry!",
    "Let’s call tonight and review the report.",
    "Thanks for your help earlier—really appreciate it.",
    "I’ll bring snacks for the study group.",
    "What time works for you to chat?",
    "The package arrived this afternoon.",
    "Happy birthday! Hope you have a great day."
]

texts = spam_texts + ham_texts
y = np.array([1]*len(spam_texts) + [0]*len(ham_texts))

# --------------------------------------------------
# 2) Train/test split
# --------------------------------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, y, test_size=0.3, random_state=42, stratify=y
)

# --------------------------------------------------
# 3) Vectorize text → features
#    Option A: Bag-of-words counts (good with MultinomialNB)
#    Option B: TF-IDF (also works well)
#    (Uncomment one; we’ll use TF-IDF here.)
# --------------------------------------------------
# vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english")  # unigrams+bigrams
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)

X_train = vectorizer.fit_transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)

# --------------------------------------------------
# 4) Train Naïve Bayes
#    MultinomialNB is standard for counts/TF-IDF.
#    (BernoulliNB can work for presence/absence features)
# --------------------------------------------------
clf = MultinomialNB(alpha=1.0)  # Laplace smoothing
clf.fit(X_train, y_train)

# --------------------------------------------------
# 5) Evaluate
# --------------------------------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham","spam"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4.5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham","spam"], yticklabels=["ham","spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6) Inspect most “spammy” tokens (log prob ratios)
# --------------------------------------------------
# For MultinomialNB, feature_log_prob_ has shape [n_classes, n_features]
feature_names = np.array(vectorizer.get_feature_names_out())
log_prob = clf.feature_log_prob_
# log-odds: log P(token|spam) - log P(token|ham)
spam_score = log_prob[1] - log_prob[0]
top_n = 12
top_idx = np.argsort(spam_score)[-top_n:][::-1]
print("\nTop tokens indicative of SPAM:")
for tok, score in zip(feature_names[top_idx], spam_score[top_idx]):
    print(f"{tok:>20s}    log-odds={score: .3f}")

# --------------------------------------------------
# 7) Try a few custom messages
# --------------------------------------------------
samples = [
    "You have been selected for a free prize! claim now",
    "Running late—see you at the meeting in 10?",
    "Limited time: act now to win cash rewards",
    "Happy birthday! Want to get dinner tonight?"
]
X_samples = vectorizer.transform(samples)
proba = clf.predict_proba(X_samples)
pred = clf.predict(X_samples)

print("\nSample predictions:\n")
for s, p, pr in zip(samples, pred, proba):
    label = "spam" if p==1 else "ham"
    print(f"- {s}\n  -> Predicted: {label:>4s} | P(ham)={pr[0]:.2f}, P(spam)={pr[1]:.2f} \n")

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

    Accuracy: 0.833
    
```python
    Classification Report:
                  precision    recall  f1-score   support
    
             ham       0.75      1.00      0.86         3
            spam       1.00      0.67      0.80         3
    
        accuracy                           0.83         6
       macro avg       0.88      0.83      0.83         6
    weighted avg       0.88      0.83      0.83         6 
```


<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_64_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    


```python
    Top tokens indicative of SPAM:
                   click    log-odds= 0.456
                   claim    log-odds= 0.416
                    card    log-odds= 0.406
               gift card    log-odds= 0.406
                    gift    log-odds= 0.406
                   offer    log-odds= 0.384
                 limited    log-odds= 0.384
                    free    log-odds= 0.367
                 details    log-odds= 0.293
            click verify    log-odds= 0.293
                    scam    log-odds= 0.293
              scam click    log-odds= 0.293
    
    Sample predictions:
    
    - You have been selected for a free prize! claim now
      -> Predicted: spam | P(ham)=0.34, P(spam)=0.66 
    
    - Running late—see you at the meeting in 10?
      -> Predicted:  ham | P(ham)=0.60, P(spam)=0.40 
    
    - Limited time: act now to win cash rewards
      -> Predicted: spam | P(ham)=0.42, P(spam)=0.58 
    
    - Happy birthday! Want to get dinner tonight?
      -> Predicted:  ham | P(ham)=0.64, P(spam)=0.36 
``` 



{% endcapture %}
{% include codeoutput.html content=ex %}  










### Limitations and When to Use Naïve Bayes

By now, we’ve seen that Naïve Bayes is:
- **Simple** to understand and implement,  
- **Fast** to train (no iterative optimization), and  
- **Surprisingly effective** across a wide range of problems.

But, like all models, it comes with some caveats.

#### Strengths

1. **Speed:**  
   Naïve Bayes requires only a few simple statistical estimates (means, variances, probabilities).  
   This makes it *orders of magnitude faster* than models like logistic regression or random forests.

2. **Scalability:**  
   Works well with **high-dimensional data** — especially in text classification, 
   where each word is treated as a feature.

3. **Robustness to Irrelevant Features:**  
   Because each feature is treated independently, adding or removing one 
   doesn’t drastically affect the model.

4. **Probabilistic Output:**  
   Produces meaningful class probabilities, which can be used for ranking or decision thresholds.


#### Limitations

1. **Independence Assumption:**  
   The biggest weakness — real-world features are often correlated (like petal length and width).  
   This can distort probabilities, even if predictions are still accurate.

2. **Continuous vs. Categorical Features:**  
   Gaussian Naïve Bayes assumes normal (bell-shaped) distributions for numeric data.  
   If your data isn’t Gaussian, accuracy can suffer.

3. **Zero Probability Problem:**  
   If a feature value never appears in training data for a class,  
   the model assigns a probability of 0 (which kills that class).  
   *Solution:* Use “Laplace Smoothing” to prevent zeros.

4. **Not Good for Boundary Precision:**  
   Naïve Bayes makes “soft” decisions — it’s great for ranking but may be less sharp 
   when you need precise decision boundaries.

---

Let’s visualize a couple of quirks — 1. the *independence assumption failure*, and 2. data with non-normal shapes.



{% capture ex %}
```python
# === Independence Violation Demo ===
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Generate correlated features ---
X, y = make_classification(
    n_samples=300, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.0, random_state=42
)

# Add correlation manually
X[:,1] = 0.8*X[:,0] + 0.3*np.random.randn(300)

# --- Standardize ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Fit models ---
nb = GaussianNB()
lr = LogisticRegression()
nb.fit(X_scaled, y)
lr.fit(X_scaled, y)

# --- Plot decision boundaries ---
x_min, x_max = X_scaled[:,0].min()-1, X_scaled[:,0].max()+1
y_min, y_max = X_scaled[:,1].min()-1, X_scaled[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z_nb = nb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_lr = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].contourf(xx, yy, Z_nb, cmap="coolwarm", alpha=0.3)
axes[0].scatter(X_scaled[:,0], X_scaled[:,1], c=y, cmap="coolwarm", edgecolor="k")
axes[0].set_title("Naïve Bayes (Independence Violated)")

axes[1].contourf(xx, yy, Z_lr, cmap="coolwarm", alpha=0.3)
axes[1].scatter(X_scaled[:,0], X_scaled[:,1], c=y, cmap="coolwarm", edgecolor="k")
axes[1].set_title("Logistic Regression (Correlated Features)")

for ax in axes:
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3)

plt.suptitle("Effect of Feature Correlation on Naïve Bayes vs Logistic Regression", fontsize=14)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_66_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  



    

#### Discussion

In this example:
- The two features are **highly correlated**, meaning they carry similar information.
- Logistic Regression handles this fine — it learns a *tilted* boundary that follows the true data shape.
- Naïve Bayes, however, struggles because it assumes the features are independent —
  it ends up forming a boundary that’s less aligned with the data cloud.

This is why Naïve Bayes sometimes underperforms on datasets 
where feature correlations carry important information.

However, it’s *still* often good enough — and far faster to train.


{% capture ex %}
```python
# === Comparing Clustering and Classification Methods on Nonlinear Data (with Naïve Bayes) ===

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# --- Generate nonlinear "half-moon" data ---
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)

# --- Standardize features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Define models ---
models = {
    "k-Means (Unsupervised)": KMeans(n_clusters=2, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Naïve Bayes": GaussianNB()
}

# --- Create decision grid ---
xx, yy = np.meshgrid(
    np.linspace(X_scaled[:,0].min() - 0.5, X_scaled[:,0].max() + 0.5, 300),
    np.linspace(X_scaled[:,1].min() - 0.5, X_scaled[:,1].max() + 0.5, 300)
)

# --- Plot results ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ax, (name, model) in zip(axes, models.items()):
    if "k-Means" in name:
        model.fit(X_scaled)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        labels = model.labels_
    else:
        model.fit(X_scaled, y)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        labels = model.predict(X_scaled)
    
    Z = Z.reshape(xx.shape)
    
    # Background decision regions
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    
    # Data points
    ax.scatter(X_scaled[:,0], X_scaled[:,1], c=y, cmap="coolwarm", s=30, edgecolor="k")
    
    # Accuracy (for supervised models)
    acc = accuracy_score(y, labels)
    ax.set_title(f"{name}\nAccuracy = {acc:.2f}")
    
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3)

plt.suptitle("Comparing k-Means, Logistic Regression, k-NN, and Naïve Bayes on Nonlinear Data", fontsize=14)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_68_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

{% endcapture %}
{% include codeoutput.html content=ex %}  










### When to Use and Avoid Naïve Bayes


#### When to Use Naïve Bayes

| Use Case | Why It Works |
|:----------|:--------------|
| **Text Classification (Spam, Sentiment, Topics)** | Words act as independent features; “bag-of-words” fits perfectly. |
| **Document Categorization** | Fast even with thousands of features. |
| **Medical Diagnosis** | Probabilistic nature allows uncertainty modeling. |
| **Real-Time Systems** | Training is instantaneous — ideal for frequent retraining. |


#### When to Avoid

| Situation | Why It Fails |
|:-----------|:--------------|
| Strongly correlated features | Violates independence assumption. |
| Continuous data that’s not Gaussian | Breaks the likelihood model. |
| Complex nonlinear boundaries | Better served by decision trees, SVMs, or neural networks. |










## Choosing the Right Analysis

Each algorithm answers a *different kind* of question.  


Here are examples of the problems each method is best suited for.

### **k-Nearest Neighbors (k-NN)**
**Goal:** Classify or predict using the *closest examples*.
- What species is this flower most similar to?
- Will this customer buy a product, given their profile?
- Estimate a house’s price by looking at nearby houses.

**Output:** Category or value based on neighborhood voting or averaging.


### **k-Means Clustering**
**Goal:** Discover natural groups when you *don’t have labels*.
- Can we group customers into market segments?
- Which patients show similar symptom patterns?
- Do galaxies form clusters in observed data?

**Output:** Cluster assignments and centroids (group centers).


### **Linear & Multiple Linear Regression**
**Goal:** Predict a continuous quantity.
- How does study time predict exam scores?
- Can we estimate house prices from square footage and number of rooms?
- What’s the relationship between temperature and ice cream sales?

**Output:** A numeric prediction (and an interpretable equation).


### **Logistic Regression**
**Goal:** Predict categorical outcomes with a *linear* decision boundary.
- Will a student pass or fail given GPA and attendance?
- Is an email spam or not spam based on keyword frequency?
- Does a tumor appear benign or malignant?

**Output:** Class probabilities (and log-odds relationships).

### **Naïve Bayes**
**Goal:** Probabilistic classification assuming feature independence.
- Is a text positive or negative in sentiment (word frequencies)?
- What’s the probability this patient has a disease given their symptoms?
- Does an animal belong to species A, B, or C?

**Output:** Most probable class and posterior probabilities.





### Summary

| Model | Type | Goal | Key Idea |
|:------|:------|:------|:------|
| **Linear Regression** | Supervised | Predict numeric value | Fit a straight line through data |
| **Logistic Regression** | Supervised | Binary / multiclass classification | Linear separation via sigmoid |
| **Naïve Bayes** | Supervised | Probabilistic classification | Uses Bayes’ theorem & feature independence |
| **k-NN** | Supervised | Classification / regression | Uses local similarity (neighbors) |
| **k-Means** | Unsupervised | Clustering | Finds groups by minimizing within-cluster distance |



<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b> 

Before choosing an algorithm, ask yourself:
1. Do I have **labels** (supervised) or not (unsupervised)?
2. Is my **target variable** continuous or categorical?
3. Do I care more about **interpretability** or **flexibility**?
4. Are my **features independent**, correlated, or complexly structured?

Once you can answer those, the right tool usually reveals itself.
</div>


## One-Cell Code

{% capture ex %}

{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}

{% endcapture %}
{% include codeoutput.html content=ex %}  
```python
# === Lecture 04: Logistic Regression vs Naïve Bayes (Iris Demo) ===
# Fully working single-cell example

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------------
# 1) Load and prepare data
# ----------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# 2) Train models
# ----------------------------------------------------------
logreg = LogisticRegression(max_iter=1000, multi_class="auto", random_state=42)
nb = GaussianNB()

logreg.fit(X_train_scaled, y_train)
nb.fit(X_train_scaled, y_train)

y_pred_log = logreg.predict(X_test_scaled)
y_pred_nb = nb.predict(X_test_scaled)

# ----------------------------------------------------------
# 3) Summarize model performance
# ----------------------------------------------------------
def summarize_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    return cm

cm_log = summarize_model("Logistic Regression", y_test, y_pred_log)
cm_nb  = summarize_model("Gaussian Naïve Bayes", y_test, y_pred_nb)

# ----------------------------------------------------------
# 4) Plot confusion matrices
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9,4))
for ax, cm, title in zip(
    axes, [cm_log, cm_nb], ["Logistic Regression", "Gaussian Naïve Bayes"]
):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 5) PCA Visualization (2D projection)
# ----------------------------------------------------------
pca = PCA(n_components=2)
X_vis = pca.fit_transform(StandardScaler().fit_transform(X))
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.3, random_state=42, stratify=y
)

# Train models on 2D PCA data
logreg_vis = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_vis, y_train_vis)
nb_vis = GaussianNB().fit(X_train_vis, y_train_vis)

xx, yy = np.meshgrid(
    np.linspace(X_vis[:,0].min()-1, X_vis[:,0].max()+1, 200),
    np.linspace(X_vis[:,1].min()-1, X_vis[:,1].max()+1, 200)
)
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(10,4))
for ax, model, title in zip(
    axes, [logreg_vis, nb_vis],
    ["Logistic Regression (PCA 2D)", "Naïve Bayes (PCA 2D)"]
):
    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="Accent")
    scatter = ax.scatter(X_vis[:,0], X_vis[:,1], c=y, s=40, edgecolor="k", cmap="Accent")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

# Add legend manually (no truth-value ambiguity)
handles, labels = scatter.legend_elements()
plt.figlegend(handles, class_names, loc="upper center", ncol=3, frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

```

    
    === Logistic Regression ===
    Accuracy: 0.967
                  precision    recall  f1-score   support
    
          setosa      1.000     1.000     1.000        10
      versicolor      0.909     1.000     0.952        10
       virginica      1.000     0.900     0.947        10
    
        accuracy                          0.967        30
       macro avg      0.970     0.967     0.967        30
    weighted avg      0.970     0.967     0.967        30
    
    
    === Gaussian Naïve Bayes ===
    Accuracy: 0.967
                  precision    recall  f1-score   support
    
          setosa      1.000     1.000     1.000        10
      versicolor      0.909     1.000     0.952        10
       virginica      1.000     0.900     0.947        10
    
        accuracy                          0.967        30
       macro avg      0.970     0.967     0.967        30
    weighted avg      0.970     0.967     0.967        30
    



<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_72_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    


<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_72_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    

    



```python
# === Lecture 04: Logistic Regression vs Naïve Bayes (Iris Demo) ===
# Build for creating threshold plots for precision, recall, and F1-score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

# ----------------------------------------------------------
# 1) Load and prepare data
# ----------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

target_num = 2 # y == 0 "setosa" vs non-setosa
               # y == 1 "versicolor" vs non-versicolor
               # y == 2 "virginica" vs non-virginica

# Select right labels:
if target_num == 0:
    use_labels = ["non-setosa","setosa"]
elif target_num == 1:
    use_labels = ["non-versicolor","versicolor"]
else:
    use_labels = ["non-virginica","virginica"]


# To simplify threshold visualization, make it binary: class 0 vs all others
y_binary = (y == target_num).astype(int)  

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, stratify=y_binary
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# 2) Train models
# ----------------------------------------------------------
logreg = LogisticRegression(max_iter=1000, random_state=2)
nb = GaussianNB()

logreg.fit(X_train_scaled, y_train)
nb.fit(X_train_scaled, y_train)

y_pred_log = logreg.predict(X_test_scaled)
y_pred_nb = nb.predict(X_test_scaled)

# ----------------------------------------------------------
# 3) Summarize model performance
# ----------------------------------------------------------
def summarize_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=use_labels, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    return cm

cm_log = summarize_model("Logistic Regression", y_test, y_pred_log)
cm_nb  = summarize_model("Gaussian Naïve Bayes", y_test, y_pred_nb)

# ----------------------------------------------------------
# 4) Plot confusion matrices
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10,4))
for ax, cm, title in zip(
    axes, [cm_log, cm_nb], ["Logistic Regression", "Gaussian Naïve Bayes"]
):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=use_labels,
                yticklabels=use_labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 5) Precision / Recall / F1 vs. Threshold
# ----------------------------------------------------------
def compute_threshold_metrics(model, X, y_true, label):
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

thr_log, prec_log, rec_log, f1_log = compute_threshold_metrics(logreg, X_test_scaled, y_test, "Logistic")
thr_nb,  prec_nb,  rec_nb,  f1_nb  = compute_threshold_metrics(nb, X_test_scaled, y_test, "Naïve Bayes")

fig, axes = plt.subplots(1, 2, figsize=(10,4))
for ax, thr, prec, rec, f1, title in zip(
    axes, 
    [thr_log, thr_nb],
    [prec_log, prec_nb],
    [rec_log, rec_nb],
    [f1_log, f1_nb],
    ["Precision, Recall, and F1 vs Threshold (LogReg)", "Precision, Recall, and F1 vs Threshold (NB)"]
):
    ax.plot(thr, prec, "--", label="Precision (NaïveBayes)", color="tab:blue")
    ax.plot(thr, rec, "--", label="Recall (NaïveBayes)", color="tab:orange")
    ax.plot(thr, f1, "--", label="F1 (NaïveBayes)", color="tab:green")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall, and F1 vs Threshold")
    ax.legend()
plt.tight_layout()
plt.show()

```

    
    === Logistic Regression ===
    Accuracy: 0.900
                   precision    recall  f1-score   support
    
    non-virginica      0.947     0.900     0.923        20
        virginica      0.818     0.900     0.857        10
    
         accuracy                          0.900        30
        macro avg      0.883     0.900     0.890        30
     weighted avg      0.904     0.900     0.901        30
    
    
    === Gaussian Naïve Bayes ===
    Accuracy: 0.933
                   precision    recall  f1-score   support
    
    non-virginica      1.000     0.900     0.947        20
        virginica      0.833     1.000     0.909        10
    
         accuracy                          0.933        30
        macro avg      0.917     0.950     0.928        30
     weighted avg      0.944     0.933     0.935        30
    



<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_73_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
    


<img
  src="{{ '/courses/machine-learning-foundations/images/lec04/output_73_2.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    

