---
layout: jupyternotebook
title: Machine Learning Foundations – Lecture 02
course_home: /courses/machine-learning-foundations/
nav_section: lectures
nav_order: 2
---

# Lecture 02: Instance-Based & Cluster-Based Learning  

{% capture ex %}

{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}

{% endcapture %}
{% include codeoutput.html content=ex %}

## Setup: Import Libraries

Let's import the libraries and functions we will need for this lecture:

{% capture ex %}
```python
# ===============================================================
# === Import Libraries ===
# ===============================================================

# --- Core Python / Math Tools ---
import numpy as np                    # numerical operations, arrays, distance computations
import pandas as pd                   # data handling and manipulation
import math

# --- Visualization Libraries ---
import matplotlib.pyplot as plt        # general plotting
from matplotlib.colors import ListedColormap
import seaborn as sns                 # polished statistical plots
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# --- scikit-learn: Datasets ---
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, make_blobs
)

# --- scikit-learn: Model Preparation ---
from sklearn.model_selection import train_test_split, cross_val_score   # split data into train/test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # feature scaling & label encoding

# --- scikit-learn: Metrics ---
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score, pairwise_distances, adjusted_rand_score
)

from sklearn.dummy import DummyClassifier

# --- scikit-learn: Algorithms ---
from sklearn.neighbors import KNeighborsClassifier       # k-Nearest Neighbors
from sklearn.cluster import KMeans                       # k-Means clustering

# --- scikit-learn: Dimensionality Reduction ---
from sklearn.decomposition import PCA                    # reduce features for 2D visualization

# --- Visualization Utilities for Trees / Boundaries ---
from matplotlib.colors import ListedColormap             # color maps for decision boundaries

# --- Other Useful Tools ---
import warnings
warnings.filterwarnings("ignore")  # keep output clean for class demos
```
{% endcapture %}
{% include codeinput.html content=ex %}






# k-Nearest Neighbors (k-NN) & k-Means Clustering  

---

## Learning Goals
By the end of this session, you will be able to:
- Explain the difference between **supervised** and **unsupervised** learning.  
- Describe how **k-NN** (classification) and **k-Means** (clustering) operate conceptually.  
- Understand how **distance metrics** and **feature scaling** affect model performance.  
- Apply k-NN and k-Means in Python using `scikit-learn`.  
- Interpret clustering results.

---


## Supervised vs. Unsupervised Learning


There are two main types of machine learning models: **supervised** and **unsupervised**.  

The distinction lies in whether or not the algorithm is trained using *labeled* data — that is, whether it already knows the “right answers” during training.

| Type | Goal | Examples | Data Has Labels? |
|------|------|-----------|------------------|
| **Supervised** | Learn to predict known outcomes from examples | Classification (e.g., k-NN, Logistic Regression, Decision Trees), Regression (e.g., Linear Regression) | ✅ Yes |
| **Unsupervised** | Find structure or patterns hidden in the data | Clustering (e.g., k-Means), Dimensionality Reduction (e.g., PCA) | ❌ No |



### Supervised Learning

In supervised learning, the model is given input data **and** the corresponding correct answers (labels). 

It learns to map inputs → outputs by finding patterns that link the features to the labels.  

Think of it as **teaching by example**:  
> You show the model hundreds of penguins with known species labels, and it learns to predict the species for new, unseen penguins.

Once trained, we evaluate how well it generalizes to new data using metrics like accuracy, precision, or recall.




### Unsupervised Learning

In unsupervised learning, there are **no labels**.  

The model explores the data **on its own**, searching for patterns, relationships, or groupings.  

Think of it as **exploration without guidance**:  
> You give the model penguin measurements with no species information, and it tries to group similar penguins together based on their features alone.

The goal isn’t to predict a known outcome but to *discover structure* that might not be obvious — such as clusters, correlations, or underlying trends.




### In Practice

Most real-world problems start with **unsupervised exploration** (to understand and clean the data) and move toward **supervised learning** (to build predictive models once labels are available).


### Conceptual Link
- **k-NN** is supervised, meaning we need ***labeled*** data to train our model. These models then apply new labels to data by comparing them to our known ones.
- **k-Means** is unsupervised, meaning we can use ***unlabeled*** data t otrain out model. The model ***discovers*** groups and "invents" labels for them.
    - You, as the human operator, have to check that the groups are meaningful and then given them a name.




## What Is k-Nearest Neighbor Modeling?

k-Nearest Neighbors (k-NN) is one of the simplest and most intuitive machine learning algorithms.

It works on a simple idea:

> **“To predict something about a new data point, look at its closest examples.”**

> **“Birds of a feather flock together (in their features).”**

That’s it! There’s no complicated training. The model *stores* the data and makes predictions by comparing distances.


### The Basic Process

1. **Pick a number of neighbors** \(k\) (for example, 3 or 5).  
2. **Find the k closest points** in the training data to your new point.  
3. **Vote!**
   - For classification → the new point takes the *most common* class among its neighbors.
   - For regression → the new point takes the *average* value of its neighbors.  
4. That’s it — nothing coplicated, just proximity.


### What “Learning” Means Here
k-NN doesn’t have *learned parameters* like slope in linear regression or transformers in neural networks.

It simply **remembers** the data and uses it for comparison later.

This is why we call it an **instance-based** or **lazy** learner:
- *Instance-based* → it predicts based on nearby examples (instances)).  
- *Lazy* → it doesn’t build a model until you ask it to make a prediction.




{% capture ex %}
```python
# HOW MANY NEIBHBORS?
num_neighbors = 2

# WHERE IS THE NEW POINT
x_new = -2
y_new = -2.5

# Generate a simple 2D dataset
X, y = make_blobs(n_samples=100, centers=3, random_state=2, cluster_std=1.2)

# Create a k-NN classifier and fit
knn = KNeighborsClassifier(n_neighbors = num_neighbors)
knn.fit(X, y)

# New sample to classify
new_point = np.array([[x_new, y_new]])

# Predict and find neighbors
neighbors = knn.kneighbors(new_point, return_distance=False)

# Plot everything
plt.figure(figsize=(7,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=70, label="Training Data")
plt.scatter(new_point[0, 0], new_point[0, 1], c="red", s=200, edgecolors="black", marker="*", label="New Point")

# Highlight nearest neighbors
plt.scatter(X[neighbors[0], 0], X[neighbors[0], 1], s=250, facecolors="none", edgecolors="red", linewidths=2, label="Nearest Neighbors")

plt.title(f"How k-NN Classifies a New Point (k={num_neighbors})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_5_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}



    


<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion:</b> 
    <ul>
      <li>What would happen if we changed k from 5 to 1?</li>
      <li>Does the red star’s predicted class depend only on its closest points?</li>
      <li>What if we added a noisy point nearby — how might that change the vote?</li>
    </ul>
</div>



### Decision Boundaries

When we train a **classification model** like **k-NN**, we’re teaching the computer how to *divide* the feature space into regions that correspond to different categories (or "classes").  

A **decision boundary** is a colored plot that separates these decision regions. One regions could correspond to Option 1 and another to Option 2.

For example, if we have the three groups from the previous example, then the k_NN decision boundaries would look something like:


{% capture ex %}
```python
# HOW MANY NEIBHBORS?
num_neighbors = 5

# Create a k-NN classifier and fit
knn = KNeighborsClassifier(n_neighbors = num_neighbors)
knn.fit(X, y)

h = 0.05  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on a grid of points to plot decision boundaries
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(7,6))
plt.contourf(xx, yy, Z, cmap=ListedColormap(sns.color_palette("pastel", 3)), alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k", s=70)
plt.scatter(new_point[0, 0], new_point[0, 1], c="red", s=200, edgecolors="black", marker="*")
plt.title("k-NN Decision Boundaries (k=5)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_8_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">    
{% endcapture %}
{% include codeoutput.html content=ex %}



 


#### What Is a Decision Boundary?
A **decision boundary** is the “border” where a model changes its prediction from one class to another.  

Every point on one side of the boundary will be predicted as one class; points on the other side belong to a different class.

The decision boundary naturally forms wherever the votes change from one class to another.


#### Why Visualize Decision Boundaries?
Visualizing decision boundaries helps us **see how a model thinks**:

- It shows *how complex* or *simple* a model’s decision rules are.
- It reveals where *confusion or overlap* happens between classes.
- It helps us spot *overfitting* (boundaries that are too wiggly and specific to the training data).


#### What to Look For
When you plot the decision boundaries:
1. **Smoothness:** Do the boundaries look too jagged? That may mean overfitting, which is not good.
2. **Separation:** Are the classes clearly separated? That suggests good model performance.
3. **Misclassified points:** Are there training points on the “wrong” side of the line?
4. **Scalability:** Would the same boundary make sense if new data were added?


#### Example: k-NN Decision Boundaries for increasing k
In **k-NN**, each region in the plot represents the predicted class for that area of feature space.  

The boundaries are determined by *distances* to the training points:
- A small value of **k** → very detailed, wiggly boundaries (possibly overfitting).
- A large value of **k** → smoother, simpler boundaries (possibly underfitting).

{% capture ex %}
```python
fig, axes = plt.subplots(3, 1, figsize=(5, 14))

for ax, k in zip(axes, [1, 5, 11]):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=ListedColormap(sns.color_palette("pastel", 3)), alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k", s=50)
    ax.set_title(f"k = {k}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.suptitle("How Changing k Affects the k-NN Decision Boundary", fontsize=14)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_11_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}





    



In a bit, we’ll discuss how to choose a good `k`, but to do that we first need to talk about *accuracy*.



## Evaluating Model Performance



###  Model Accuracy

When we train a machine learning model, we need a way to measure **how well it performs** — that’s where **accuracy** comes in.


#### What Accuracy Means
Accuracy measures the **proportion of correct predictions** the model makes out of all predictions.

> You train the model on your training set, and get the accuracy of the model's predictions by applying it to the test set.

For instance, suppose your model correctly predicts 90 out of the 100 data points in your test set. The accuracy of the model in that test set would be:

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}} = \frac{90}{100} = 0.90
$$

or **90%**.


#### How It’s Calculated
1. The model is given data it hasn’t seen before (the **test set**).  
2. For each test example, it predicts a label (e.g., species = “Setosa”).  
3. The predictions are compared to the **true labels**.  
4. The number of matches determines the accuracy.


#### What is condisered "good"?

There’s no single number that defines a “good” accuracy, and entirely depending on the context, data quality, and the problem you are attemping to solve. That said, here’s a solid, practical breakdown of good rules of thumb:

| Accuracy Range | General Interpretation                                           | Typical Contexts                                                             |
| -------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **< 60%**      | Usually poor — model not learning much beyond random guessing    | Complex data, unbalanced classes, or wrong model choice                      |
| **60–70%**     | Barely acceptable, only “okay” for noisy or highly variable data | Social science / behavior prediction / subjective ratings                    |
| **70–80%**     | Reasonably good for real-world data; often a solid baseline      | Customer churn, sentiment analysis, handwriting recognition                  |
| **80–90%**     | Strong performance — model generalizes well                      | Many classical ML problems (e.g., species classification, medical diagnosis) |
| **90–95%**     | Very strong — often “production-ready” if not overfitting        | Well-defined tasks with clean data (e.g., digit recognition, spam filters)   |
| **>95%**       | Excellent — *or suspiciously perfect*                            | Only realistic when the problem is simple or the model is overfitting        |


#### Important Caveats

- **Always compare to a baseline model (a random guessing model).**
- **Accuracy isn’t always the best metric.**
    - For imbalanced datasets (e.g., disease detection, fraud detection), accuracy can mislead.
    - Use precision, recall, or F1 score instead — these capture how well your model identifies rare but important cases.
- **Dataset difficulty matters.**
    - Predicting whether a penguin is Adélie or Chinstrap? You might get 95%.
    - Predicting if someone will click an ad? You might be thrilled with 70%.
- **Compare multiple ML models.**
    - Often, what matters is whether one model outperforms another on the same data — not the absolute value.



<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #8EC9DC;
    padding: 12px;
    border-radius: 6px;
">
<b style="color:#1b4965;">Professional Practice:</b>  
<br><br>
Accuracy can make your model look better than it really is.  
As a data scientist, it’s your responsibility to ensure that the metrics you report
actually reflect the model’s performance and limitations.  
<br><br>
If <b>random guessing gives you 50%</b>, then <b>70% is a big win</b>.  <br>
But <b>if guessing gives you 90%</b> (like in imbalanced data), then <b>92% is not impressive</b>.
<br><br>
Or, take for example, if <b>99 % of patients don’t have a disease you are sceening for</b>, 
a model that always predicts “no disease” achieves <b>99 % accuracy</b>—but fails completely at detecting the people who are sick (loads of false negatives).
When the event you’re predicting is rare, <b>accuracy alone can be misleading</b>.  
That’s when metrics like <b>precision</b>, <b>recall</b>, and <b>F1-score</b> become essential.
</div>



#### Let's do it. 

The following code will run a k=5 k-NN for the iris data and report the accuracy of our method. 

We will need to do the following:

1. Let's pull in the data,
2. split it into test and training sets,
3. scale the data,
4. train the k-NN clssifier
5. Run the model on the test data to get the accuracy.


<!-- Reflection -->
<div style="
    background-color: #ffebeb;
    border-left: 6px solid #ff0000;
    padding: 10px;
    border-radius: 5px;
">
<b>Warning:</b> If the data were not numeric, then we would need to encode it. Though, as we will see, we would have to think very carefully about which features make sense to use in k-NN and k-Means. We will discuss this today.
</div>


{% capture ex %}
```python
# --- Load dataset ---
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Split into train/test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train a k-NN classifier ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# --- Predict and evaluate ---
y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
Accuracy: 0.933
{% endcapture %}
{% include codeoutput.html content=ex %}



### Baseline Models

Now, let's compare this to a **baseline model**. 


A **baseline model** gives us a simple reference point to evaluate how well our model learned from the data.

- The baseline can be set up in a number of different ways:
    -  always predict the **most frequent label** from the training data.
    -  **randomly assign labels** from the training data.  
- If your model performs **similarly to the baseline**, it’s not finding real patterns.
- If it performs **much better than the baseline**, it’s learning potentially meaningful structures from within the data.

>Think of the baseline as the **“null hypothesis”** of machine learning:  
> if your model can’t beat it, it’s time to rethink the features, preprocessing, or algorithm choice.



{% capture ex %}
```python
# --- Baseline model ---
baseline = DummyClassifier(strategy="uniform") 
baseline.fit(X_train_scaled, y_train)
baseline_pred = baseline.predict(X_test_scaled)
baseline_acc = accuracy_score(y_test, baseline_pred)

# --- Compare results ---
print(f"Baseline (most frequent class) Accuracy: {baseline_acc:.3f}")
print(f"k-NN Accuracy: {acc:.3f}")
print(f"Improvement over baseline: {100 * (acc - baseline_acc):.1f}%")
```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
Baseline (most frequent class) Accuracy: 0.367
k-NN Accuracy: 0.933
Improvement over baseline: 56.7%
{% endcapture %}
{% include codeoutput.html content=ex %}





Notice our model's accuracy was much better than the baseline. This suggests our model is potentially learning something meaningful from the data. 

#### **Note on the various strategies for building a baseline model**:

| Strategy            | What It Does                                                                                                     | Example Behavior                                                                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`most_frequent`** | Always predicts the **most common class** in the training data.                                                  | If 70% of your data are "Setosa" penguins, it will predict *Setosa* every time.                                                                     |
| **`prior`**         | Same as `most_frequent`, but is used to investigate the distribution of labels from the training data.                                                                                     |
| **`stratified`**    | Makes **random predictions**, but in proportion to the training set’s class distribution.                        | If 70% of your training data are “Setosa” and 30% are “Versicolor,” then 70% of its predictions will be *Setosa* and 30% *Versicolor* (on average). |
| **`uniform`**       | Predicts **completely at random**, with equal chance for every class.                                            | Each class has the same probability (e.g., 33% each in a 3-class problem).                                                                          |
| **`constant`**      | Always predicts a **specific class** you define (via the `constant` parameter).                                  | You can tell it to always predict *“Adelie”* no matter what.                                                                                        |


#### **When to Use Each**

- `most_frequent` → Default baseline for classification; checks if your model beats a simple majority guess.
- `stratified` → Good for sanity checks in imbalanced data (it mimics the class distribution).
- `uniform` → Tests how a model performs against pure random guessing.
- `constant` → Occasionally useful for debugging or demonstrating bias.

<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaway:</b> 

The goal isn’t for the DummyClassifier to be “good” — it’s to remind you what “bad but honest” looks like.  If your trained model doesn’t perform better than one of these, your model hasn’t learned anything meaningful yet.
</div>




<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #8EC9DC;
    padding: 12px;
    border-radius: 6px;
">
<b style="color:#1b4965;">Professional Practice:</b>  
<br><br>
Always compare your machine learning model’s performance to a <b>baseline model</b>.
<br><br>
Baselines help you confirm that your model is actually learning something meaningful.
If your model doesn’t outperform a simple baseline, it’s a signal to:
<ul>
<li>check your features and data preprocessing,</li>
<li>review whether your model is appropriate for the task, or</li>
<li>reconsider whether the problem itself is predictable with the available data.</li>
</ul>
Comparing to a baseline keeps your results honest, interpretable, and professionally defensible.
</div>



### Finding the Best k — The “Elbow Method”


When using **k-NN**, the number of neighbors (`k`) is a *hyperparameter* of the model we must choose. That is, it is not somthing the model selects, or can be taught to select, but it something we have to manually imput ourselves.

- Too small a `k` (like 1): the model memorizes noise → **overfitting**.  
- Too large a `k`: the model over-smooths patterns → **underfitting**.  

To find a good balance, we can test several values of `k` and plot **accuracy vs. k**.  
The point where the accuracy stops improving noticeably — the *elbow* — is often a good choice.


{% capture ex %}
```python
# --- Evaluate accuracy for different k values ---
k_values = range(1, 21)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

# --- Plot accuracy vs. k ---
plt.figure(figsize=(8,5))
plt.plot(k_values, accuracies, marker='o')
plt.title("k-NN Elbow Method (Accuracy vs. Number of Neighbors)")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.xticks(k_values)  # ✅ Force integer tick marks only
plt.grid(True)
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_22_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">    
{% endcapture %}
{% include codeoutput.html content=ex %}



    


<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion:</b> 
    <ul>
      <li>Where does the accuracy begin to level off?</li>
      <li>Why does very small `k` sometimes perform worse on test data?</li>
      <li>How might this curve change with more or less noisy datasets?</li>
    </ul>
</div>


A **good rule of thumb** is to choose a moderately small k — large enough to smooth out noise but small enough to preserve meaningful patterns.

In practice, values between 3 and 15 often work well, but the best k depends on your data’s size and complexity.

#### **General Rules of Thumb for Choosing k**

- **Avoid k that’s too small** (like 1–2):
    - Small k means the model reacts too strongly to noise — it memorizes the training data instead of learning general patterns.
    - This leads to overfitting (perfect accuracy on training, poor performance on new data).
- **Avoid k that’s too large**:
    - As k grows, each prediction includes more neighbors, blurring class boundaries.
    - The model becomes too “smooth,” leading to underfitting (missing important local patterns).
    - This is a bit like Goldilocks and the Three Bears...
- **Typical starting range**:
    - Many practitioners start with **odd values** between 3 and 15 (for binary classification) **to avoid ties**.
    - For multiclass problems, you can safely explore up to about √N, where N is the number of samples.
        - e.g., if you have 100 samples → try k ≈ 10.
- **Best practice**:
    - Don’t rely on a single rule — instead, plot model accuracy versus k (like your elbow plot!)
    - Choose the smallest k that gives high accuracy without sharp fluctuations.


### Implementation Note: Matching `X_train` and `X_test`


When we split our dataset into **training** and **testing** sets, we’re simulating how a model will perform on *new, unseen data*.

This means all preprocessing steps —  like scaling, encoding, or feature selection —  must be applied **consistently** across both sets.

#### Common Mistake

A frequent beginner error is to “fit” the scaler separately on the test set, like this:

{% capture ex %}
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit the transformation to the training data
X_test_scaled = scaler.fit_transform(X_test)     # Fit the transformation to the test data
                                                 # ❌ wrong!
```
{% endcapture %}
{% include codeinput.html content=ex %}


This leaks information from the test set into the model.

#### Correct Approach 

Always ***fit a transformation*** only **on the training data**, then ***transformation*** to **the test data**:

{% capture ex %}
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit the transformation to the training data
X_test_scaled = scaler.transform(X_test)         # Transformation the test data
                                                 # ✅ correct!
```
{% endcapture %}
{% include codeinput.html content=ex %}



The same rule applies for:
- LabelEncoder or OneHotEncoder
- PCA or dimensionality reduction methods
- Feature selection steps

Always fit on the trianing data and transform the test data.

#### Why It Matters

When we deploy a model, it will only see new data. If our scaling or encoding is inconsistent, the model will make nonsense predictions — even if it looked perfect in training.

Treat your preprocessing like part of your model: 
>once fitted on training data, it should be reused exactly as-is for testing and deployment.


{% capture ex %}
```python
# --- Load data ---
iris = load_iris()
X = iris.data
y = iris.target

# --- Split into training and testing data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# INCORRECT SCALING APPROACH
# ==========================================
scaler_wrong = StandardScaler()

# Fit on training *and* test data
X_train_wrong = scaler_wrong.fit_transform(X_train)   # Fit the transformation to the training data
X_test_wrong = scaler_wrong.fit_transform(X_test)     # Fit the transformation to the test data
                                                      # ❌ wrong!

# Train k-NN model
knn_wrong = KNeighborsClassifier(n_neighbors=5)
knn_wrong.fit(X_train_wrong, y_train)

# Evaluate accuracy
y_pred_wrong = knn_wrong.predict(X_test_wrong)
acc_wrong = accuracy_score(y_test, y_pred_wrong)

print(f"❌ Incorrect Scaling — Test Accuracy: {acc_wrong:.3f}")

# ==========================================
# CORRECT SCALING APPROACH
# ==========================================
scaler_right = StandardScaler()

# Fit on training data /and/ transform the test data
X_train_scaled = scaler_right.fit_transform(X_train)   # Fit the transformation to the training data
X_test_scaled = scaler_right.transform(X_test)         # Transformation the test data
                                                       # ✅ correct!

# Train k-NN model
knn_right = KNeighborsClassifier(n_neighbors=5)
knn_right.fit(X_train_scaled, y_train)

# Evaluate accuracy
y_pred_right = knn_right.predict(X_test_scaled)
acc_right = accuracy_score(y_test, y_pred_right)

print(f"✅ Correct Scaling — Test Accuracy: {acc_right:.3f}")


```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
❌ Incorrect Scaling — Test Accuracy: 0.956
✅ Correct Scaling — Test Accuracy: 0.911
{% endcapture %}
{% include codeoutput.html content=ex %}







#### Why the Accuracy Changes

In the incorrect version, the test set is scaled *independently* — so its mean and variance differ from the training set.  That means the model sees data that’s numerically inconsistent with what it learned from.

Even though the model and parameters are the same, the test data “lived” in a different coordinate space due to the refitting of the transformation.

When trasnformed correctly, both training and test data are transformed using the **same mean and standard deviation**. Now the model is comparing like with like.



### Confusion Matrix

As we saw, accuracy tells us the *overall* percentage of correct predictions, but it doesn’t tell us *what kinds* of mistakes our model is making.

A **confusion matrix** gives us a detailed breakdown:

|               | Predicted Class A | Predicted Class B | Predicted Class C |
|---------------|------------------|------------------|------------------|
| **Actual A**  | True Positive    | False Negative   | ... |
| **Actual B**  | False Positive   | True Negative    | ... |
| $\vdots$  | $\vdots$   | $\vdots$    | $\vdots$ |

Each row represents the **true class**, and each column represents the **predicted class**.

- ✅ A perfect classifier would have all counts along the diagonal.  
- ❌ Off-diagonal entries indicate *misclassifications*.

There are 2 types of misclassifications:
 
- **False Positives (aka Type I Error)**: The model predicts the positive class when the true label is actually negative.
  > Example: Predicting a patient has a disease when they actually don’t.  
  > Example: Sample is predicted as begin Class A when it is actually Class B. 
- **False Negatives (aka Type II Error)**: The model predicts the negative class when the true label is actually positive.
  > Example: Predicting a patient doesn’t have a disease when they actually do.  
  > Example: Sample is predicted as begin Class B when it is actually Class A. 

|               | Predicted Yes Disease | Predicted No Disease |
|---------------|------------------|------------------|
| **Actual Yes Disease**  | True Positive    | False Negative   |
| **Actual No Disease**  | False Positive   | True Negative    |


#### Let's do it. 

Let’s visualize the confusion matrix for our k-NN model of the iris flowers.


{% capture ex %}
```python
# --- Confusion Matrix ---
y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("k-NN Confusion Matrix")
plt.show()
```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
Accuracy: 0.956

<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_29_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}






    


<h3 style="
    color: white;
    background-color: #f4b942;
    padding: 8px;
    border-radius: 6px;
">
Cross-Validation: Measuring Generalization
</h3>

A single train/test split might give us a lucky (or unlucky) result. To get a more stable measure of performance, we can use **cross-validation**.

Cross-validation splits the data into *k folds* (e.g., 5). The model trains on 4 folds and tests on the remaining one — repeated for all folds.

The average score tells us how well the model generalizes.


{% capture ex %}
```python
# Run 5-fold cross-validation
cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=5),
                            X_train_scaled, y_train, cv=5)

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
Cross-Validation Scores: [0.9047619  0.95238095 1.         0.9047619  1.        ]
Mean CV Accuracy: 0.952
{% endcapture %}
{% include codeoutput.html content=ex %}

We will discuss this in more detail in Lecture 06.







## Comparing Distance Metrics in k-NN

The k-NN algorithm depends on how we measure “closeness” between points. Different **distance metrics** can produce different neighborhood shapes and, therefore, different predictions.

Let’s compare a few common ones:

| Metric | Formula | Notes |
|---------|----------|-------|
| **Euclidean** | $$ \sqrt{\sum_i (p_i - q_i)^2} $$ | Straight-line (“as the crow flies”) distance |
| **Manhattan** | $$ \sum_i \lvert p_i - q_i \rvert $$ | City-block distance — useful for grid-like or discrete data |
| **Minkowski** | $$ \left(\sum_i \lvert p_i - q_i \rvert^p \right)^{1/p} $$ | General form (p=1 → Manhattan, p=2 → Euclidean) |
| **Cosine** | $$ 1 - \frac{p \cdot q}{\lvert\lvert p \rvert \rvert\,\,\lvert\lvert q \rvert \rvert} $$ | Measures *angle* similarity; useful for text or high-dimensional data |

We’ll test each metric on the Iris dataset and see how it affects accuracy.



### Quick Visualizations

Let’s visualize how these metrics measure distance between the same two points:


{% capture ex %}
```python
# Visual comparison of Euclidean and Manhattan distances
a = np.array([1, 4])
b = np.array([4, 1])

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(*a, color="blue", s=100, label="Point A")
ax.scatter(*b, color="red", s=100, label="Point B")

# Draw Euclidean (diagonal) line
ax.plot([a[0], b[0]], [a[1], b[1]], "k--", label="Euclidean Path")

# Draw Manhattan path
ax.plot([a[0], a[0]], [a[1], b[1]], "orange", lw=2, label="Manhattan Path (option 1)")
ax.plot([a[0], b[0]], [b[1], b[1]], "orange", lw=2)

# Draw arrows for Cosine distance
ax.arrow(0, 0, a[0]*0.9, a[1]*0.9, head_width=0.2, fc="blue", ec="blue")
ax.arrow(0, 0, b[0]*0.9, b[1]*0.9, head_width=0.2, fc="red", ec="red")

# Compute distances
from sklearn.metrics import pairwise_distances
dist_euc = pairwise_distances([a], [b], metric="euclidean")[0][0]
dist_man = pairwise_distances([a], [b], metric="manhattan")[0][0]
dist_min = pairwise_distances([a], [b], metric="minkowski", p=1.5)[0][0]

# Compute cosine similarity and distance
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
cos_dist = 1 - cos_sim

angle_deg = np.degrees(math.acos(cos_sim))

ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()
ax.set_title("Euclidean vs. Manhattan Distance Illustration")
plt.grid(True)
plt.show()

# Print distance results
print(f"Euclidean = {dist_euc:.2f}")
print(f"Manhattan = {dist_man:.2f}")
print(f"Minkowski (p=1.5) ≈ {dist_min:.2f}")
print(f"Cosine Distance = {cos_dist:.2f}")
```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_35_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">

    


Euclidean = 4.24  
Manhattan = 6.00  
Minkowski (p=1.5) ≈ 4.76  
Cosine Distance = 0.53  
{% endcapture %}
{% include codeoutput.html content=ex %}






## Weighted k-NN — Giving Closer Points More Influence

In standard k-NN, all neighbors contribute equally to the prediction. But it can be smarter to **weigh closer points more heavily**.

The *weighted* version uses the inverse of distance:  
> closer neighbors get more “voting power.”

{% capture ex %}
```python
# --- Compare k-NN performance using different distance metrics ---

metrics = ["euclidean", "manhattan", "minkowski", "cosine"]
results = []

for metric in metrics:
    model = KNeighborsClassifier(n_neighbors=5, metric=metric)
    # Setting the metric introduces the weighting.
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append((metric, acc))

# Convert to DataFrame for neat display
df_results = pd.DataFrame(results, columns=["Distance Metric", "Accuracy"])
df_results

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance Metric</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>euclidean</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>manhattan</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>minkowski</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cosine</td>
      <td>0.777778</td>
    </tr>
  </tbody>
</table>
</div>
{% endcapture %}
{% include codeoutput.html content=ex %}









<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion:</b> 
    <ul>
      <li>Which metric produced the highest accuracy? </li>
      <li>Why might <b>cosine distance</b> behave differently than <b>Euclidean</b>?</li>
      <li>What kind of datasets would favor <b>Manhattan</b> distance?</li>
    </ul>
</div>


---

## What Is k-Means Clustering?

k-Means is an **unsupervised learning** algorithm used to find **groups (clusters)** in unlabeled data.

It looks for natural groupings, points that are *closer to each other* than to others, and assigns each point to one of **k clusters**.


### The k-Means Process

1. **Choose the number of clusters (k).**  
   You decide how many groups the algorithm should find.

2. **Randomly place k centroids.**  
   Each centroid begins at some initial location (often random).

3. **Assign each data point to the nearest centroid (E-step).**  
   Every data point is assigned to its closest centroid — and only one centroid — no partial or multiple memberships allowed.  

4. **Update the centroids (M-step).**  
   Each centroid moves to the *average* position (the "center") of the points connected to the centroid.

   When centroids move, some points that were previously closer to Centroid A might now be closer to Centroid B — so they switch groups.

5. **Repeat steps 3–4** until the centroids stop moving (the solution has *converged*).

The algorithm naturally spreads the centroids out until each has a stable, non-overlapping set of points.


### Intuition
k-Means alternates between:
- **E-Step (Expectation):** Assign points to the nearest cluster.  
- **M-Step (Maximization):** Move centroids to the average of their assigned points.  

It keeps doing this until the cluster boundaries no longer change significantly.

That’s it!


{% capture ex %}
```python
# --- Create a simple 2D dataset ---
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)

# --- Choose the number of clusters ---
k = 3
np.random.seed(1)  # try different seeds to see different starting positions

# --- Initialize centroids randomly from the data points ---
centroids = X[np.random.choice(range(len(X)), k, replace=False)]
centroids_list = [centroids.copy()]

# --- Manual k-Means loop with convergence check ---
tol = 1e-3   # stop when centroids move less than this distance
max_iter = 100

for iteration in range(max_iter):
    # Step 1: Assign points to nearest centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    
    # Step 2: Recompute centroids
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    
    # Record new positions
    centroids_list.append(new_centroids.copy())
    
    # Step 3: Check for convergence (movement < tolerance)
    shift = np.linalg.norm(new_centroids - centroids)
    if shift < tol:
        print(f"Converged after {iteration+1} iterations.")
        break
    
    centroids = new_centroids

# --- Plot centroid movements with arrows ---
colors = ['tab:blue', 'tab:orange', 'tab:green']
fig, ax = plt.subplots(figsize=(8,6))

# Plot data
ax.scatter(X[:, 0], X[:, 1], s=40, c='lightgray')

for c in range(k):
    trajectory = np.array([cent[c] for cent in centroids_list])
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=colors[c], linewidth=2)
    
    # Add arrows showing movement between iterations
    for i in range(len(trajectory)-1):
        ax.annotate(
            '', xy=trajectory[i+1], xytext=trajectory[i],
            arrowprops=dict(arrowstyle='->', color=colors[c], lw=2, alpha=0.8)
        )
    
    # Mark final centroid
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=300,
               edgecolors='black', facecolors=colors[c], linewidths=2)

ax.set_title("k-Means: Centroid Movement and Convergence")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
Converged after 8 iterations.



<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_40_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}


    

    


<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b> 

- Each iteration moves centroids toward the center of their assigned points.  
- The algorithm stops when these movements become minimal.  
- k-Means is simple, efficient, and often surprisingly effective, but it assumes clusters are roughly **spherical** and of similar size.
</div>


## Problems can and do arise in unsupervised learning! 

{% capture ex %}
```python
# --- Create a simple 2D dataset ---
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)

# --- Choose the number of clusters ---
k = 3
np.random.seed(42)  # try different seeds to see different starting positions

# --- Initialize centroids randomly from the data points ---
centroids = X[np.random.choice(range(len(X)), k, replace=False)]
centroids_list = [centroids.copy()]

# --- Manual k-Means loop with convergence check ---
tol = 1e-3   # stop when centroids move less than this distance
max_iter = 100

for iteration in range(max_iter):
    # Step 1: Assign points to nearest centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    
    # Step 2: Recompute centroids
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    
    # Record new positions
    centroids_list.append(new_centroids.copy())
    
    # Step 3: Check for convergence (movement < tolerance)
    shift = np.linalg.norm(new_centroids - centroids)
    if shift < tol:
        print(f"Converged after {iteration+1} iterations.")
        break
    
    centroids = new_centroids

# --- Plot centroid movements with arrows ---
colors = ['tab:blue', 'tab:orange', 'tab:green']
fig, ax = plt.subplots(figsize=(8,6))

# Plot data
ax.scatter(X[:, 0], X[:, 1], s=40, c='lightgray')

for c in range(k):
    trajectory = np.array([cent[c] for cent in centroids_list])
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=colors[c], linewidth=2)
    
    # Add arrows showing movement between iterations
    for i in range(len(trajectory)-1):
        ax.annotate(
            '', xy=trajectory[i+1], xytext=trajectory[i],
            arrowprops=dict(arrowstyle='->', color=colors[c], lw=2, alpha=0.8)
        )
    
    # Mark final centroid
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=300,
               edgecolors='black', facecolors=colors[c], linewidths=2)

ax.set_title("k-Means: Centroid Movement and Convergence")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
Converged after 11 iterations.



<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_43_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}


    

    




### Decision Boundaries

Just like in k-NN, we can visualize how k-Means divides the feature space.

Each region in this plot corresponds to the points *closest to one centroid* —  these are called **Voronoi regions**.

- The **colored background** shows the areas where each cluster dominates.
- The **dots** are the actual data points.
- The **red Xs** are the final centroid positions after convergence.

Notice how each centroid’s region is separated by straight-line boundaries — this happens because k-Means uses **Euclidean distance** (straight-line distance).


{% capture ex %}
```python
# --- Generate example data ---
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# --- Fit k-Means ---
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# --- Create grid for decision boundaries ---
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# --- Predict cluster label for each point in the grid ---
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# --- Plot decision boundaries and data points ---
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')      # decision regions
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, edgecolor='k')  # data points
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='red', s=250, marker='X', edgecolor='black', linewidths=2,
            label='Centroids')

plt.title("k-Means Decision Boundaries (Voronoi Regions)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_45_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;"> 
{% endcapture %}
{% include codeoutput.html content=ex %}


   
    




### The Effect of Choosing k

As you change the value of k in k-means, you change the number of groups/clusters the model will search for. It has no way of knowing how many groups it should expect to see, so does exactly as told and breaks the data points into k groups. 

- With **too few clusters**, distinct groups can get merged together.  
- With **too many clusters**, k-Means starts “overfitting” and slicing up natural groups into smaller pieces.  
- The “right” k balances simplicity (fewer clusters) with accuracy (capturing true patterns).

We’ll later use the **Elbow Method** to help identify a reasonable k value quantitatively.


{% capture ex %}
```python
# --- Generate sample data ---
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# --- Try several k values ---
k_values = [2, 3, 4, 5]

fig, axes = plt.subplots(len(k_values), 1, figsize=(6, 16))

for ax, k in zip(axes, k_values):
    # Fit k-Means for each k
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Create grid for decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries and points
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, edgecolor='k')
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c='red', s=120, marker='X', edgecolor='black', linewidths=1.5)
    
    ax.set_title(f"k = {k}")

fig.suptitle("Effect of Varying Number of Clusters (k) in k-Means", fontsize=16)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_47_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    



## General Things to Rememer


### Random Initialization Matters!

k-Means starts with random centroid positions, so different initial seeds can produce different outcomes — especially if clusters overlap or data has outliers.

In the following example:

- Each run will use the same data.
- But, each run will start with a different random seed.  
- Notice how the centroids and cluster shapes vary slightly between runs.  

Modern implementations (like scikit-learn’s) handle this by running the algorithm multiple times (`n_init=10` by default) and keeping the best solution — the one with the lowest total *inertia* (within-cluster variance) which we will discuss soon.


{% capture ex %}
```python
# --- Different random seeds to illustrate initialization differences ---
seeds = [1, 2, 12, 42]

# --- Generate data ---
X, _ = make_blobs(n_samples=300, centers=6, cluster_std=1.2, random_state=10)

fig, axes = plt.subplots(len(seeds), 1, figsize=(6, 16))

for ax, seed in zip(axes, seeds):
    kmeans = KMeans(n_clusters=3, random_state=seed)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Create grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, edgecolor='k')
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c='red', s=120, marker='X', edgecolor='black', linewidths=1.5)
    
    ax.set_title(f"Random Seed = {seed}")

fig.suptitle("Effect of Random Initialization in k-Means", fontsize=16)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_49_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">    
{% endcapture %}
{% include codeoutput.html content=ex %}




    


### Feature Scaling Matters!

Both **k-Means** and **k-NN** rely on distance calculations; usually Euclidean distance:

$$
\text{Distance between points} = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \dots}
$$

When one feature has a much larger numeric range than another, its differences dominate that distance. This effectively drowns out the smaller features.

In the following example:

- The **unscaled plot** has a vertical axis with values roughly 30× larger than the horizontal.  
  The algorithm thinks “vertical distance” is more important and shapes clusters along that axis.
  
- The **scaled plot** is of the same data, but scaled so both features contribute equally.  
  This allows both features to contribute equally in claiming points.


{% capture ex %}
```python
# --- Create sample data ---
# Feature 1: small numeric range
# Feature 2: large numeric range
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
X[:, 1] = X[:, 1] * 30   # stretch 2nd feature so scales differ dramatically

# --- Fit k-Means on unscaled data ---
kmeans_unscaled = KMeans(n_clusters=3, random_state=42)
kmeans_unscaled.fit(X)
labels_unscaled = kmeans_unscaled.labels_
centroids_unscaled = kmeans_unscaled.cluster_centers_

# --- Scale the data and fit again ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_scaled = KMeans(n_clusters=3, random_state=42)
kmeans_scaled.fit(X_scaled)
labels_scaled = kmeans_scaled.labels_
centroids_scaled = kmeans_scaled.cluster_centers_

# --- Plot both results side-by-side ---
fig, axes = plt.subplots(2, 1, figsize=(6, 9))

# Left: Without scaling
axes[0].scatter(X[:, 0], X[:, 1], c=labels_unscaled, cmap='viridis', s=40, edgecolor='k')
axes[0].scatter(centroids_unscaled[:, 0], centroids_unscaled[:, 1],
                c='red', s=200, marker='X', edgecolors='black', linewidths=2)
axes[0].set_title("Without Scaling")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

# Right: With scaling
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_scaled, cmap='viridis', s=40, edgecolor='k')
axes[1].scatter(centroids_scaled[:, 0], centroids_scaled[:, 1],
                c='red', s=200, marker='X', edgecolors='black', linewidths=2)
axes[1].set_title("With Scaling (Standardized Features)")
axes[1].set_xlabel("Feature 1 (scaled)")
axes[1].set_ylabel("Feature 2 (scaled)")

fig.suptitle("Why Feature Scaling Matters for Distance-Based Algorithms", fontsize=15)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_51_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    


<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #4BA3C3;
    padding: 12px;
    border-radius: 6px;
">
<b style="color:#1b4965;">Professional Practice:</b>  
<br><br>
Always <b>standardize or normalize your features</b> before using models that rely on distance, like k-NN or k-Means.  
<br><br>
In real-world datasets, different features often have very different numeric ranges (e.g., “age” in years vs. “income” in dollars).  
If you skip scaling, one feature can silently dominate the distance metric and distort your model’s understanding of similarity.  
<br><br>
<i>Tip:</i> In practice, use <code>StandardScaler</code> (for zero mean and unit variance) or <code>MinMaxScaler</code> (for values between 0 and 1) from <code>scikit-learn</code> before training.
</div>



## How Compact Are Our Clusters? Introducing Inertia

When k-Means runs, it tries to make each cluster as **tight** as possible.   The algorithm minimizes a measure called **inertia**, which is the total *within-cluster variance*.

Think of it like this:
> “How far, on average, are points from their cluster’s center?”

Mathematically (you don’t need to memorize this):
$$
\text{Inertia} = \sum_{\text{clusters}} \hspace{0.3cm} \sum_{\text{points in cluster}} ||x - x_\text{center}||^2
$$

A **lower inertia** means the clusters are more compact — the points fit their groups better.

- Shorter lines → tighter clusters → lower inertia.  
- Longer lines → looser clusters → higher inertia.

{% capture ex %}
```python
# --- Generate example data ---
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.9, random_state=42)

# --- Try multiple k values ---
k_values = [2, 3, 4, 5]
inertias = []
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, axes = plt.subplots(len(k_values), 1, figsize=(6, 16), sharex=True, sharey=True)

for ax, k in zip(axes, k_values):
    # Fit k-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Plot clusters and inertia lines
    for i, color in enumerate(colors[:k]):
        cluster_points = X[labels == i]
        centroid = centroids[i]
        
        # Draw faint connecting lines
        for point in cluster_points:
            ax.plot([point[0], centroid[0]], [point[1], centroid[1]],
                    color=color, alpha=0.25, linewidth=1.5)
        
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=35, c=color, alpha=0.7)
        ax.scatter(centroid[0], centroid[1], s=150, c=color, edgecolors='black', marker='X', linewidths=2)
    
    ax.set_title(f"k = {k}   |   Inertia = {kmeans.inertia_:.0f}", fontsize=12)

fig.suptitle("How Changing k Affects k-Means Clustering and Inertia", fontsize=14, y=1)
plt.tight_layout()
plt.show()
```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_54_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    



### Changing k and Inertia

Notice, as `k` increases inertia always decreases. This makes sense, the more clusters you prescribe, the closer each point will be to its centroid.

Let's plot the Inertia as a function of k:



{% capture ex %}
```python
inertias = []
k_values = range(1, 10)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_train_scaled)
    inertias.append(km.inertia_)

plt.plot(k_values, inertias, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_56_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    


The "elbow" point represents a good tradeoff: adding more clusters beyond this point doesn’t improve tightness much, but increases model complexity.

<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion:</b> 
    <ul>
      <li>Where’s the “elbow” on your plot?</li>
      <li>Why does adding more clusters always reduce inertia?</li>
      <li>How could we *validate* that the clusters make sense, even without true labels?</li>
    </ul>
</div>



## How Well-Separated Are the Clusters? Introducing Silhouette Score

While **inertia** measures how tight the clusters are, **silhouette score** measures how **distinct** they are from one another.

It considers both:
- **a:** average distance to points *within* the same cluster  
- **b:** average distance to points in the *nearest neighboring cluster*

and computes:
$$
\text{Silhouette} = \frac{b - a}{\max(a, b)}
$$

The score ranges from **–1 to 1**:
- **+1:** well-separated clusters  
- **0:** overlapping clusters  
- **–1:** points likely in the wrong cluster


{% capture ex %}
```python
# --- Generate example data ---
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)

# --- Range of k values to explore ---
k_values = [2, 3, 4, 5]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# --- Create subplots (one per k) ---
fig, axes = plt.subplots(len(k_values), 1, figsize=(7, 16), sharex=True, sharey=True)
results = []

for ax, k in zip(axes, k_values):
    # Fit k-Means model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Compute metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)
    results.append((k, inertia, silhouette))
    
    # Plot points and connections
    for i, color in enumerate(colors[:k]):
        cluster_points = X[labels == i]
        centroid = centroids[i]
        
        # Lines from points to centroids (inertia)
        for point in cluster_points:
            ax.plot([point[0], centroid[0]], [point[1], centroid[1]],
                    color=color, alpha=0.25, linewidth=1.2)
        
        # Points and centroids
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=35, c=color, alpha=0.7)
        ax.scatter(centroid[0], centroid[1], s=180, c=color, edgecolors='black', marker='X', linewidths=2)
    
    # Lines between centroids (separation)
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            ax.plot([centroids[i][0], centroids[j][0]],
                    [centroids[i][1], centroids[j][1]],
                    color='black', linestyle='--', alpha=0.6, linewidth=1.8)
    
    ax.set_title(f"k = {k}   |   Inertia = {inertia:.0f}   |   Silhouette = {silhouette:.2f}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Inertia vs. Silhouette for Different k in k-Means", fontsize=14, y=0.93)
plt.tight_layout()
plt.show()

# --- Print comparison table ---
import pandas as pd
results_df = pd.DataFrame(results, columns=["k", "Inertia", "Silhouette"])
display(results_df)
```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_59_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>Inertia</th>
      <th>Silhouette</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3720.112349</td>
      <td>0.706617</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>364.473321</td>
      <td>0.846700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>314.873350</td>
      <td>0.679849</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>273.619951</td>
      <td>0.509465</td>
    </tr>
  </tbody>
</table>
</div>
{% endcapture %}
{% include codeoutput.html content=ex %}




    







### Changing k and Silhouette

Notice, as `k` increases the silhouette also changes! 

One way to find an optimal k value would be to plot the silhouette for multiple k values and see if one performs better than the others. 

In fact, this is one of the most common and reliable tools used to find a good k value to use.

Let's plot the Silhouette as a function of k:


{% capture ex %}
```python
# --- Generate example data ---
X, _ = make_blobs(n_samples=400, centers=5, cluster_std=1.0, random_state=42)

# --- Range of k values to explore ---
k_values = range(2, 10)
inertias = []
silhouettes = []

# --- Compute both metrics for each k ---
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# --- Find best k based on silhouette score ---
best_k = k_values[np.argmax(silhouettes)]
best_sil = max(silhouettes)

# --- Fit k-Means again using the best k ---
best_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
best_labels = best_model.fit_predict(X)
centroids = best_model.cluster_centers_

# --- Create figure with 2 rows: metrics + cluster plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# ======================
# Plot 1: Inertia vs Silhouette
# ======================
color_inertia = "#E15759"
color_silhouette = "#4E79A7"

# Left y-axis (Inertia)
ax1.plot(k_values, inertias, "o-", color=color_inertia, label="Inertia (↓ better)")
ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
ax1.set_ylabel("Inertia", color=color_inertia, fontsize=12)
ax1.tick_params(axis="y", labelcolor=color_inertia)

# Right y-axis (Silhouette)
ax3 = ax1.twinx()
ax3.plot(k_values, silhouettes, "s--", color=color_silhouette, label="Silhouette (↑ better)")
ax3.set_ylabel("Silhouette Score", color=color_silhouette, fontsize=12)
ax3.tick_params(axis="y", labelcolor=color_silhouette)

# Annotate best k
ax3.axvline(best_k, color=color_silhouette, linestyle=":", linewidth=2)
ax3.text(best_k + 0.1, best_sil - 0.02, f"Best k = {best_k}", color=color_silhouette)

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper right")
ax1.set_title("Choosing k: Inertia vs. Silhouette", fontsize=14)
ax1.grid(True, linestyle="--", alpha=0.6)

# ======================
# Plot 2: Best Clustering
# ======================
scatter = ax2.scatter(X[:, 0], X[:, 1], c=best_labels, cmap="tab10", s=35, alpha=0.7)
ax2.scatter(centroids[:, 0], centroids[:, 1], s=250, c="black", marker="X",
            edgecolors="white", linewidths=2, label="Centroids")
ax2.set_title(f"k-Means Clustering (Best k = {best_k}, Silhouette = {best_sil:.2f})", fontsize=13)
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_61_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    


<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion:</b> 
    <ul>
      <li>How would silhouette score change if we chose k=2 vs k=6?</li>
      <li>What does a low score suggest about our data’s structure?</li>
      <li>Why might overlapping or uneven clusters reduce the score?</li>
    </ul>
</div>



## Evaluating Clustering Without Labels

In supervised learning (like k-NN), we can directly compare predictions to known answers, and we can calculate **accuracy**, **precision**, or **recall** because we know the truth.

But in unsupervised learning (like k-Means), there are **no true labels**, the algorithm is discovering structure on its own.

So how do we know if it did a *good* job?

### Common Evaluation Metrics

**Inertia (Within-Cluster Variance):**  
    - Lower is better

**Silhouette Score:**  
    - +1 → well-clustered (distinct and compact)  
    - 0 → overlapping clusters  
    - -1 → likely misclassified  

**Visual Inspection:**  
  For 2D data, plotting the clusters and centroids can be incredibly informative.  
  Human intuition is often the best first check.



### Comparing Clusters to Known Labels (If Possible)

When evaluating clustering algorithms like **k-Means**, we often want to know **how similar** our predicted clusters are to the **true labels** (if we have them).  

That’s where something called the **Adjusted Rand Index (ARI)** comes in.

### Intuition

ARI compares **all possible pairs in the data** and asks:
> "For each pair of points, did the model place them in the same or a different cluster?"

It counts in the following manner:

- If the model and true labels place the points in the same cluster → **agreement**
- If the model and true labels place the points in a different cluster → **agreement**
- If the model and true labels disagree (one puts them in the same cluster and the other does not) → **disagreement**

The more agreements, the higher the score.

### How It’s Calculated

ARI starts from the **Rand Index (RI)**:
$$
RI = \frac{ \text{\# of agreements} }{ \text{\# of total pairs} }
$$

But the Rand Index can be **inflated by chance** by allowing even randomly selected labels to get a moderately high score.

The **Adjusted Rand Index (ARI)** corrects for by first calculating  the RI for the model, $ RI_\text{model} $, and the RI for a baseline (random labels), $ RI_\text{baseline} $.

It them performs the following calculation using those values:

$$
ARI = \frac{RI_\text{model} - RI_\text{baseline}}{1 - RI_\text{baseline}}
$$

The ARI can take on the following values, with their meaning:

| ARI Value | Interpretation |
|------------|----------------|
| **1.0** | Perfect match — identical cluster assignments |
| **0.0** | What you’d expect from random chance |
| **< 0.0** | Worse than random — systematic disagreement |

In short:  
> ARI tells you *how close* your clustering results are to the true labels, accounting for potentialy random agreements.


{% capture ex %}
```python
# Compare k-Means clusters to actual labels (if available)
clusters = kmeans.fit_predict(X_train)
ari = adjusted_rand_score(y_train, clusters)
print(f"Adjusted Rand Index (vs. true labels): {ari:.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
Adjusted Rand Index (vs. true labels): 0.409
{% endcapture %}
{% include codeoutput.html content=ex %}


    


<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion:</b> 
    <ul>
      <li>Why is this metric not always possible for real-world data? </li>
      <li>What would it mean if ARI is close to 1?</li>
      <li>Why might ARI still be low even for good-looking clusters?</li>
    </ul>
</div>



## Visualizing Multi-Feature Data in 2D

Many real-world datasets have more than two features — sometimes dozens or even hundreds. But most of our visualizations (like scatter plots and decision boundaries) can only show **two axes**.

So how can we plot data with 4, 8, or 100 features on a flat 2D screen?


### The Idea: Feature Compression

We can use mathematical tools to **compress** high-dimensional data into two new features that capture the most important patterns.

One of the most common tools for this is called **Principal Component Analysis (PCA)**.

- PCA combines the features to create compressed features that explain the **greatest variation** in the data.  
- These compressed, new features (called *principal components*) summarize how each data point varies along those directions.  
- The first two principal components often capture the majority of the structure in the dataset.
    - Generally enough to make a meaningful 2D plot.


### What That Means for Us

When you see a 2D scatter plot of multi-feature data, those axes might not be original features like “height” or “mass.”  Instead, they’re almost certainly combinations of many features.

We will not discuss PCA in any greater detail since the mathematics behind it gets a little complicated. As a data science practitioner, it is sufficient to remember the following:
> **PCA lets us see high-dimensional data in 2D by finding the two directions that best capture its structure.**


{% capture ex %}
```python
# --- Load dataset ---
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Split into train/test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Apply k-Means ---
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)

# --- Plot clusters ---
plt.figure(figsize=(7,5))
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=clusters, cmap="viridis", s=40)
plt.title("k-Means Clustering on Iris (first 2 features)")
plt.xlabel("Standardized Feature 1")
plt.ylabel("Standardized Feature 2")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_68_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    


{% capture ex %}
```python
# === Exploring the Handwritten Digits Dataset ===
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# --- Load dataset ---
digits = load_digits()
X, y = digits.data, digits.target

# --- Show sample digit images ---
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle("Sample Images from the Handwritten Digits Dataset", fontsize=14)

indices = np.random.choice(len(digits.images), 10, replace=False)
for ax, idx in zip(axes.ravel(), indices):
    ax.imshow(digits.images[idx], cmap="gray")
    ax.set_title(f"Label: {digits.target[idx]}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()

print("\n")

# --- Show the numerical pixel representation ---
example_idx = indices[0]
example_image = digits.images[example_idx]
example_label = digits.target[example_idx]

print(f"=== Numerical Matrix for Digit '{example_label}' ===")
print(example_image)


print("\n")
print(f"\nFlattened (64-feature) representation: \n{digits.data[example_idx]}")


print("\n")
# --- PCA visualization of all digits ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.7, s=30)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Handwritten Digits (64D → 2D)")
plt.legend(*scatter.legend_elements(num=10), title="Digit", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_69_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
    


    
    
    === Numerical Matrix for Digit '6' ===
    [[ 0.  0.  4. 12.  1.  0.  0.  0.]
     [ 0.  0. 14. 13.  0.  0.  0.  0.]
     [ 0.  2. 16.  3.  0.  0.  0.  0.]
     [ 0.  7. 13.  0.  0.  0.  0.  0.]
     [ 0.  7. 12.  7. 12.  6.  2.  0.]
     [ 0.  4. 15. 15. 12. 13. 11.  0.]
     [ 0.  1. 13. 16.  5. 11. 12.  0.]
     [ 0.  0.  5. 13. 16. 11.  1.  0.]]
    
    
    
    Flattened (64-feature) representation: 
    [ 0.  0.  4. 12.  1.  0.  0.  0.  0.  0. 14. 13.  0.  0.  0.  0.  0.  2.
     16.  3.  0.  0.  0.  0.  0.  7. 13.  0.  0.  0.  0.  0.  0.  7. 12.  7.
     12.  6.  2.  0.  0.  4. 15. 15. 12. 13. 11.  0.  0.  1. 13. 16.  5. 11.
     12.  0.  0.  0.  5. 13. 16. 11.  1.  0.]
    
    



<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_69_2.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    



## What Kind of Data Can (and Should) Be Used in k-Means?


k-Means is a powerful and simple clustering algorithm — but it **does not work equally well for all types of data**.  

Because it relies on *distance*, certain data types and structures fit much better than others.

### Works Well For

- **Numeric, continuous data.**  
  Data where distances between points are meaningful.  
  Examples: height and weight, pixel intensities, geographic coordinates, or sensor measurements.

- **Roughly spherical clusters.**  
  It performs best when clusters are compact and evenly sized.  
  Think of data that forms “blobs” in space rather than long, stretched-out shapes.

- **Similar scale and units across features.**  
  Since distances depend on raw values, features should be **standardized** (mean = 0, std = 1).  
  Otherwise, large-scale features (e.g., “income” vs. “years of experience”) will dominate.

- **Moderate noise or outliers.**  
  A few outliers are fine, but extreme ones can pull centroids off-center.  
  It’s often good practice to clean or trim data first.

### Use With Caution

- **Categorical or nominal data.**  
  k-Means can’t directly handle text labels like “red,” “blue,” or “green.”  
  If you must use it, these features should first be encoded numerically *and* made meaningful (e.g., via embeddings or one-hot encoding).

- **Non-spherical clusters.**  
  If clusters are elongated, curved, or overlapping, k-Means will force them into round shapes.  
  Alternatives like **density-based spatial clustering algorithm (DBSCAN)** or **Gaussian Mixture Models (GMMs)** handle these cases better.

- **Strongly skewed or correlated features.**  
  Highly correlated variables can distort the distance metric.  
  Using **PCA** before clustering can help reduce redundancy.

### Not Recommended For

- **Ordinal or rank-only data.**  
  Distances between “1st,” “2nd,” and “3rd” place don’t have equal spacing, meaning distances between then do not make sense. 

- **Data with many outliers or varying densities.**  
  One dense region and one sparse region will confuse k-Means: the dense cluster gets split, and the sparse cluster gets merged.

### Takeaway

k-Means is best thought of as a **geometry-based** algorithm. If your features can be meaningfully plotted on a numeric coordinate grid —  
and the distance between points reflects similarity — then k-Means can likely find useful clusters.

When in doubt, visualize first: if you can “see” clusters by eye in a scatter plot, k-Means probably will too.



## Limitations of Each Algorithm

No model is perfect — both **k-NN** and **k-Means** are powerful in the right context, but each has clear limitations you should recognize before using them.

### k-Nearest Neighbors (k-NN)

**Strengths**
- Simple, intuitive, and often very effective for small, low-dimensional datasets.
- No explicit “training” step — the algorithm just stores the data and compares new points to known ones.

**Limitations**
- **Computationally expensive at scale.** Every new prediction compares to all training points.
- **Sensitive to irrelevant features.** Distances can be distorted if unimportant variables are included.
- **Strongly affected by feature scaling.** Larger-range features dominate distance calculations.
- **Noisy or imbalanced data** can mislead the model — “nearest” neighbors may not be “best” neighbors.
- Choosing **k** is subjective; too small → overfitting, too large → underfitting.

### k-Means Clustering

**Strengths**
- Fast and efficient; scales well to large datasets.
- Provides simple, interpretable cluster centers.

**Limitations**
- **Assumes spherical, evenly sized clusters.** Irregular or elongated shapes cause poor results.
- **Requires k in advance.** You must guess the number of clusters (often by trial or elbow method).
- **Sensitive to initialization.** Different random seeds can yield different clusterings.
- **Vulnerable to outliers.** A single distant point can pull a centroid far from its true center.
- Works only with **numeric features**; categorical variables must be encoded first.



## Variants and Extensions to Explore on your Own

The basic algorithms you’ve learned this week are only the *starting points* — real-world ML expands them in creative ways.

### Variants of k-NN
- **Weighted k-NN:** closer neighbors count more heavily in the vote.
- **Distance metrics:** beyond Euclidean — Manhattan, Minkowski, cosine, etc.
- **Dimensionality reduction + k-NN:** PCA or t-SNE before classification to reduce noise and computation.

### Variants of k-Means
- **k-Medoids (PAM):** uses actual data points as cluster centers; more robust to outliers.
- **Mini-Batch k-Means:** faster approximation for very large datasets.
- **Fuzzy c-Means:** allows points to belong partially to multiple clusters (soft clustering).
- **Gaussian Mixture Models (GMMs):** assume clusters follow Gaussian distributions instead of hard boundaries.

### Alternatives to Explore Later
- **DBSCAN:** finds clusters of arbitrary shape; automatically detects outliers.
- **Hierarchical Clustering:** builds a tree (dendrogram) of clusters without pre-selecting k.
- **Spectral Clustering:** uses graph theory to handle complex, non-spherical structures.


{% capture ex %}
```python
# =====================================================
# Demo: When k-Means Fails (Non-Spherical Clusters)
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

sns.set(style="whitegrid", font_scale=1.1)

# --- Generate non-spherical (half-moon) data ---
X, y_true = make_moons(n_samples=500, noise=0.08, random_state=42)

# --- Scale the features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Define models ---
knn = KNeighborsClassifier(n_neighbors=5)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
dbscan = DBSCAN(eps=0.3, min_samples=5)

# --- Fit models ---
knn.fit(X_scaled, y_true)          # supervised
kmeans.fit(X_scaled)               # unsupervised
db_labels = dbscan.fit_predict(X_scaled)

# --- Predictions for visualization ---
xx, yy = np.meshgrid(
    np.linspace(X_scaled[:,0].min()-0.5, X_scaled[:,0].max()+0.5, 400),
    np.linspace(X_scaled[:,1].min()-0.5, X_scaled[:,1].max()+0.5, 400)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z_knn = knn.predict(grid_points).reshape(xx.shape)
Z_kmeans = kmeans.predict(grid_points).reshape(xx.shape)

# --- DBSCAN labeling (no prediction possible for unseen points)
# We'll color only the points it assigned to clusters
unique_db = np.unique(db_labels)
colors_db = sns.color_palette("tab10", len(unique_db))

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(6, 14))

# --- k-NN ---
axes[0].contourf(xx, yy, Z_knn, cmap="viridis", alpha=0.3)
axes[0].scatter(X_scaled[:,0], X_scaled[:,1], c=y_true, cmap="viridis", edgecolor="k", s=30)
axes[0].set_title("k-NN Classification (k=5)")
axes[0].set_xlabel("Feature 1 (scaled)")
axes[0].set_ylabel("Feature 2 (scaled)")

# --- k-Means ---
axes[1].contourf(xx, yy, Z_kmeans, cmap="cividis", alpha=0.3)
axes[1].scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans.labels_, cmap="cividis", edgecolor="k", s=30)
axes[1].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                c="red", marker="X", s=200, edgecolors="black", label="Centroids")
axes[1].set_title("k-Means Clustering (k=2)")
axes[1].legend()

# --- DBSCAN ---
for i, label in enumerate(unique_db):
    mask = db_labels == label
    if label == -1:
        color = "gray"
        label_name = "Noise"
    else:
        color = colors_db[i]
        label_name = f"Cluster {label+1}"
    axes[2].scatter(X_scaled[mask,0], X_scaled[mask,1], s=30, c=[color], label=label_name, edgecolor="k")
axes[2].set_title("DBSCAN Clustering (eps=0.3, min_samples=5)")
axes[2].legend()

plt.suptitle("When k-Means Fails: Non-Spherical Clusters", fontsize=16)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_73_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    


## One-Cell Code for the day

{% capture ex %}
```python
# =====================================================
# One-Cell: k-NN & k-Means on the Iris Dataset
# =====================================================

# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    silhouette_score, adjusted_rand_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import mode

sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# --- Load dataset ---
full_data = load_iris(as_frame=True)
X = full_data.data
y = full_data.target
target_names = full_data.target_names

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PCA for visualization ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# =====================================================
# Supervised: k-NN Classifier
# =====================================================
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)

# =====================================================
# Unsupervised: k-Means Clustering
# =====================================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

# --- Metrics ---
inertia = kmeans.inertia_
silhouette = silhouette_score(X_train_scaled, clusters)
ari = adjusted_rand_score(y_train, clusters)

# --- Match cluster labels to true labels (for accuracy-like metric) ---
cluster_labels = np.zeros_like(clusters)
for i in range(3):
    mask = (clusters == i)
    cluster_labels[mask] = mode(y_train[mask], keepdims=False).mode
cluster_acc = accuracy_score(y_train, cluster_labels)

# =====================================================
# Model Selection Visualization (Inertia & Silhouette)
# =====================================================
k_values = range(2, 10)
inertias, silhouettes = [], []

for k_val in k_values:
    model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    labels = model.fit_predict(X_train_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_train_scaled, labels))

best_k = k_values[np.argmax(silhouettes)]

# =====================================================
# Decision Boundaries (PCA projection)
# =====================================================
h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

knn_pca = KNeighborsClassifier(n_neighbors=k).fit(X_pca, y_train)
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_pca)
Z_knn = knn_pca.predict(grid_points).reshape(xx.shape)
Z_kmeans = kmeans_pca.predict(grid_points).reshape(xx.shape)

# =====================================================
# Visualization Suite
# =====================================================
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

# --- (1) k-NN Decision Boundaries ---
axes[0,0].contourf(xx, yy, Z_knn, cmap="viridis", alpha=0.3)
axes[0,0].scatter(X_pca[:,0], X_pca[:,1], c=y_train, cmap="viridis", edgecolor="k", s=50)
axes[0,0].set_title(f"k-NN Decision Boundaries (k={k})")
axes[0,0].set_xlabel("PCA Component 1")
axes[0,0].set_ylabel("PCA Component 2")

# --- (2) k-Means Cluster Boundaries ---
axes[0,1].contourf(xx, yy, Z_kmeans, cmap="cividis", alpha=0.3)
axes[0,1].scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="cividis", edgecolor="k", s=50)
axes[0,1].scatter(centers_pca[:,0], centers_pca[:,1], c="red", s=200, edgecolors="black", marker="X", label="Centroids")
axes[0,1].set_title("k-Means Cluster Boundaries (k=3)")
axes[0,1].legend()

# --- (3) Performance Comparison ---
bars = axes[1,0].bar(["k-NN Accuracy", "k-Means (Matched)", "k-Means ARI"],
                     [acc_knn, cluster_acc, ari],
                     color=["tab:blue", "tab:orange", "tab:green"], alpha=0.8)
axes[1,0].bar_label(bars, fmt="%.3f", padding=3)
axes[1,0].set_ylim(0, 1.05)
axes[1,0].set_title("Model Performance Comparison")
axes[1,0].set_ylabel("Score")

# --- (4) Inertia & Silhouette vs. k ---
color_inertia = "#E15759"
color_silhouette = "#4E79A7"
axes[1,1].plot(k_values, inertias, "o-", color=color_inertia, label="Inertia (↓)")
axes[1,1].set_xlabel("Number of Clusters (k)")
axes[1,1].set_ylabel("Inertia", color=color_inertia)
ax2b = axes[1,1].twinx()
ax2b.plot(k_values, silhouettes, "s--", color=color_silhouette, label="Silhouette (↑)")
ax2b.set_ylabel("Silhouette Score", color=color_silhouette)
axes[1,1].axvline(best_k, color=color_silhouette, linestyle=":", linewidth=2)
axes[1,1].set_title("Choosing k: Inertia vs. Silhouette")
axes[1,1].grid(True, linestyle="--", alpha=0.6)

# --- (5) k-Means Best Clustering (PCA) ---
best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_train_scaled)
best_clusters = best_kmeans.predict(X_train_scaled)
best_centroids = pca.transform(best_kmeans.cluster_centers_)
axes[2,0].scatter(X_pca[:,0], X_pca[:,1], c=best_clusters, cmap="tab10", s=40, alpha=0.8)
axes[2,0].scatter(best_centroids[:,0], best_centroids[:,1], s=250, c="black", marker="X", edgecolors="white", linewidths=2)
axes[2,0].set_title(f"Best k-Means Clustering (k={best_k})")
axes[2,0].set_xlabel("PCA Component 1")
axes[2,0].set_ylabel("PCA Component 2")

# --- (6) True Labels for Comparison ---
axes[2,1].scatter(X_pca[:,0], X_pca[:,1], c=y_train, cmap="tab10", edgecolor="k", s=40)
axes[2,1].set_title("True Labels (PCA Projection)")
axes[2,1].set_xlabel("PCA Component 1")
axes[2,1].set_ylabel("PCA Component 2")

plt.tight_layout()
plt.show()

# =====================================================
# Summary Output
# =====================================================
print("=== k-NN Classification ===")
print(f"Accuracy (k={k}): {acc_knn:.3f}")
print(classification_report(y_test, y_pred_knn, target_names=target_names))

print("\n=== k-Means Clustering ===")
print(f"Inertia (within-cluster variance): {inertia:.2f}")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Adjusted Rand Index (vs. true labels): {ari:.3f}")
print(f"Approx. Matched Accuracy: {cluster_acc:.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_75_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:80%;">

    


    === k-NN Classification ===
    Accuracy (k=5): 0.933
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        10
      versicolor       0.83      1.00      0.91        10
       virginica       1.00      0.80      0.89        10
    
        accuracy                           0.93        30
       macro avg       0.94      0.93      0.93        30
    weighted avg       0.94      0.93      0.93        30
    
    
    === k-Means Clustering ===
    Inertia (within-cluster variance): 110.20
    Silhouette Score: 0.486
    Adjusted Rand Index (vs. true labels): 0.635
    Approx. Matched Accuracy: 0.842
{% endcapture %}
{% include codeoutput.html content=ex %}









    
