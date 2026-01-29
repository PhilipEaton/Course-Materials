---
layout: jupyternotebook
title: Machine Learning Foundations – Lecture 02
course_home: /courses/machine-learning-foundations/
nav_section: lectures
nav_order: 2
---

# Lecture 02: Instance-Based & Cluster-Based Learning  

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

Machine learning models come in two broad categories: **supervised** and **unsupervised**.

The key distinction is whether the algorithm is trained using **labeled data** or not. In other words, does the model learn by being shown the “right answers,” or does it infer structure on its own by examining how the features relate to one another?

| Type | Goal | Examples | Data Has Labels? |
|------|------|----------|:----------------:|
| **Supervised** | Predict outcomes based on known examples | Classification (e.g., k-NN, Logistic Regression, Decision Trees), Regression (e.g., Linear Regression) | Yes |
| **Unsupervised** | Discover structure or patterns in the data | Clustering (e.g., k-Means), Dimensionality Reduction (e.g., PCA) | No |






### Supervised Learning

In supervised learning, the model is given input data **along with the correct answers** (called *targets* or *labels*).

The model learns a mapping from features to labels/targets by identifying patterns that connect them. Different models look for different kinds of patterns, which is why multiple supervised approaches exist.

Think of it as **teaching by example**:

> You show the model hundreds of penguins with known species labels, and it learns how to predict the species of a new, unseen penguin.

After training, we evaluate how well the model **generalizes to new data** using metrics such as **accuracy**, **precision**, and **recall**.






### Unsupervised Learning

In unsupervised learning, there are **no labels**. The model explores the data **on its own**, searching for patterns, relationships, or groupings.

As you may have guessed, different unsupervised models are sensitive to different types of groupings. For exaple, fome look for clusters based on distance, others on density, and still others on correlations between the features.

Think of it as **exploration without guidance**:

> You give the model penguin measurements with no species information, and it groups similar penguins together based only on their features.

The goal is **not prediction**, but **discovery**. Unsupervised learning helps reveal structure that may not be obvious, such as clusters, correlations, or trends in high-dimensional data. 

Since each feature represents a new dimension, real datasets often live in spaces with dozens or even hundreds of dimensions. Humans cannot visualize this directly, but machine learning models (with computers and a bit of math) can.







### In Practice

Most real-world workflows begin with **exploration** to understand and clean the data. From there, you would move toward:

- **Supervised learning**, if labels are available and prediction is the goal, or
- **Unsupervised learning**, if you want to uncover hidden structure or patterns.

Often, both approaches are used together.






### Today

- **k-NN** is a **supervised** model. It assigns labels to new data points by comparing them to other "near by" labeled examples.
- **k-Means** is an **unsupervised** model. It discovers groups by minimizing distances between both data points and clusters.
  - You, as the human analyst, must decide whether the resulting groups are meaningful and how to interpret or name them.










## What Is k-Nearest Neighbor Modeling?

k-Nearest Neighbors (k-NN) is one of the simplest and most intuitive machine learning algorithms. It works on a simple idea:

> **“To predict something about a new data point, look at its closest examples.”**

That is, 

> **“Birds of a feather flock together (in their features).”**

That’s it! There’s no complicated training. No crazy math. Nothing. Just distances and cmparing with neighbors. 








### The Basic Process

1. **Pick a number of neighbors** $k$ (for example, 3 or 5).  
2. **Find the $k$ closest points** in the training data to your new point.  
3. **Vote!**
   - For classification with labels → the new point takes the *most common* class among its neighbors.
   - For regression with numbers → the new point takes the *average* value of its neighbors.  
4. ... Profit!





### What “Learning” Means Here
k-NN doesn’t have *learned parameters* like slope in linear regression or transformers in neural networks.

It **stores** the trianing data and uses it for comparison later. 

This is why we call it an **instance-based** or **lazy** learner:
- **Instance-based**: it predicts based on nearby examples (instances).  
- **Lazy**: it doesn’t build a model until you ask it to make a prediction.




{% capture ex %}
```python
# HOW MANY NEIBHBORS?
num_neighbors = 3

# WHERE IS THE NEW POINT
x_new = -2
y_new = -2.5

# Generate a simple 2D dataset
X, y = make_blobs(n_samples=100, centers=3, random_state=2, cluster_std=1.2)

# Create a k-NN classifier and fit
knn = KNeighborsClassifier(n_neighbors = num_neighbors) # Open the model
knn.fit(X, y)                                           # "Train" the model

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
  alt="Scatter plot titled “How k-NN Classifies a New Point (k=2).” The plot shows three colored clusters of training data points in a two-dimensional feature space labeled Feature 1 (horizontal axis) and Feature 2 (vertical axis). A red star marks a new data point near the center-left of the plot. Two nearby training points are highlighted with red circles to indicate the nearest neighbors used by the k-nearest neighbors algorithm. The new point is classified based on the labels of these closest neighbors."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}



    


<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<strong>Discussion:</strong> 
    <ul>
      <li>What would happen if we changed `k` from 5 to 1?</li>
      <li>Does the new point's (the red star) predicted class depend only on its closest points?</li>
      <li>What if we added a noisy point(s) nearby, how might that change the vote?</li>
    </ul>
</div>



### Decision Boundaries

When we train a **classification model** like **k-NN**, we’re teaching the computer how to *divide* the feature space into regions that correspond to different categories (or “labels” or “classes” — yes, there are a lot of words that all mean basically the same thing. Which one people use is often context-dependent, and no one is especially strict about it).

A **decision boundary** is a curve (a line or surface) in feature space that separates these regions. On one side of the boundary, points are predicted to belong to one category; on the other side, they’re predicted to belong to a different category.

In two dimensions, we can draw these boundaries explicitly. In higher dimensions, they still exist, even if we can’t easily visualize them.

For example, if we take the three groups from the previous example, then the `k-NN` decision boundaries might look something like the plots shown below.



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
  alt="Decision boundary plot for a k-nearest neighbors classifier with k=5, showing three class regions in a two-dimensional feature space and a new point classified based on its location."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
{% endcapture %}
{% include codeoutput.html content=ex %}



 




#### Why Visualize Decision Boundaries?

Visualizing decision boundaries helps us **see how a model "thinks"**:

- It shows *how complex* or *simple* a model’s decision rules are.
    - Simple straight lines versus complex curves that wind around in odd ways.
- It reveals where *confusion or overlap* happens between classes.
    - Where our model is not going to work well.
- It helps us spot *overfitting*! 
    - Boundaries that are too complicated, wiggly and jittery, may be hypertuned to the training data. Not good!






#### What to Look For
When you plot the decision boundaries:
1. **Smoothness:** Do the boundaries look too jagged? 
    - That may mean overfitting, which is not good.
2. **Separation:** Are the classes clearly separated? 
    - That suggests good model performance.
3. **Misclassified points:** Are there training points on the “wrong” side of the line?
4. **Scalability:** Would the same boundary make sense if new data were added?
    - This takes a bit of domain knowledge.







#### Example: k-NN Decision Boundaries for Increasing k

In **k-NN**, each region in the plot represents the class the model would predict for that area of feature space.

As we’ve discussed, these boundaries are determined entirely by **distances** to the training points. Changing the value of `k` changes how much influence individual neighbors have:

- A **small value** of `k`: very detailed and very sensitive to individual data points.  
  - Produces highly irregular, “wiggly” decision boundaries and is prone to **overfitting**.

- A **large value** of `k`: much smoother and far less sensitive to individual neighbors.  
  - Produces simple decision boundaries and can **underfit** the data.

- A **medium value** of `k`: often the **sweet spot**.  
  - Common choices are somewhere between about 5 and 12 (sometimes a bit higher), depending on the size and structure of the training data.

We’ll discuss how to choose a good `k` in a bit. First need to talk about *accuracy*.

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
  alt="Three-panel plot showing k-nearest neighbors decision boundaries for k = 1, 5, and 11, demonstrating how small k produces complex boundaries while larger k yields smoother, more generalized regions."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}





    



## Evaluating Model Performance

###  Model Accuracy

When we train a machine learning model, we need a way to measure **how well it performs**. That’s where **accuracy** comes in.


#### What Accuracy Means

Accuracy measures the **proportion of correct predictions** the trained model makes on the test set. You can think of this as the score, as a percentage, the model got on the test. It did its trianing montage on the trianing data. How well did it do on the test? That is the accuracy.

> You train the model on your training set, and 
> you get the accuracy of the model's predictions by applying it to the test set.

Suppose your model correctly predicts 90 out of the 100 data points in your test set. The accuracy of the model in that test set would be:

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Size of Test Set}} = \frac{90}{100} = 0.90
$$

or **90%**.







#### How It’s Calculated
1. The model is given data it hasn’t seen before (the **test set**).  
2. For each data point in the test example, it predicts a label (e.g., Penguin A versus Penguin B).  
3. The predictions are compared to the **true labels** and are marked as right or wrong.  
4. Take the number of correct and divide by the number data points in the test set.






#### What Is Considered “Good”?

This is a tricky question, and honestly, it doesn’t really have a clean answer. How “good” a model is depends entirely on **what question you’re trying to answer** and **what the consequences of being wrong are**.

Sometimes, a model can be very useful even if its accuracy isn’t especially high. For example, imagine a medical screening test designed to flag *potential* cases of a rare disease. The model might only be 70% accurate overall, but if it reliably catches most true cases (this is called recall, as we wll see soon), it could still be extremely valuable as a first-pass filter, even if a more precise test is needed afterward.

Because of this, there’s no single number that defines a “good” accuracy, and certainly no single number that defines a “good” model. Everything depends on the **context**, the **quality of the data**, and the **problem you’re trying to solve**.

That said, it *is* helpful to have some practical **rules of thumb**. The table below gives a rough sense of how accuracy is often interpreted in practice:

| Accuracy Range | General Interpretation | Typical Contexts |
|----------------|------------------------|------------------|
| **More than 95%** | **Excellent** — or possibly *suspiciously perfect* (worth checking for overfitting) | Simple problems, very clean data, or cases where overfitting may be occurring |
| **90–95%** | **Very strong** — often “production-ready” (if it isn't overfitting) | Well-defined tasks with clean data (e.g., digit recognition, spam filtering) |
| **80–90%** | **Strong performance** — model generalizes well | Many classical ML problems (e.g., species classification, medical diagnosis) |
| **70–80%** | **Reasonably good** for real-world data; often a solid baseline | Customer churn, sentiment analysis, handwriting recognition |
| **60–70%** | **Barely acceptable** — may still be useful for noisy or subjective data | Social science, behavior prediction, human ratings |
| **Less than 60%** | **Usually poor** — model may not be learning much beyond chance | Very complex data, severe class imbalance, or poor model choice |

The key takeaway is that **accuracy alone is never the full story**. You should always think about *what kinds of mistakes the model is making*, *how costly those mistakes are to the task at hand*, and whether other metrics (precision, recall, F1, ROC-AUC) tell a more meaningful story for your specific problem.






#### Important Caveats

- **Always compare to a baseline model (a random guessing model).**
    - We'll talk about these next. 
- **Accuracy isn’t always the best metric.**
    - For imbalanced datasets (e.g., disease detection, fraud detection), accuracy can mislead.
    - Use precision, recall, or F1 score instead. 
        - As we will see, these capture how well your model identifies rare but important cases.
- **Dataset difficulty matters.**
    - Predicting whether a penguin is Adélie or Chinstrap? You might get 95%.
    - Predicting if someone will click an ad? You might be thrilled with 70%.
- **Compare multiple ML models.**
    - Often, what matters is whether one model outperforms another on the same data, not the absolute value.



<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #8EC9DC;
    padding: 12px;
    border-radius: 6px;
">
<strong style="color:#1b4965;">Professional Practice:</strong>  
<br><br>
Accuracy can make your model look better than it really is.  

As a data scientist, it’s your responsibility to ensure that the metrics you report actually reflect the model’s performance and limitations.  

<br><br>
If <strong>random guessing gives you 50%</strong>, then <strong>70% is a big win</strong>.  <br>
But <strong>if guessing gives you 90%</strong> (like in imbalanced data), then <strong>92% is not impressive</strong>.
<br><br>
For example, if <strong>99% of patients don’t have a disease you are sceening for</strong>, a model that always predicts “no disease” achieves a <strong>99 % accuracy</strong>. However, this model will always fail at detecting the people who are sick.

When the event you’re predicting is rare, <strong>accuracy alone can be misleading</strong>.  

That’s when metrics like <strong>precision</strong>, <strong>recall</strong>, and <strong>F1-score</strong> become essential.
</div>



#### Let's do it. 

The following code will run a `k=5` `k-NN` for the iris data and report the accuracy. 

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
<strong>Warning:</strong> If the data were not numeric, then we would need to encode it. Though, as we will see, we would have to think very carefully about which features make sense to use in k-NN and k-Means.
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

- The baseline can be set up in a number of different ways:
    -  always predict the **most frequent label** from the training data.
    -  **randomly assign labels** from the training data.  
- If your model performs **similarly to the baseline**, 
    - it’s not finding any meaningful patterns.
- If it performs **much better than the baseline**, 
    - it’s learning potentially meaningful structures within the data.

A **baseline model** gives us a simple reference point to evaluate how well our model learned from the data.

> Think of the baseline as the **“null hypothesis”** of machine learning:  
> if your model can’t beat it, it’s time to rethink the features being used, preprocessing that was done, and/or models being used.



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

#### Note on the various strategies for building a baseline model:

| Strategy            | What It Does                                                                                                     | Example Behavior                                                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`most_frequent`** | Always predicts the **most common class** in the training data.                                                  | If 70% of your training labels are *Setosa*, it will predict *Setosa* for every sample.                                                           |
| **`prior`**         | Makes **random predictions** according to the **overall class distribution** in the training data (ignores `X`). | If 70% of the training labels are *Setosa* and 30% are *Versicolor*, about 70% of predictions will be *Setosa* and 30% *Versicolor* (on average). |
| **`stratified`**    | Makes **random predictions** using the class distribution of **similar feature rows** in the training data.      | If among training samples with the same features as a test point, 90% are *Setosa* and 10% *Versicolor*, it will sample from that distribution.   |
| **`uniform`**       | Predicts **completely at random**, with equal probability for each class.                                        | Each class has the same chance (e.g., 33% each in a 3-class problem).                                                                             |
| **`constant`**      | Always predicts a **user-specified class** (via the `constant` parameter).                                       | You can force it to always predict *Adelie* regardless of the input.                                                                              |



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

The goal isn’t for the baseline to be “good”. The baseline is there to remind you what “bad but honest” looks like.  If your trained model doesn’t perform better than your baseline, your model hasn’t learned anything meaningful yet.
</div>




<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #8EC9DC;
    padding: 12px;
    border-radius: 6px;
">
<strong style="color:#1b4965;">Professional Practice:</strong>  
<br><br>
Always compare your machine learning model’s performance to a <strong>baseline model</strong>.
<br><br>
Baselines help you confirm that your model is actually learning something meaningful. If your model doesn’t outperform a simple baseline, it’s a signal to:
<ul>
<li>check your features and data preprocessing,</li>
<li>review whether your model is appropriate for the task, or</li>
<li>reconsider whether the problem itself is predictable with the available data.</li>
</ul>
Comparing to a baseline keeps your results honest, interpretable, and professionally defensible.
</div>








### Finding the Best `k` — The “Elbow Method”

When using **k-NN**, the number of neighbors `k` is a *hyperparameter* of the model. That means it is **not something the model learns on its own**. It’s a value we have to choose.

- If `k` is **too small** (for example, `k = 1`), the model becomes extremely sensitive to individual data points and noise, leading to **overfitting**.  
- If `k` is **too large**, the model smooths over real structure in the data, which can lead to **underfitting**.

To find a reasonable balance, we can evaluate the model across a range of `k` values and plot **accuracy versus k**.

Often, the plot shows a point where accuracy improves quickly at first and then begins to level off. This bend in the curve is called the *elbow*, and it’s frequently a good choice for `k`.

> Note: You *can* automate this process by selecting the value of `k` that maximizes a chosen performance metric (like accuracy or F1-score) using cross-validation. However, this is still a decision **you** are making. You’re telling the model, “Choose the value of `k` that best optimizes this particular criterion.” That choice reflects your priorities and the problem you’re trying to solve. It is not something the model independently figures out.



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
  alt="Line plot titled “k-NN Elbow Method (Accuracy vs. Number of Neighbors).” The horizontal axis shows the number of neighbors k ranging from 1 to 20, and the vertical axis shows classification accuracy. Accuracy increases rapidly from low values at small k, peaks around k values between approximately 6 and 12, and then declines or fluctuates slightly for larger k. The plot illustrates how model performance depends on the choice of k, with intermediate values providing the best balance between underfitting and overfitting."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
{% endcapture %}
{% include codeoutput.html content=ex %}



    


<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<strong>Discussion:</strong> 
    <ul>
      <li>Where does the accuracy begin to level off?</li>
      <li>Why does very small `k` sometimes perform worse on test data?</li>
      <li>How might this curve change with more or less noisy datasets?</li>
    </ul>
</div>

A **good rule of thumb** is to choose a medium-small `k` value. It should be large enough to smooth out noise, but small enough to preserve meaningful patterns.

In practice, values between 3 and 15 often work well, but the best `k` depends on your data’s size and complexity.

#### **General Rules of Thumb for Choosing k**

- **Avoid `k` that’s too small** (like 1–2):
    - Small `k` means the model memorizes the training data instead of learning general patterns.
    - This leads to overfitting (perfect accuracy on training, poor performance on new data).
- **Avoid `k` that’s too large**:
    - As `k` grows, each prediction includes more neighbors, blurring class boundaries.
    - The model becomes too “smooth,” leading to underfitting (missing important local patterns).

This is a bit like Goldilocks and the Three Bears...

- **Typical starting range**:
    - Many practitioners start with **odd values** between 3 and 15 (for binary classification) **to avoid ties**.
    - For multiclass problems, you can safely explore up to about $\sqrt{N}$, where $N$ is the total number of samples.
        - e.g., if you have 100 samples → try `k` = 10. (or better 9 or 11)
- **Best practice**:
    - Don’t rely on a single rule! Plot model accuracy versus `k` (like your elbow plot!)
    - Choose the smallest `k` that gives high accuracy without sharp fluctuations.








### Implementation Note: Matching `X_train` and `X_test`


When we split our dataset into **training** and **testing** sets, we’re simulating how a model will perform on *new, unseen data*.

This means all preprocessing steps like scaling, encoding, or feature selection must be applied **consistently** across both sets.




#### Common Mistake

A frequent beginner error is to “fit” the scalerizer separately on the test set, like this:

{% capture ex %}
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit the transformation to the training data
X_test_scaled = scaler.fit_transform(X_test)     # Fit the transformation to the test data
                                                 # ❌ wrong!
```
{% endcapture %}
{% include codeinput.html content=ex %}

This leaks information (called a data leak) from the test set into the model. This can result in a model that looks really good... because it got to peek at the answers.




#### Correct Approach 

Always ***fit a transformation*** only **on the training data**, then ***transformation*** to **the test data**:

{% capture ex %}
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit the transformation to the training data
X_test_scaled = scaler.transform(X_test)         # *Transform* the test data
                                                 # ✅ correct!
```
{% endcapture %}
{% include codeinput.html content=ex %}



The same rule applies for:
- **Encoding**: LabelEncoder or OneHotEncoder
- **Reduing Dimensions**: PCA or dimensionality reduction methods
- **Feature Selection**

Always fit on the trianing data and transform the test data.



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







### Confusion Matrix

As we have learned, accuracy tells us about the *overall* percentage of correct predictions, but it doesn’t tell us *what kinds* of mistakes our model is making.

A **confusion matrix** gives us a detailed breakdown:

|               | Predicted Class A | Predicted Class B | Predicted Class C |
|---------------|------------------|------------------|------------------|
| **Actual A**  | True Positive    | False Negative   | ... |
| **Actual B**  | False Positive   | True Negative    | ... |
| $\vdots$  | $\vdots$   | $\vdots$    | $\vdots$ |

Each row represents the **true class**, and each column represents the **predicted class**.

- Good: A perfect classifier would have all counts along the diagonal.  
- Bad: Off-diagonal entries indicate *misclassifications*.

There are 2 types of misclassifications:
 
- **False Positives (aka Type I Error)**: The model predicts the positive class when the true label is actually negative.
  > Example: Predicting a patient **has** a disease when they actually don’t.  
  > Example: Sample is predicted as **being** a Pig when it is actually a Cow. 
- **False Negatives (aka Type II Error)**: The model predicts the negative class when the true label is actually positive.
  > Example: Predicting a patient **doesn’t** have a disease when they actually do.  
  > Example: Sample was a Cow, but has been clasified as a Pig.

|               | Predicted Yes Disease | Predicted No Disease |
|---------------|------------------|------------------|
| **Actual Yes Disease**  | True Positive    | False Negative   |
| **Actual No Disease**  | False Positive   | True Negative    |


This comes down to how you frame the question. 
- If we are intested in if something is a Pig, then:
    - False Positive: Sample is not a Pig (it is a Cow), but has been predicted as being a Pig.
    - False Negative: Sample is a Pig, but has been predicted as being not a Pig (a Cow).

|               | Predicted Yes Pig | Predicted No Pig (a Cow) |
|---------------|------------------|------------------|
| **Actual Yes Pig**  | True Positive    | False Negative   |
| **Actual No Pig (a Cow)**  | False Positive   | True Negative    |

- If we are intested in if something is a Cow, then:
    - False Positive: Sample is not a Cow (it is a Pig), but has been predicted as being a Cow.
    - False Negative: Sample is a Cow, but has been predicted as being not a Cow (a Pig).

|               | Predicted Yes Cow | Predicted No Cow (a Pig) |
|---------------|------------------|------------------|
| **Actual Yes Cow**  | True Positive    | False Negative   |
| **Actual No Cow (a Pig)**  | False Positive   | True Negative    |



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
  alt="Confusion matrix titled “k-NN Confusion Matrix” for a three-class classification problem with classes setosa, versicolor, and virginica. The horizontal axis shows predicted classes and the vertical axis shows actual classes. All 15 setosa samples are correctly classified as setosa, and all 15 versicolor samples are correctly classified as versicolor. Of the 15 virginica samples, 13 are correctly classified as virginica and 2 are misclassified as versicolor. This indicates strong overall performance with minor confusion between the versicolor and virginica classes."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}






    



### Cross-Validation: Measuring Generalization

A single train/test split *might* give us a lucky (or unlucky) result. To get a more stable measure of performance, we can use **cross-validation**.

Cross-validation splits the data into *k folds*. For example, let's suppose we did 5-fold cross-validation. 

- The model is trained on 4 folds and tested on the remaining one. 
- Repeat until all folds have had a change to be the test set.

The average score from this folding gives us a more stable measure of how well the model generalizes to new data.


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







## Comparing Distance Metrics in k-NN

The k-NN algorithm depends on how we measure “closeness” between points. Different was of measuring distance (**distance metrics**) can produce different neighborhood shapes and, therefore, different predictions.

Let’s compare a few common ones:

| Metric | Formula | Notes |
|---------|----------|-------|
| **Euclidean** | $$ \sqrt{\sum_i (p_i - q_i)^2} $$ | Straight-line (“as the crow flies”) distance |
| **Manhattan** | $$ \sum_i \lvert p_i - q_i \rvert $$ | City-block distance — useful for grid-like or discrete data |
| **Minkowski** | $$ \left(\sum_i \lvert p_i - q_i \rvert^p \right)^{1/p} $$ | General form (p=1 → Manhattan, p=2 → Euclidean) |
| **Cosine** | $$ 1 - \frac{p \cdot q}{\lvert\lvert p \rvert \rvert\,\,\lvert\lvert q \rvert \rvert} $$ | Measures *angle* similarity; useful for text or high-dimensional data |

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

angle_deg = math.degrees(math.acos(cos_sim))

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
  alt="Illustration comparing Euclidean (straight-line) and Manhattan (grid-based) distances between two points in a two-dimensional feature space."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">

Euclidean = 4.24  
Manhattan = 6.00  
Minkowski (p=1.5) ≈ 4.76  
Cosine Distance = 0.53  
{% endcapture %}
{% include codeoutput.html content=ex %}






## Weighted k-NN: Giving Closer Points More Influence

You may be wondering why we would ever want to change the way we are measureing distance. 

Well, in standard k-NN, all neighbors contribute equally to the prediction. 

But it sometimes makes more sense to **weigh closer points more heavily**. That is,

> closer neighbors get more “voting power.”

That is where the different metrics come into play.

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
| # | Distance metric | Accuracy |
|---|-----------------|---------:|
| 0 | euclidean       | 0.911111 |
| 1 | manhattan       | 0.911111 |
| 2 | minkowski       | 0.911111 |
| 3 | cosine          | 0.777778 |

{% endcapture %}
{% include codeoutput.html content=ex %}









<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<strong>Discussion:</strong> 
    <ul>
      <li>Which metric produced the highest accuracy? </li>
      <li>Why might <strong>cosine distance</strong> behave differently than <strong>Euclidean</strong>?</li>
      <li>What kind of datasets would favor <strong>Manhattan</strong> distance?</li>
    </ul>
</div>






------





## What Is k-Means Clustering?

k-Means is an **unsupervised learning** algorithm used to find **groups (clusters)** in unlabeled data.

It looks for natural groupings of points that are **closer to each other** than to others, and assigns each point to one of **`k` clusters**.


### The k-Means Process

1. **Choose the number of clusters (k).**  
    - You decide how many groups the algorithm should find.

2. **Randomly place `k` centroids.**  
    - Each centroid begins at some initial location (often random).

3. **Assign each data point to the nearest centroid (E-step).**  
    - Every data point is assigned to its closest centroid.  

4. **Update the centroids (M-step).**  
    - Each centroid moves to the *average* position (the "center") of the points currently connected to the centroid.
    - When centroids move, some points that were previously closer to Centroid A might now be closer to Centroid B.

5. **Repeat steps 3–4** until the centroids stop moving and point groupings have settled (the solution has *converged*).

The algorithm naturally spreads the centroids out until each has a stable, non-overlapping set of points, but problems can occationally pop up.


### Intuition
k-Means alternates between:

- **E-Step (Expectation):** Assign points to the nearest cluster.  
- **M-Step (Maximization):** Move centroids to the average of their assigned points.  

It keeps doing this until everything stops moving.

That’s it!


{% capture ex %}
```python
# --- Create a simple 2D dataset ---
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)

# --- Choose the number of clusters ---
k = 3
np.random.seed(1)

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
  alt="Scatter plot titled “k-Means: Centroid Movement and Convergence.” Three clusters of gray data points are shown in a two-dimensional feature space labeled Feature 1 and Feature 2. Colored centroids are plotted with arrows indicating their movement across multiple iterations of the k-means algorithm. Each centroid moves from its initial position toward the center of its assigned cluster, illustrating how k-means iteratively updates centroid locations until convergence."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}


    

    





## Problems can and do arise in unsupervised learning! 

{% capture ex %}
```python
# --- Create a simple 2D dataset ---
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)

# --- Choose the number of clusters ---
k = 3
np.random.seed(42) # THE ONLY CHANGE IN THE PREVIOUS CODE!

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
  alt="Scatter plot titled “k-Means: Centroid Movement and Convergence,” showing a later stage of the algorithm. Centroids are now clustered tightly near the centers of their respective groups, with very short movement arrows. This indicates that the centroids have largely stabilized and the k-means algorithm has nearly converged."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}


<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<strong>Key Takeaways:</strong> 

- Each iteration moves centroids toward the center of their assigned points.  
- The algorithm stops when these movements become minimal.  
- k-Means is simple, efficient, and often surprisingly effective, but it assumes clusters are roughly **spherical** and of similar size.
</div>
    

    




### Decision Boundaries

Just like in k-NN, we can visualize how k-Means divides the feature space.

As we previously learned, each region in this plot corresponds to the points *closest to one centroid* (these have the fun name: **Voronoi regions**).

> Notice in the following example how each centroid’s region is separated by straight-line boundaries. This happens because k-Means is using a **Euclidean distance** (straight-line distance). The boundaries will change their shape depending on the distance metric being used.


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
  alt="Scatter plot titled “k-Means Decision Boundaries (Voronoi Regions).” The two-dimensional feature space is divided into three colored regions, each corresponding to the area closest to one centroid. Data points are colored by their assigned cluster, and red X symbols mark the centroid locations. The straight-line boundaries between regions illustrate how k-means assigns points based on nearest centroid distance."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;"> 
{% endcapture %}
{% include codeoutput.html content=ex %}


   
    




### The Effect of Choosing k

As you change the value of `k` in k-means, you change the number of groups/clusters the model will search for. 

The model has no way of knowing how many groups it should expect to see. So it **exactly as told** and always breaks the data points into `k` groups. 
- With **too few clusters**, distinct groups may get merged together.  
- With **too many clusters**, k-Means starts “overfitting” and slicing up natural groups into smaller pieces.  
- The “right” `k` balances simplicity (fewer clusters) with accuracy (capturing true patterns).

The **Elbow method** can be helpful in selecting a reasonable `k` value.


{% capture ex %}
```python
# --- Generate sample data ---
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# --- Try several `k` values ---
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
  alt="Series of plots illustrating k-means clustering behavior, including centroid movement during training, Voronoi decision regions, and the effect of increasing the number of clusters k on how data are partitioned."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    



## General Things to Rememer


### Random Initialization Matters!

k-Means starts with **random** centroid positions, so different random seeds can produce different outcomes; especially if clusters overlap or data has outliers.

In the following example:
- Each run will use the same data.
- But, each run will start with a different random seed.  
- Notice how the centroids and cluster shapes vary slightly between runs.  

Modern implementations (like scikit-learn’s) handle this by running the algorithm multiple times (`n_init=10`, 10 iterations, by default) and keeping the best solution — the one with the lowest total *inertia*, which we discuss below.


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
  alt="Four vertically stacked scatter plots titled “Effect of Random Initialization in k-Means,” each labeled with a different random seed (1, 2, 12, and 42). Each plot shows the same data points in a two-dimensional feature space with Feature 1 on the horizontal axis and Feature 2 on the vertical axis. Colored regions indicate k-means cluster assignments, and red X symbols mark the final centroid locations. The plots demonstrate that different random initializations can lead to different cluster boundaries and centroid positions, even when the number of clusters is the same."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">    
{% endcapture %}
{% include codeoutput.html content=ex %}




    


### Feature Scaling Matters!

Both **k-Means** and **k-NN** rely on distance calculations (usually Euclidean):

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
  alt="Two stacked scatter plots titled “Why Feature Scaling Matters for Distance-Based Algorithms.” The top plot, labeled “Without Scaling,” shows three clusters in a feature space where Feature 2 has a much larger numerical range than Feature 1, causing distance calculations to be dominated by Feature 2. The bottom plot, labeled “With Scaling (Standardized Features),” shows the same data after standardization, with both features on comparable scales. Cluster shapes and centroid placement are more balanced after scaling, illustrating why scaling is important for distance-based methods like k-means."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    


<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #4BA3C3;
    padding: 12px;
    border-radius: 6px;
">
<strong style="color:#1b4965;">Professional Practice:</strong>  
<br><br>
Always <strong>standardize or normalize your features</strong> before using models that rely on distance, like k-NN or k-Means.  
<br><br>
In real-world datasets, different features often have very different numeric ranges (e.g., “age” in years vs. “income” in dollars).  

If you skip scaling, one feature can silently dominate the distance metric and distort your model’s understanding of similarity.  
<br><br>
<em>Tip:</em> In practice, use <code>StandardScaler</code> (for zero mean and unit variance) or <code>MinMaxScaler</code> (for values between 0 and 1) from <code>scikit-learn</code> before training.
</div>







## How Compact Are Our Clusters? Introducing Inertia

When k-Means runs, it tries to make each cluster as **tight** as possible. This is done by minimizing a measure called **inertia**, which is the total *within-cluster variance*.

Think of it like this:
> “How far, on average, are points from their cluster’s center?”
**Goal**: Minimize that.

Mathematically (you don’t need to memorize this, it is just for those who are interested):

$$
\text{Inertia} = \sum_{\text{clusters}} \hspace{0.3cm} \sum_{\text{points in cluster}} \lvert\lvert x - x_\text{center} \rvert\rvert^2
$$

A **lower inertia** means the clusters are more compact.

In the following example: 
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
  alt="Four vertically stacked scatter plots titled “How Changing k Affects k-Means Clustering and Inertia,” corresponding to k values of 2, 3, 4, and 5. Each plot shows clustered data points in a two-dimensional feature space, with centroids marked by X symbols and lines connecting points to their assigned centroid. As k increases, clusters become smaller and more numerous, and the reported inertia value decreases, illustrating the tradeoff between model complexity and compactness of clusters."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    



### Changing k and Inertia

Notice, as the number of clusters are a looking for `k` increases, the inertia *always* decreases. 

This makes sense! The more clusters you demand, the closer each point will be to its newest centroid.

To see this, let's plot the inertia as a function of k:

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
  alt="Line plot titled “Elbow Method for Optimal k.” The horizontal axis shows the number of clusters k, and the vertical axis shows inertia. Inertia decreases rapidly as k increases from 1 to about 3, then declines more slowly for larger values of k. The curve forms an “elbow” shape, suggesting that k around 3 or 4 provides a reasonable balance between cluster compactness and model complexity."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    


The "elbow" point represents a good tradeoff: adding more clusters beyond this point doesn’t improve tightness.

<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<strong>Discussion:</strong> 
    <ul>
      <li>Where’s the “elbow” on your plot?</li>
      <li>Why does adding more clusters always reduce inertia?</li>
      <li>How could we *validate* that the clusters make sense, even without true labels?</li>
    </ul>
</div>








## How Well-Separated Are the Clusters? Introducing Silhouette Score

While **inertia** measures how tight the clusters are, the **silhouette** measures how **distinct** the clusters are from one another.

It considers the average distance between centroids to points:
- **a:** *within* the same cluster  
- **b:** in the *nearest neighboring cluster*

The **silhouette score** is then calculated via:

$$
\text{Silhouette} = \frac{b - a}{\max(a, b)}
$$

The silhouette ranges from **–1 to 1**:
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
  alt="The figure shows four scatter plots illustrating k-means clustering results for different numbers of clusters: k = 2, 3, 4, and 5. In each panel, data points are colored by their assigned cluster, and cluster centroids are marked with black “X” symbols. Dashed lines connect data points to their respective centroids, visually representing within-cluster distances (inertia). As k increases, clusters become smaller and more localized, inertia decreases, and cluster assignments become more granular. Each panel also reports the corresponding inertia and silhouette score, showing that silhouette score peaks at an intermediate k and declines as clusters become overly fragmented."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">


| # | `k` | Inertia      | Silhouette |
|---|---|--------------|------------|
| 0 | 2 | 3720.112349  | 0.706617   |
| 1 | 3 | 364.473321   | 0.846700   |
| 2 | 4 | 314.873350  | 0.679849   |
| 3 | 5 | 273.619951  | 0.509465   |
{% endcapture %}
{% include codeoutput.html content=ex %}




    







### Changing `k` and Silhouette

Notice, as `k` increases the silhouette changes! 

One way we could find an optimal `k` value would be to plot the silhouette for multiple `k` values and see if one performs better than the others. 

In fact, **this is one of the most common and reliable tools used to find a good `k` value to use**.

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
  alt="The figure displays a dual-axis line plot comparing inertia and silhouette score as functions of the number of clusters k. The x-axis represents k (number of clusters). The left y-axis shows inertia, which decreases monotonically as k increases. The right y-axis shows silhouette score, which initially increases, reaches a maximum near k = 4, and then decreases. A vertical dashed line highlights k = 4 as a balanced choice that minimizes inertia while maintaining a high silhouette score."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    


<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<strong>Discussion:</strong> 
    <ul>
      <li>How would silhouette score change if we chose k=2 vs k=6?</li>
      <li>What does a low score suggest about our data’s structure?</li>
      <li>Why might overlapping or uneven clusters reduce the score?</li>
    </ul>
</div>







## Evaluating Clustering Without Labels

In supervised learning (like k-NN), we can calculate **accuracy**, **precision**, or **recall** because we know the right answers.

But in unsupervised learning (like k-Means), there are **no true labels**, the algorithm is discovering structure on its own.

How do we know if it did a *good* job?

### Common Evaluation Metrics

**Inertia (Within-Cluster Variance):**  
    - Lower is better

**Silhouette Score:**  
    - Closer to 1 is better.

**Visual Inspection:**  
  - Project the data into 2 dimensions and plot the proposed clusters and centroids. This can be incredibly informative!









### Comparing Clusters to Known Labels (If Possible)

Suppose you have the true labels for the test set? Can we calculate something like the accuracy for k-means model?

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

This is essentially the accuracy of the model. The problem is that the Rand Index can be **inflated by chance** by allowing even randomly selected labels to get a moderately high score.

The **Adjusted Rand Index (ARI)** corrects for by first calculating the RI for the model, $ RI_\text{model} $, and the RI for a baseline (random labels), $ RI_\text{baseline} $.

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

# --- Calculate the ARI ---
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
<strong>Discussion:</strong> 
    <ul>
      <li>Why is this metric not always possible for real-world data? </li>
      <li>What would it mean if ARI is close to 1?</li>
      <li>Why might ARI still be low even for good-looking clusters?</li>
    </ul>
</div>








## Visualizing Multi-Feature Data in 2D

Many real-world datasets have more than two features. Sometimes there are dozens or even hundreds of features we have to incorpotate into our models. 

But most of our visualizations (like scatter plots and decision boundaries) can only show **two axes**, three at best.

How can we plot data with 4, 8, or 100 features on a flat 2D screen?





### The Idea: Feature Compression or Dimensionality Reduction

We can use mathematical tools to **compress** high-dimensional data into two "new features" that capture the most important patterns.

One of the most common tools for this is called **Principal Component Analysis (PCA)**.

- PCA combines the features to create compressed features that explain the greatest variation in the data.  
- These compressed, new features (called **principal components**) summarize how each data point varies along those directions.  
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

# --- Print numner of features ---
print(f"Number of features in data: {X_train.shape[1]:.0f}")

# --- Plot clusters using first two ---
plt.figure(figsize=(7,5))
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=clusters, cmap="viridis", s=40)
plt.title("k-Means Clustering on Iris (first 2 features)")
plt.xlabel("Standardized Feature 1")
plt.ylabel("Standardized Feature 2")
plt.show()


# --- PCA: first two principal components (fit on training data) ---
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)

# --- Plot clusters in PCA space ---
plt.figure(figsize=(7,5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=clusters, cmap="viridis", s=40)
plt.title("k-Means Clustering on Iris (First 2 Principal Components)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/output_68_0.png' | relative_url }}"
  alt="Scatter plot using the first two features in the training set to plot the clusters in 2 dimensions. There appears to be a well separated group and two that have some overlap."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">

<img
  src="{{ '/courses/machine-learning-foundations/images/lec02/PCA_cluster_plot.png' | relative_url }}"
  alt="Scatter plot using the first two principal components of the training set to plot the clusters in 2 dimensions. All three groups appear to be well separated now, though two grops are still pretty close together."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">

{% endcapture %}
{% include codeoutput.html content=ex %}




    
## Here is a fun example using PCA on some samples of hand written numerical digits.

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
  alt="The figure displays a grid of grayscale images from the handwritten digits dataset. Each image shows a small, pixelated handwritten numeral between 0 and 9. Above each image, a label indicates the true digit class. The examples highlight variability in handwriting style, thickness, and orientation, demonstrating the challenge of clustering or classifying image-based data."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
    


    
    
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
  alt="Scatter plot showing where the letter are located in a 2 dimensional projection of the 64 dimensional space."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




    



## What Kind of Data Can (and Should) Be Used in k-Means?


k-Means is a powerful and simple clustering algorithm, but it **does not work equally well for all types of data**.  

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

No model is perfect! Both **k-NN** and **k-Means** are powerful in the right context, but each has clear limitations you should recognize before using them.

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
- **Requires `k` in advance.** You must guess the number of clusters (often by trial or elbow method).
- **Sensitive to initialization.** Different random seeds can yield different clusterings.
- **Vulnerable to outliers.** A single distant point can pull a centroid far from its true center.
- Works only with **numeric features**; categorical variables must be encoded first.


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
  alt="Three stacked scatter plots on the same two-moon dataset comparing methods. Top: k-NN classification (k=5) shows a curved decision boundary that follows the crescent-shaped structure of the data. Middle: k-means clustering (k=2) shows a straight partition with two centroids marked, splitting the moons incorrectly because it assumes roughly spherical clusters. Bottom: DBSCAN clustering groups points into two curved clusters that match the moon shapes more closely, illustrating a case where k-means is a poor fit but density-based clustering works well."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}




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




    


## One-Cell Code for the day

{% capture ex %}
```python
# =====================================================
# One-Cell: k-NN & k-Means on the Iris Dataset
# =====================================================

# --- Imports ---
import numpy as np
import pandas as pd
from scipy.stats import mode
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
    cluster_labels[mask] = mode(y_train[mask], axis=None).mode

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
axes[1,0].set_xticklabels(["k-NN Accuracy", "k-Means (Matched)", "k-Means ARI"],rotation=15)
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
  alt="Six-panel figure comparing k-NN and k-means after a PCA projection. Top-left: k-NN decision regions with curved boundaries that separate the visible groups. Top-right: k-means Voronoi-style cluster regions with centroids marked, producing more rigid partitions. Middle-left: bar chart comparing scores, where k-NN accuracy is highest, k-means accuracy is lower, and k-means agreement with true labels (ARI) is lowest. Middle-right: line plot of inertia and silhouette versus k; inertia decreases as k increases while silhouette peaks at a smaller k, indicating a tradeoff in choosing k. Bottom-left: scatter plot of the “best” k-means clustering in PCA space with centroids marked. Bottom-right: scatter plot of true class labels in the same PCA space, showing that k-means clusters only partially match the true labeled structure."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">

    


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









    
