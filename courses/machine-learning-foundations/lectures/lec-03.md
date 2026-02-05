---
layout: jupyternotebook
title: Machine Learning Foundations – Lecture 03
course_home: /courses/machine-learning-foundations/
nav_section: lectures
nav_order: 3
---

# Lecture 03: Basic Regression Models


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

# --- Visualization Libraries ---
import matplotlib.pyplot as plt        # general plotting
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns                 # polished statistical plots
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# --- scikit-learn: Datasets ---
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, make_blobs, fetch_california_housing
)

# --- scikit-learn: Model Preparation ---
from sklearn.model_selection import train_test_split   # split data into train/test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures  # feature scaling & label encoding


# --- scikit-learn: Metrics ---
from sklearn.metrics import (mean_squared_error, r2_score,
                            mean_absolute_error, mean_absolute_error)

from sklearn.dummy import DummyClassifier

# --- scikit-learn: Algorithms ---
from sklearn.linear_model import LinearRegression # Linear Regression Model

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








## Linear Regression

### *Teaching Models How to Draw a Line*

Last time, we explored distance-based models like **k-NN** and **k-Means**, which made predictions by comparing asking: “Who are your closest neighbors?” and then letting those neighbors inform our test point what its label should be (majority rules) or what value it should take (average of the neightbors).

Today, we’ll shift from *comparing* to *modeling*.  

Instead of storing data and measuring distances, regression models truely **learn relationships**. That is, they find the mathematical patterns that describe how one variable changes in relation to another.

In simple terms:
> k-NN memorize. Regression models generalize.

We’ll start with **Linear Regression**, the simplest and most widely used model in data science, and see how that works with a single and multiple features. Next lecture we will extend the ideas introduced today to **Logistic Regression**, which helps us predict *categories* like “yes/no,” “survived/didn’t,” or “spam/not spam.”

As we will see, the Linear and Logistic Regressions have many similarities, but are used to acheive very different outcomes.

---

### Learning Objectives
By the end of this lecture, you will be able to:
- Explain what regression is and why it’s different from correlation.  
- Implement Linear and Logistic Regression using `scikit-learn`.  
- Interpret model coefficients, predictions, and residuals.  
- Evaluate regression performance using appropriate metrics.  
- Visualize model fits, residuals, and decision boundaries.  

---

Let’s begin by exploring how a simple line (yes, the same one from algebra class) becomes one of the most powerful tools in machine learning.







## Simple Linear Regression

### What Does Regression Do / Mean?

Before working with real data, let’s take a second to clarify what **linear regression** is actually doing.

We’ll start with a simple synthetic example where we already know the underlying relationship:

$$
y = 3x + 5 + \text{Noise}
$$

When we talk about linear regression, it’s helpful to think in terms of two components: **signal** and **noise**.

**Signal** refers to the meaningful relationship we are trying to uncover. Depending on context, this can mean:
- the true relationship that generated the data, or
- the relationship the model *has learned* from the data.

Either way, the signal is the part we actually care about.

**Noise**, on the other hand, refers to variation in the data that is *not* explained by the model. This usually comes from a few sources:
1. factors we cannot measure or do not understand well enough to include, and  
2. genuinely random fluctuations that have no underlying structure and therefore cannot be modeled in a meaningful way.

Noise is the part of the data left over after we account for everything our features can explain.

Here’s a basic fact of nature:

> Every measurement, from sensor readings to survey responses, contains noise.

Noise can arise from measurement error, natural randomness in human or environmental systems, or simply missing variables that influence the outcome. A good regression model aims to capture as much of the **signal** as possible while not overreacting to the **noise**, which does not repeat in a predictable way.


{% capture ex %}
```python
# --- Create synthetic linear data ---
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)        # 50 evenly spaced points between 0 and 10
y_true = 3 * X + 5                               # true line: y = 3x + 5
noise = np.random.normal(0, 2, X.shape)          # random noise (mean 0, std 2)
y = y_true + noise                               # observed (noisy) data

# --- Plot results ---
plt.figure(figsize=(8,5))
plt.scatter(X, y, label="Observed Data", alpha=0.7)
plt.plot(X, y_true, color="gray", linestyle="--", label="True Relationship")
plt.xlabel("X (Input Feature)")
plt.ylabel("y (Target Variable)")
plt.title("Data Radmonly Scattered Around a Line")
plt.legend()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  



{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_4_0.png' | relative_url }}"
  alt="A straight line representing the true signal in the data. Dots are scattered around the line to represent the noisey data we would be working with in real life."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}  
    



What we want is a model that **captures the signal without *overreacting* to the noise**.

We’ll train a regression model to see if it can *discover* or *extract* the signal from noisy data.

{% capture ex %}
```python
# --- Fit Linear Regression model ---
model = LinearRegression()
model.fit(X, y)

# --- Predictions ---
y_pred = model.predict(X)

# --- Plot results ---
plt.figure(figsize=(8,5))
plt.scatter(X, y, label="Observed Data", alpha=0.7)
plt.plot(X, y_true, color="gray", linestyle="--", label="True Relationship")
plt.plot(X, y_pred, color="red", label="Fitted Regression Line")
plt.xlabel("X (Input Feature)")
plt.ylabel("y (Target Variable)")
plt.title("Linear Regression: Fitting a Line to Data")
plt.legend()
plt.show()

# --- Print learned parameters ---
print(f"Model learned slope (β₁): {model.coef_[0][0]:.3f}")
print(f"Model learned intercept (β₀): {model.intercept_[0]:.3f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  



    
{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_6_0.png' | relative_url }}"
  alt="A scatter plot of the previous data with the true signal line plotted and our extracted line from fitting to the noisy data. The fit and the true lines are almost, but not quite, perfect matches."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
  
Model learned slope ($\beta_1$): 2.884
Model learned intercept ($\beta_0$): 5.129
{% endcapture %}
{% include codeoutput.html content=ex %}  
    


    



### How Good Is Our Fit?

Linear regression doesn’t just *draw a line* and hope for the best. Instead, it finds the line that **minimizes the total squared error** between the model’s predictions and the actual data points.  

> This particular choice of error (squared error) is not the *only* way to fit a line, but it is the most common and mathematically convenient. We’ll revisit alternative approaches later.

To quantify how well our model fits the data, we use several standard metrics:

- **Mean Squared Error (MSE)**: The average of the squared differences between the predicted values and the actual values. (Squaring penalizes large errors more heavily.)

- **Root Mean Squared Error (RMSE)**: The square root of the MSE, which puts the error back into the same units as $ y $. Makes RMSE easier to interpret directly.
  
- **Mean Absolute Error (MAE)**: The average of the absolute differences between predictions and actual values. (This treats all errors linearly and is less sensitive to outliers than MSE.)

- **$R^2$ (Coefficient of Determination)**: Measures how much of the variance in $ y $ is explained by the model. A value of $ R^2 = 1.0 $ corresponds to a perfect fit.

- **Adjusted $R^2$**: A modified version of $ R^2 $ that **penalizes unnecessary model complexity**. This helps prevent us from adding extra features that don’t meaningfully improve the model. You use this over the normal $R^2$ almost always.



{% capture ex %}
```python
# --- Generate example data ---
X = np.linspace(0, 10, 20)
y = 2.5 * X + 5 + np.random.normal(0, 3, 20)

# --- Define two candidate lines ---
y_pred_ok = 2 * X + 2
y_pred_best = 2.5 * X + 5

# --- Compute residuals ---
resid_ok = y - y_pred_ok
resid_best = y - y_pred_best

# --- Calculate evaluation metrics ---
mse_ok = mean_squared_error(y, y_pred_ok)
rmse_ok = np.sqrt(mse_ok)
mae_ok = mean_absolute_error(y, y_pred_ok)
r2_ok = r2_score(y, y_pred_ok)

mse_best = mean_squared_error(y, y_pred_best)
rmse_best = np.sqrt(mse_best)
mae_best = mean_absolute_error(y, y_pred_best)
r2_best = r2_score(y, y_pred_best)

# --- Create a 2x1 subplot grid (top: fits, bottom: residuals) ---
fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# === Top Row: Regression Fits ===
for ax, (y_pred, title) in zip(axes, [
    (y_pred_ok, "An Okay Fit"),
    (y_pred_best, "Best Fit (Minimized MSE)")
]):
    ax.scatter(X, y, color="gray", alpha=0.6, label="Actual Data")
    ax.plot(X, y_pred, color="red", label="Model Prediction")

    # Draw residuals as dashed lines
    for xi, yi, ypi in zip(X, y, y_pred):
        ax.plot([xi, xi], [yi, ypi], color="blue", linestyle="--", alpha=0.5)
    
    ax.set_title(title)
    ax.set_ylabel("y (Target)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Minimizing the MSE", fontsize=14)
plt.tight_layout()
plt.show()

# --- Print evaluation metrics ---
print(f"Mean Squared Error (MSE)        |  Okay: {mse_ok:.1f}   |  Best: {mse_best:.1f}")
print(f"Root Mean Squared Error (RMSE)  |  Okay: {rmse_ok:.2f}   |  Best: {rmse_best:.2f}")
print(f"Mean Absolute Error (MAE)       |  Okay: {mae_ok:.2f}   |  Best: {mae_best:.2f}")
print(f"R² Score                        |  Okay: {r2_ok:.2f}   |  Best: {r2_best:.2f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  



    
{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_8_0.png' | relative_url }}"
  alt="The top plot is a line of okay fit: it has the right slope by is shifted down from where it really should be placed. The second plot bumbs the old like up and fits the data much better."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">

Mean Squared Error (MSE)        |  Okay: 42.4   |  Best: 9.7
Root Mean Squared Error (RMSE)  |  Okay: 6.51   |  Best: 3.12
Mean Absolute Error (MAE)       |  Okay: 6.02   |  Best: 2.34
R$^2$ Score                     |  Okay: 0.33   |  Best: 0.85


{% endcapture %}
{% include codeoutput.html content=ex %}   


    


### Understanding Fit Statistics

When we build a regression model, we want to know **how well our line explains the data**. Fit statistics are numerical measures that summarize *how far off* our model is overall.

#### Errors vs. Residuals

In regression analysis, you’ll often hear the terms **error** and **residual** used to describe the difference between what the model predicts and what the data actually shows.

People frequently use these terms interchangeably, but there is a subtle distinction that is worth knowing, even if we do not always emphasize it:

- **Error:** The difference between the *true* value and the model’s *predicted* value. 
    - This is something we cannot actually know in real life, because the true underlying relationship is unknown.  
- **Residual:** The difference between the *observed* (measured) value and the model’s *predicted* value. 
    - This is what we can actually compute from our data.

In practice, since we never know the true model, we treat **residuals as our observable stand-in for errors**.

For most of this course, we will ignore this distinction to avoid unnecessary technicalities.

In fact, most non-experts will not be aware that this distinction exists at all. If you ever have to choose between the words *error* and *residual* when explaining your results to a general audience, it is usually clearer and more natural to use the word *error*.






#### 1. Mean Squared Error (MSE)
The mean squared error (MSE) can be calculated for $N$ data points via:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N}(y_{\text{data}, i} - y_{\text{predicted}, i})^2
$$

Notice $\, y_{\text{data}, i} - y_{\text{predicted}, i}\, $ is the difference between the *prediction from the model* and the *data we have been given*.

Squaring that diffence and summing over all of the points $\sum_{i=1}^{n}(y_{\text{predicted}, i} - y_{\text{data}, i})^2$ gives a measure called the sum of squares residuals (SSR), which we will be ignoring to avoid extra jargon.

Finally, dividing by the total number of data points $N$ gives us the average, or mean, of the squared errors (thus mean squared errors). 


##### Squaring the error does two things: 
1. **Emphasizes big errors** 
    - large mistakes get punished heavily.
2. **Prevents positive and negative errors from cancelling out.** 
    - Error is error, who cares if it is over or under.


##### Interpretation:
- Lower = better.  
- Zero means perfect prediction. 
    - If you get 0, be very very suspisious! You are likely overfitting to the training data.  
- Resulting units are *squared* 
    - e.g., if the target had units of dollars, then the MSE would be in dollars-squared.





#### 2. Root Mean Squared Error (RMSE)
This is calculated by taking the square-root of the MSE:

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

This brings the measure back into the same units as the original data. Very convenient! 

This makes it easier to interpret when you want to say things like, 
- “Our model is off by about ±5 mpg on average.”
- “Our model is off by about ±20 dollars on average.”

##### Interpretation:
- Lower = better.  
- Zero means perfect prediction. 
    - If you get 0, be very very suspisious! You are likely overfitting to the training data.  
- Units are the same as the target 
    - e.g., if the target had units of dollars, then the RMSE would also be in dollars.





#### 3. Mean Absolute Error (MAE)
The mean absolute error (MAE) is calculated in a similar manner as the MSE, but an absolute value is used instead of a square:

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \vert y_{\text{data}, i} - y_{\text{predicted}, i}\vert
$$

##### The absolute value does two things: 
1. **More forgiving of big errors** compared to squaring.
2. **Prevents positive and negative errors from cancelling out.** 
    - Error is error, who cares if it is over or under.

##### Interpretation: 
- Lower = better.  
- Gives a sense of the *average "distance"* between predictions and reality.
- Units are the same as the target.





#### 4. Coefficient of Determination ($R^2$)

The coefficient of determination (aka the $R^2$) is calculated via:

$$
R^2 = 1 - \frac{\text{SSR}}{\text{SST}}
$$

where SSR is the sum of squared residuals: 

$$ 
\text{SSR} = \sum_{i=1}^{n}(y_{\text{data}, i} - y_{\text{predicted}, i})^2
$$

and SST is the sum of squared totals:

$$ 
\text{SST} = \sum_{i=1}^{n}(y_{\text{data}, i} - (\text{mean of data}))^2
$$

This tells us **how much of the variation in the data our model explains**.  
- $R^2 = 1$ means a perfect fit. 
    - Because the SSR = 0
- $R^2 = 0$ means no better than predicting the mean every time.  
    - Because the SSR = SST
- Negative values can occur if the model is *worse than guessing the average!*


##### Adjusted $R^2$

Something to consider, adding more features *almost always* increases $R^2$. As the saying goes:
> More data is always better.


This is a problematic statement for multiple reasons...

1. What if the extra data is meaningless?
2. What is the extra data is messy and poorly collected?
3. etc.

I prefer the saying:
> More data is *generally* better.


the adjusted $R^2$ corrects for adding extra features into the model by penalizing potentially unnecessary complexity:

$$
R^2_{\text{adj}} = 1 - \left(1 - R^2\right) \frac{N - 1}{N - F - 1}
$$

where  
- $N$ = number of observations  
- $F$ = number of features

###### Interpretation:  
- If a new feature genuinely improves the model, Adjusted $R^2$ goes up.  
- If it only adds noise, Adjusted $R^2$ drops.  
- It’s a fairer metric when comparing models with different numbers of predictors.


In a simple linear regression where we only have one freature ($F=1$) the $R^2$ and adjusted-$R^2$ will be identical. 

However, when we start adding more features, we will want to use the adjusted $R^2$. 

*Moral of the story*: Just use the adjusted $R^2$.






##### Summary

| Metric | Description | Best Value | Notes |
|:--------|:-------------|:-------------:|:------|
| **MSE** | Average squared error | 0 | Sensitive to outliers |
| **RMSE** | Typical prediction error | 0 | Easier to interpret than the MSE |
| **MAE** | Average absolute error | 0 | Robust to outliers |
| **$R^2$** | % of variance explained | 1 | Can be negative if model fits poorly |
| **Adjusted $R^2$** | $R^2$ adjusted for the number of predictors | 1 | Penalizes unnecessary complexity; more reliable for comparing models |



**Rule of thumb:**  
- RMSE or MAE talk about *“how far off we are”*.
- Adjusted $R^2$ talks about *“how much of the pattern we’ve captured.”*


{% capture ex %}
```python
# This code demonstrates how to calculate common regression fit statistics.

# --- Generate synthetic data with some noise ---
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
true_slope, true_intercept = 3, 5
y_true = true_slope * X.squeeze() + true_intercept + np.random.normal(0, 2, 50)   # "true" line with noise

# --- Fit a Linear Regression model ---
model = LinearRegression()
model.fit(X, y_true)
y_pred = model.predict(X)

# --- Compute fit statistics ---
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# --- Compute Adjusted R² ---
n = X.shape[0]        # number of observations
p = X.shape[1]        # number of predictors (features)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# --- Display results ---
print("=== Model Fit Statistics ===")
print(f"Mean Squared Error (MSE):        {mse:.3f}")
print(f"Root Mean Squared Error (RMSE):  {rmse:.3f}")
print(f"Mean Absolute Error (MAE):       {mae:.3f}")
print(f"R² (Coefficient of Determination): {r2:.3f}")
print(f"Adjusted R²:                     {adj_r2:.3f}")

# --- Visual of predictions vs. reality ---
plt.figure(figsize=(7,5))
plt.scatter(X, y_true, label="Actual Data", color="gray", alpha=0.6)
plt.plot(X, y_pred, label="Model Prediction", color="red", linewidth=2)
plt.title("Linear Regression Fit: Actual vs. Predicted")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  




{% capture ex %}
```python
=== Model Fit Statistics ===
Mean Squared Error (MSE):          3.301
Root Mean Squared Error (RMSE):    1.817
Mean Absolute Error (MAE):         1.482
R² (Coefficient of Determination): 0.956
Adjusted R²:                       0.955
```

<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_10_1.png' | relative_url }}"
  alt="A scatter plot of the previous data with the line of best fit plotted over top."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}  




{% capture ex %}
```python
# === Comparing Fit Statistics Under Different Noise Levels ===

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Set up reproducible data ---
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
true_slope, true_intercept = 3, 5

# --- Create three datasets with different noise levels ---
noise_levels = [0, 2, 5]
datasets = []

for noise in noise_levels:
    y = true_slope * X.squeeze() + true_intercept + np.random.normal(0, noise, X.shape[0])
    datasets.append((noise, y))

# --- Container to store fit statistics ---
results = []

# --- Create plots ---
fig, axes = plt.subplots(3, 1, figsize=(6, 14), sharey=True)

for ax, (noise, y) in zip(axes, datasets):
    # Fit Linear Regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Compute metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Compute Adjusted R²
    n = X.shape[0]      # number of samples
    p = X.shape[1]      # number of predictors
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Store results for table
    results.append({
        "Noise Level": noise,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Adjusted R²": adj_r2
    })
    
    # Plot data and regression line
    ax.scatter(X, y, color="gray", alpha=0.6, label="Actual Data")
    ax.plot(X, y_pred, color="red", label="Regression Line")
    ax.set_title(f"Noise = {noise}")
    ax.set_xlabel("X")
    ax.set_ylabel("y (Target)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Effect of Noise on Regression Fit", fontsize=14)
plt.tight_layout()
plt.show()

# --- Display fit statistics as a table ---
results_df = pd.DataFrame(results)

# Round for readability
results_df = results_df.round({
    "MSE": 2,
    "RMSE": 2,
    "MAE": 2,
    "R²": 3,
    "Adjusted R²": 3
})

results_df

```
{% endcapture %}
{% include codeinput.html content=ex %}  





{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_11_0.png' | relative_url }}"
  alt="Three plots of the same line of best fit but with 3 different increasing levels of noise in the data. At the noise increases the data bets more irregualar and scattered around the line and out fit statistics change as a result. "
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">

```python
   Noise Level    MSE  RMSE   MAE     R²  Adjusted R²
0            0   0.00  0.00  0.00  1.000        1.000
1            2   2.96  1.72  1.35  0.962        0.961
2            5  25.23  5.02  4.02  0.748        0.743
```
{% endcapture %}
{% include codeoutput.html content=ex %}  








{% capture ex %}
#### Interpreting Fit Statistics in the Real World

##### 1. What “Good” Means Depends on Context
There’s no universal cutoff for “good” or “bad” MSE, RMSE, MAE, or $ R^2 $.  
It depends on:
- The *scale* of your target variable — RMSE of 5 is huge if $ y \approx 10 $, but small if $ y \approx 1000 $.  
- The *goal* of your model — predicting medical outcomes requires higher accuracy than predicting movie ratings.  
- The *cost of being wrong* — sometimes underpredicting is worse than overpredicting.

##### 2. Comparing Models
Fit statistics are most meaningful **when you compare models trained on the same data**.  
For example:
- Compare Linear vs. Polynomial regression using RMSE and $ R^2 $.  
- Or compare Linear Regression to a Tree-based model later in the course.

The *absolute* numbers matter less than *which model performs better and why*.

##### 3. Overfitting and Underfitting
A model that’s *too simple* (underfitting) will have high errors and low $ R^2 $.  

A model that’s *too complex* (overfitting) may have extremely low training error but will perform poorly on new data (low accuracy).  

That’s why we always evaluate models on **training sets *and* test sets separately**.

##### 4. When $ R^2 $ Misleads
$ R^2 $ can look great even when the model is conceptually wrong — for example, when your target variable grows over time, a simple trend line might have $ R^2 > 0.9 $ just because everything increases together.  

Always use adjusted$R^2$ and make a **residual plot** to check that errors look random. (We will look at these next)

##### 5. Why RMSE is generally better than MAE
RMSE penalizes large errors more harshly than MAE, which is generally preferred.  

If your problem can tolerate small errors but not big ones (like predicting medical dosages), RMSE gives a better result.  

If you just want a sense of *average error*, MAE is more intuitive.

##### Summary Thought:

Fit metrics are like *report cards* for your model. 

A good scientist always looks at both the **numbers** *and* the **patterns** in the residuals to understand *why* those grades look the way they do.
{% endcapture %}
{% include propractice.html content=ex %}  











### Visualizing the Error: Plotting Residuals

Residuals are the **errors** (the difference) between the predicted and actual target values:

$$
\text{Residual} = y_\text{actual} - y_\text{predicted}
$$

Plotting them helps us see:
- Whether errors are random (good!) or patterned (bad!).
- Whether the model systematically over- or under-predicts certain regions.

In a well-fit linear model, residuals should scatter randomly around zero. Like so:


{% capture ex %}
```python
# --- Plot residuals ---
residuals = y - y_pred

plt.figure(figsize=(8,4))
plt.scatter(X, residuals, color="purple", alpha=0.7)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("X (Input Feature)")
plt.ylabel("Residual (Error)")
plt.title("Residual Plot: Are Errors Random?")
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  



{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_14_0.png' | relative_url }}"
  alt="A scatter plot of points randomly scattered above and below the 0 line."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}  






{% capture ex %}
```python
# --- Set up the data vector and treu target ---
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
true_y = 3 * X + 5

noise_levels = [0.5, 2, 5]   # different noise intensities

fig, axes = plt.subplots(3, 2, figsize=(10, 14), sharex='col')

# --- Set a fixed residual range for all plots ---
resid_ylim = (-15, 15)

for i, noise in enumerate(noise_levels):
    # --- Generate data ---
    y = true_y + np.random.normal(0, noise, X.shape) # <- add in noise
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # --- Compute metrics ---
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # --- Residuals ---
    residuals = y - y_pred

    # --- Left plots: data + fit ---
    axes[i, 0].scatter(X, y, alpha=0.6, label=f"Noise σ={noise}")
    axes[i, 0].plot(X, y_pred, color="red", label=f"Fit (R²={r2:.2f})")
    axes[i, 0].set_title(f"Noise Level = {noise}")
    axes[i, 0].legend()
    axes[i, 0].set_ylabel("y")

    # --- Right plots: residuals (shared y-range for fair comparison) ---
    axes[i, 1].scatter(X, residuals, color="purple", alpha=0.6)
    axes[i, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[i, 1].set_xlabel("X")
    axes[i, 1].set_ylabel("Residual")
    axes[i, 1].set_title("Residual Distribution")
    axes[i, 1].set_ylim(resid_ylim)

plt.suptitle("Effect of Noise on Regression Fit and Residuals", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

```

{% endcapture %}
{% include codeinput.html content=ex %}  


    
{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_15_0.png' | relative_url }}"
  alt="6 plots, 2 to a row. The first plot in each row is the same line of best with but with increasingly more noisy data. The second plot in each row is the residuals. They grow bigger and less tightly bound to the 0 line as the noise gets bigger. "
  style="display:block; margin:1.5rem auto; max-width:1000px; width:75%;">
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
      <li>How does the spread change as noise increases?</li>
      <li>Are the residuals still roughly centered around zero?</li>
      <li>Would adding a new variable help — or is the noise purely random?</li>
    </ul>
</div>







#### What do Residual Plots Tell Us?

Residual plots help us *see* how well our model fits the data, and, more importantly, whether our modeling assumptions are valid.  

##### What a “Good” Residual Plot Looks Like:
A well-fit **linear** regression model should show:
- **Residuals randomly scattered** around zero (no clear shape or trend).
- **Roughly equal spread** — constant variance across all x-values.
- **No systematic curvature or funneling** — those suggest our model is missing something.

In short: **randomness** is *good*. **Structure** is *bad*.


##### Common Warning Signs

| Pattern | What It Suggests | Possible Fix |
|:-----------|:--------------|:--------------|
| **Curved pattern** | The relationship isn’t linear | Try a polynomial or nonlinear model |
| **Funnel shape (increasing or decreasing spread)** | Variance changes with x | Transform y (e.g., $ \log(y) $) or use weighted regression |
| **Clusters or bands** | Missing variable or categorical effect | Add that feature to the model |
| **All residuals above or below 0** | Model is biased or intercept is off | Re-check fitting, scaling, or intercept |


##### Consider the following examples of "bad" residual plots:


{% capture ex %}
```python
# --- Generate synthetic data for residual plot examples ---
np.random.seed(42)
x = np.linspace(0, 10, 100)

# Good fit: random scatter
y_good = 2 * x + np.random.normal(0, 1, 100)
resid_good = np.random.normal(0, 1, 100)

# Curved pattern (nonlinear relationship)
y_curve = 2 * x + 0.5 * (x - 5)**2 + np.random.normal(0, 1, 100)
resid_curve = 0.5 * (x - 5)**2 + np.random.normal(0, 1, 100) - np.mean(0.5 * (x - 5)**2)

# Funnel shape (heteroscedasticity)
resid_funnel = np.random.normal(0, x / 2 + 0.1)

# Clustered residuals (missing categorical variable)
resid_cluster = np.concatenate([
    np.random.normal(2, 0.5, 25),
    np.random.normal(-2, 0.5, 25),
    np.random.normal(1, 0.5, 25),
    np.random.normal(-1, 0.5, 25)
])

# --- Plot all four ---
fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharey=True)
examples = [
    ("Good Fit (Random Scatter)", resid_good),
    ("Curved Pattern (Nonlinear)", resid_curve),
    ("Funnel Shape (Changing Variance)", resid_funnel),
    ("Clusters (Missing Variable)", resid_cluster)
]

for ax, (title, resid) in zip(axes, examples):
    ax.scatter(x, resid, alpha=0.7)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title(title)
    ax.set_ylabel("Residuals")
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle("Examples of Residual Patterns and What They Reveal", fontsize=14, y=1)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  



{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_18_0.png' | relative_url }}"
  alt="Plot of residue plots showing what bad plots looks like as described in the previous table."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}  









### How Do we find the “Best Fit” Line?

When we fit a regression model, we’re not just *drawing a line that looks good*,  we’re finding the **exact line that minimizes the total error across all data points**.

#### 1. What Are We Minimizing?

Each data point $(x_i, y_{\text{data},i})$ has a predicted value $y_{\text{prediction},i}$ that comes from the model line.  

As we have seen, the **residual** or **error** for that point is the vertical distance between the actual and predicted value:

$$
\text{Residual}_i = y_{\text{data},i} - y_{\text{predicted},i}
$$

If we added up all the raw residuals, positive and negative errors would cancel out. Since error is error, regardless if it is positive or negtive, we square the residuals and sum them up to prevent them from canceling out:

$$
\text{SSR} = \sum_{i=1}^{n} (y_{\text{data},i} - y_{\text{predicted},i})^2
$$

This is the **Sum of Squared Residuals (SSR)**, also called **Residual Sum of Squares (RSS)**.


#### 2. The Goal: Minimize SSR

The best-fit line is the one that makes this SSR as small as possible. That is, we find the slope ($m$) and intercept ($b$) from the straight line equation we all know and love:

$$ 
y = m x + b
$$

that minimize the total squared error:

$$
\min_{m,b} \left( \sum_{i=1}^{n}(y_{\text{data},i} - (m x_i + b))^2 \right) 
$$

This process is called **Ordinary Least Squares (OLS)**, “least squares” simply means *“make the squares of the errors as small as possible.”*


#### 3. Why Squared Errors?

Squaring errors does two helpful things:
- It **penalizes large mistakes more heavily**, encouraging a line that fits all data points reasonably well.
- It ensures all errors are positive, so they don’t cancel each other out.

However, squaring also makes the method **sensitive to outliers**,  since one very large residual can dominate the SSR. If this happens you can switc hto minimizing the Mean Absolute Errors (MAE).


#### 4. Visual Intuition

Let’s look at a simple example. Each vertical dashed line is a residual — the distance between what our model predicts and what the actual value is.

Below are two possible lines: one “okay” and one “best.”  

The **best-fit line** minimizes the total area of those squared vertical distances.


{% capture ex %}
```python
# --- Generate example data ---
X = np.linspace(0, 10, 20)
y = 2.5 * X + 5 + np.random.normal(0, 3, 20)

# --- Define a few candidate slopes/intercepts ---
models = [
    (1.0, 2.0, "Initial Guess (Underfitting)"),
    (1.0, 10.0, "Better Fit (Lower SSR)"),
    (2.5, 5.0, "Best Fit (Minimized SSR)"),
]

# --- Function to compute SSR ---
def compute_ssr(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

# --- Create figure with 3 rows (one per model), 2 columns (fit | residuals) ---
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex='col')

# Set consistent residual plot range
resid_min, resid_max = -20, 20

for row, (m, b, title) in enumerate(models):
    # Compute predictions and residuals
    y_pred = m * X + b
    resid = y - y_pred
    ssr = compute_ssr(y, y_pred)
    
    # --- Left Column: Regression Fit ---
    ax_fit = axes[row, 0]
    ax_fit.scatter(X, y, color="gray", alpha=0.6, label="Actual Data")
    ax_fit.plot(X, y_pred, color="red", label=f"y = {m:.1f}x + {b:.1f}")
    
    # Draw residuals
    for xi, yi, ypi in zip(X, y, y_pred):
        ax_fit.plot([xi, xi], [yi, ypi], color="blue", linestyle="--", alpha=0.4)
    
    ax_fit.set_title(f"{title}\nSSR = {ssr:.1f}")
    ax_fit.set_ylabel("y (Target)")
    ax_fit.legend()
    ax_fit.grid(True, linestyle="--", alpha=0.5)
    
    # --- Right Column: Residual Plot ---
    ax_resid = axes[row, 1]
    ax_resid.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_resid.scatter(X, resid, color="blue", alpha=0.7)
    ax_resid.set_xlabel("X (Feature)")
    ax_resid.set_ylabel("Residuals")
    ax_resid.set_ylim(resid_min, resid_max)
    ax_resid.grid(True, linestyle="--", alpha=0.5)
    ax_resid.set_title(f"Residuals for {title}")

# --- Overall formatting ---
plt.suptitle("Minimizing the Sum of Squared Residuals: Fits and Residuals", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  



{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_20_0.png' | relative_url }}"
  alt="6 plots, 2 to a row. The left plots show the same scattered data points and three different lines of fit, each with different MSR. The right plots show the residuals and how they change as the line of best fit is zeroed in on. "
  style="display:block; margin:1.5rem auto; max-width:1000px; width:90%;">
{% endcapture %}
{% include codeoutput.html content=ex %}  













### Example: House Prices versus Square Footage

Let's use a simple linear model to predict the price of a house given only its square footage. 

We will make up our own data fo this example, but will look at some real housing data soon.


{% capture ex %}
```python
# ============================================================
# Example: Predicting House Prices from Square Footage
# ============================================================

# --- Example dataset (toy version of real-world housing data) ---
data = {
    "Square_Feet": [850, 900, 1000, 1200, 1500, 1600, 1700, 1800, 2000, 2100, 2500],
    "Price":       [130000, 140000, 155000, 180000, 210000, 225000, 240000, 250000, 275000, 290000, 330000]
}

df = pd.DataFrame(data)

# --- Separate features (X) and target (y) ---
X = df[["Square_Feet"]]   # must be 2D for sklearn
y = df["Price"]

# --- Create and fit the model ---
model = LinearRegression()
model.fit(X, y)

# --- Get model parameters ---
slope = model.coef_[0]
intercept = model.intercept_

print(f"Regression Equation: Price = {intercept:.2f} + {slope:.2f} × Square_Feet")

# --- Predict using the model ---
X_pred = np.array([[1800]])  # Example: predict price for 1800 sq ft
predicted_price = model.predict(X_pred)[0]

# --- Evaluate model fit ---
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE: +/- ${rmse:,.0f}")
print(f"R²: {r2:.3f}")
print()
print(f"Predicted Price for 1800 sq ft: ${predicted_price:,.0f} +/- ${rmse:,.0f}")
print()



# --- Plot ---
plt.figure(figsize=(7,5))
plt.scatter(X, y, color="gray", alpha=0.7, label="Actual Data")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.scatter(1800, predicted_price, color="blue", s=100, label="Prediction (1800 sq ft)")
plt.title("Predicting House Prices with Simple Linear Regression")
plt.xlabel("Square Footage")
plt.ylabel("House Price ($)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  





{% capture ex %}

    Regression Equation: Price = 31375.70 + 121.28 × Square_Feet
    RMSE: +/- $2859.40 
    R$^2$: 0.998

    Predicted Price for 1800 sq ft: $249,671 +/- $2859.40 
    
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_22_1.png' | relative_url }}"
  alt="Scatter plot of house cost versus square footage. The line of best fit is drawn as is a point representing an estimation of a house price using the model."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}  










---

## Multiple Linear Regression (MLR): When One Predictor Isn't Enough

Up to now, we’ve modeled relationships between a single feature $x$ and a target $y$, and have gotten a simple line that best fits the data.

But in the real world, most outcomes depend on *several* factors at once:
- House prices depend on **size**, **location**, **age**, ...
- Student grades depend on **hours studied**, **sleep**, **stress**, ...
- Sales depend on **price**, **advertising**, **season**, ...

That’s where **Multiple Linear Regression (MLR)** comes in.




### **The Equation**
The MLR equation is just sum of each feature multiplied by an adjustable coefficient plus a constant (the "intercept"):

$$
y_\text{prediction} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p
$$

Each predictor $x_j$ adds a new “dimension” of explanation for the target $y$:
- $\beta_0$ is the "intercept", just like the $y$-intercept
    - The predicted value of $y$ when all features are set to 0 ($x_i = 0$).
- Each $\beta_i$ represents how much $y$ changes when $x_i$ increases by one unit, **holding all other predictors constant**.
    - This assumes you have standardized your data, which you should always do! 


### **Geometric Picture**

With one feature → we fit a **line** in 2D space.  
With two features → we fit a **plane** in 3D space.  
With three or more → we fit a **hyperplane** (something we can’t see directly, but can still model mathematically).

The goal is still the same:  
> Find the parameters $\beta_0, \beta_1, \beta_2, \dots$ that minimize the **Sum of Squared Residuals (SSR)**.

Otherwise, we will use the same fit statistics and the same concepts when working with this model!

### **Visualizing a Plane Fit**

Let’s create an example dataset with two features ($x_1$ and $x_2$) that both influence the target $y$.  

We’ll fit a regression plane and see how it captures the trend in 3D space.


{% capture ex %}
```python
# === Example: Visualizing Multiple Linear Regression (2 Predictors) ===
# --- Generate synthetic data ---
np.random.seed(42)
n = 100
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)
# True relationship: y = 3*x1 + 2*x2 + 5 + noise
y = 3*x1 + 2*x2 + 5 + np.random.normal(0, 3, n)

# Combine predictors into a single feature matrix
X = np.column_stack((x1, x2))

# --- Fit the Multiple Linear Regression model ---
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# --- Create a grid for the regression plane ---
x1_range = np.linspace(x1.min(), x1.max(), 20)
x2_range = np.linspace(x2.min(), x2.max(), 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
y_grid = model.intercept_ + model.coef_[0]*x1_grid + model.coef_[1]*x2_grid

# --- Plot the data points and regression plane ---
fig = plt.figure(figsize=(6, 4.75))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, x2, y, color='gray', alpha=0.6, label='Actual Data')
ax.plot_surface(x1_grid, x2_grid, y_grid, color='red', alpha=0.4, label='Regression Plane')

ax.set_xlabel("Feature 1 (x1)")
ax.set_ylabel("Feature 2 (x2)")
ax.set_zlabel("Target (y)")
ax.set_title("Multiple Linear Regression: Fitting a Plane in 3D Space")
plt.legend()
plt.show()

# --- Display model coefficients ---
print("Model Coefficients:")
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"β₁ (x₁): {model.coef_[0]:.2f}")
print(f"β₂ (x₂): {model.coef_[1]:.2f}")

```
{% endcapture %}
{% include codeinput.html content=ex %}  




{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_24_0.png' | relative_url }}"
  alt="A plane of best fit in 3D space."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">


Model Coefficients:  
Intercept ($\beta_0$): 4.73  
$\beta_1$ ($x_1$): 2.90  
$\beta_2$ ($x_2$): 2.22
{% endcapture %}
{% include codeoutput.html content=ex %}  








### Model Evaluation: MSE, R², and Adjusted R²

We have covered these fit statistics in details already, and they mean the same things they did previously. 

Let's look at an example: 


{% capture ex %}
```python
# === Example: Evaluating Model Fit with MLR + Equation Display ===
# --- Load dataset ---
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns

# --- Split into training and test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Standardize features (important for multivariate regression) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Fit Linear Regression model ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# --- Compute metrics ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
n, p = X_test.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# --- Create a readable multi-line regression equation ---
coeffs = model.coef_
intercept = model.intercept_

print("\n=== Regression Equation ===")
print("ŷ = ")

for name, coef in zip(feature_names, coeffs):
    print(f"   {coef:>8.3f} × {name}")

print(f" + {intercept:>8.3f}  (Intercept)\n")

# --- Display model performance metrics ---
print("=== Model Fit Statistics ===")
print(f"Mean Squared Error (MSE):      {mse:.3f}")
print(f"R² (Coefficient of Determination): {r2:.3f}")
print(f"Adjusted R²:                   {adj_r2:.3f}")

# --- Visual comparison: Actual vs Predicted ---
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color="teal", alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--", label="Perfect Fit"
)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted Values (California Housing Data)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


```
{% endcapture %}
{% include codeinput.html content=ex %}  


    
 

{% capture ex %}
   === Regression Equation ===  
    ŷ =   
          0.854 × MedInc  
          0.123 × HouseAge  
         -0.294 × AveRooms  
          0.339 × AveBedrms  
         -0.002 × Population  
         -0.041 × AveOccup  
         -0.897 × Latitude  
         -0.870 × Longitude  
     +    2.072  (Intercept)  
    
    === Model Fit Statistics ===  
    Mean Squared Error (MSE):      0.556  
    R² (Coefficient of Determination): 0.576  
    Adjusted R²:                   0.575  


<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_26_1.png' | relative_url }}"
  alt="Scatter plot of the predicted versus actual median housing prices with the perfect fit line plotted on top. "
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}  







###  Overfitting vs. Underfitting

When we train a regression model with multiple features, we’re trying to find the right balance between **bias** and **variance**. That is, how well the model captures real patterns without memorizing noise.

- **Underfitting** happens when the model is *too simple* to capture the underlying structure of the data.  
  - ***Example***: fitting a straight line to data that clearly curves.  
  - ***Symptoms***: high error on *both* training and test data.  
  - ***Analogy***: using a ruler to trace a mountain range; you miss the peaks and valleys.

- **Overfitting** happens when the model is *too complex*, capturing random fluctuations and noise.  
  - ***Example***: fitting a 10th-degree polynomial to 20 data points.  
  - ***Symptoms***: very low training error ***but*** high test error.  
  - ***Analogy***: connecting every dot in the training set, even when it doesn’t reflect reality.

**Goal:** Find the “Goldilocks zone.” 
- Not too simple, not too complex. 
- Where the model generalizes well to *new* data.

We can visualize this by fitting different degrees of polynomial regression to the same dataset and comparing how they behave on unseen data.


{% capture ex %}
```python
# === Visualizing Underfitting vs. Overfitting ===
# --- Generate synthetic data (nonlinear) ---
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y_true = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# --- Split into training and test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

# --- Define polynomial degrees to compare ---
degrees = [1, 3, 15]

fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharey=True)
x_plot = np.linspace(0, 10, 200).reshape(-1, 1)

for ax, d in zip(axes, degrees):
    # Transform features into polynomial terms
    poly = PolynomialFeatures(degree=d)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    # Fit regression model
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred_train = model.predict(X_poly_train)
    y_pred_test = model.predict(X_poly_test)
    
    # Evaluate errors
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    # Plot model predictions
    y_plot = model.predict(poly.transform(x_plot))
    ax.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.7)
    ax.scatter(X_test, y_test, color="orange", label="Test Data", alpha=0.7)
    ax.plot(x_plot, np.sin(x_plot), "k--", label="True Function")
    ax.plot(x_plot, y_plot, "r", label=f"Model (degree={d})")
    
    ax.set_title(f"Degree={d}\nTrain MSE={train_mse:.3f}, Test MSE={test_mse:.3f}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

plt.suptitle("Underfitting vs. Overfitting: The Bias–Variance Tradeoff", fontsize=14)
plt.tight_layout()
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  


{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_28_0.png' | relative_url }}"
  alt="Three plots of the same data follwoing a sine wave. The plots change in that the model being used increases in complexity until it perfectly fits the points."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}  









### Bias–Variance Tradeoff

#### Why It Matters

When your model doesn’t perform well, there are two main kinds of mistakes it might be making:

- **Bias**: Systematic error. Your model is too simple and misses important relationships.
- **Variance**: Random error. Your model is too complex and reacts too much to noise in the data.

The Bias–Variance Tradeoff describes the tension between these two forces: improving one often worsens the other.

### The Bias–Variance Tradeoff Curve

Let's take the previous polynominal model we considered just above:
- Degree 1 (a straight line): too simple → high bias, low variance
- Degree 15: too wiggly → low bias, high variance

When you plot the training error and test error as model complexity increases, you typically get a curve like this: 

| Model Complexity | Training Error | Test Error | Interpretation              |
| ---------------- | -------------- | ---------- | --------------------------- |
| Low              | High           | High       | Underfitting (high bias)    |
| Moderate         | Low            | Lowest     | Ideal balance               |
| High             | Very Low       | High       | Overfitting (high variance) |

We are looking for low error in both out training set *and* test set results.

Below we have plotted the MSE for the training and test set error with a polynominal model of increasing degree (more terms):

{% capture ex %}
```python
# === Bias–Variance Tradeoff Curve ===
degrees = range(1, 16)
train_errors, test_errors = [], []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model = LinearRegression().fit(X_poly_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_poly_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_poly_test)))

plt.figure(figsize=(8,5))
plt.plot(degrees, train_errors, 'bo-', label='Training Error')
plt.plot(degrees, test_errors, 'ro-', label='Test Error')
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel("Mean Squared Error")
plt.title("Bias–Variance Tradeoff: Finding the Sweet Spot")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

```
{% endcapture %}
{% include codeinput.html content=ex %}  


{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_30_0.png' | relative_url }}"
  alt="A plot of the training error and the test error as a function of the model complexity. The more comples the model, the better both get untill the test error begins to degrade and get worse. The best model fall in the region where both test and training error are small."
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">
{% endcapture %}
{% include codeoutput.html content=ex %}  





#### Interpreting Model Complexity

As you look at the bias–variance curve, notice how the two error lines behave:

- The **training error** steadily decreases as model complexity increases; more flexibility always helps the model fit the training data.
- The **test error** initially decreases but eventually *rises again* as the model starts fitting noise rather than signal.


<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaway:</b> 

A model with the *lowest test error* (near the bottom of the red curve above) achieves the best balance between bias and variance — it captures the essential structure without memorizing random fluctuations.
</div>


<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion:</b> 
    <ul>
      <li>Which polynomial degree achieved the best generalization?</li>
      <li>How might this concept apply when using k-NN (where “complexity” isn’t degree but number neighbors)?</li>
      <li>What would happen if we had more training data? Less noise?</li>
    </ul>
</div>








### Model Selection for Multiple Linear Regression

Adding features to a model tends to improve *fit*; at least on the training data, as we just saw. 

But a “better fit” isn’t always a *better model*.

#### Why Model Selection Matters
- Every new predictor adds complexity.
- Some features contribute little information or duplicate others.
- More complexity increases the risk of **overfitting** (great on training data, bad on new data).

We want a model that balances **accuracy** and **simplicity**.

#### Common Tools for Model Selection

| Tool | What It Measures | Goal |
|:-----|:------------------|:-----|
| **Adjusted R²** | Adjusts for number of predictors | Higher = better |
| **p-values** | Tests whether each coefficient is significantly different from zero | Keep only significant features |
| **AIC / BIC** | Penalize unnecessary complexity | Lower = better |
| **VIF** | Checks for multicollinearity (redundant predictors) | < 5 is usually acceptable |



##### Statistical Significance of Coefficients (p-values)

When we fit a multiple regression model, we get one equation, but many questions:
- Which features *really matter*?  
- Which ones are just noise?

That’s where **p-values** come in.

Each coefficient in a regression has its own hypothesis test:

$$
H_0: \beta_i = 0 \quad\text{(no relationship between feature $x_i$ and $y$)}  
$$

$$
H_a: \beta_i \neq 0 \quad\text{(feature $x_i$ contributes to predicting $y$)}
$$

- A **small p-value** (< 0.05 by convention) means the feature’s effect is statistically significant—unlikely due to chance.
    - Keep!
- A **large p-value** means we can’t confidently say that feature matters.
    - Remove!

###### Why it matters
In model selection, we want to:
- Keep features with *low p-values* (significant predictors).  
- Consider dropping those with *high p-values* (weak or redundant predictors).  
- Remember: significance doesn’t always mean importance—look at context, effect size, and correlation too.




##### A Few New Diagnostics

In addition to error-based fit statistics, there are a few other quantities that show up in multiple factor regression analysis. These do **not** tell us whether a model is “right” or “wrong,” but instead help us reason about *model structure* and *model choice*.

###### Variance Inflation Factor (VIF)

**Variance Inflation Factor (VIF)** is a way to check whether input features are **too closely related to each other**.

If two (or more) input variables carry very similar information, the model has trouble deciding how much credit to give to each one. This does not necessarily hurt prediction accuracy, but it **makes the model harder to interpret**.

- Low VIF: predictors are mostly independent of each other  
- High VIF: predictors overlap a lot (possible multicollinearity)

VIF is mainly an **interpretability diagnostic**, not a measure of model quality.




###### Akaike Information Criterion (AIC)

**AIC** is a tool for comparing multiple models that are fit to the *same dataset*.

It balances two competing goals:
- fitting the data well
- keeping the model simple

AIC rewards models that fit well, but **penalizes models for using too many parameters**.

- Lower AIC is better  
- AIC values are only meaningful **relative to each other**, not on their own

You should not interpret AIC as an “accuracy score.” It is a comparison tool.



###### Bayesian Information Criterion (BIC)

**BIC** works very similarly to AIC, but it **penalizes complexity more strongly**.

This means BIC is more conservative:
- it is quicker to favor simpler models
- it is more skeptical of small improvements in fit

- Lower BIC is better  
- Like AIC, BIC is only used to compare models fit to the same data

###### Big Picture

- **VIF** asks: *Are my input features too redundant?*  
- **AIC** asks: *Is this model a good balance of fit and simplicity?*  
- **BIC** asks: *Is a simpler model likely the better explanation?*

None of these tell you “the correct model.” They help you **think more clearly about tradeoffs** when building and comparing models.


{% capture ex %}
```python
# === Example: Evaluating Coefficient Significance (with Standardization) ===

import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Generate synthetic data ---
np.random.seed(42)
n = 100
X1 = np.random.normal(5, 2, n)
X2 = np.random.normal(10, 3, n)
X3 = np.random.normal(50, 10, n)

# Target depends mainly on X1 and X2 — X3 adds mostly noise
y = 3.0 * X1 + 1.5 * X2 + np.random.normal(0, 8, n)

df = pd.DataFrame({
    "Feature_1": X1,
    "Feature_2": X2,
    "Feature_3": X3,
    "Target": y
})

# --- Standardize the feature columns ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Feature_1", "Feature_2", "Feature_3"]])
X_scaled = pd.DataFrame(
    X_scaled,
    columns=["Feature_1", "Feature_2", "Feature_3"]
)

# --- Fit Multiple Linear Regression using standardized features ---
X = sm.add_constant(X_scaled)  # add intercept
y = df["Target"]

model = sm.OLS(y, X).fit()

# --- Display regression results ---
print(model.summary())

# --- Compute Variance Inflation Factor (VIF) ---
# VIF is computed using the design matrix *without* the target
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

vif_data

```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 Target   R-squared:                       0.376
    Model:                            OLS   Adj. R-squared:                  0.356
    Method:                 Least Squares   F-statistic:                     19.24
    Date:                                   Prob (F-statistic):           7.52e-10
    Time:                                   Log-Likelihood:                -335.41
    No. Observations:                 100   AIC:                             678.8
    Df Residuals:                      96   BIC:                             689.2
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         30.3320      0.707     42.917      0.000      28.929      31.735
    Feature_1      4.1398      0.726      5.700      0.000       2.698       5.582
    Feature_2      3.9797      0.713      5.578      0.000       2.563       5.396
    Feature_3      0.2322      0.720      0.322      0.748      -1.197       1.661
    ==============================================================================
    Omnibus:                        1.353   Durbin-Watson:                   1.821
    Prob(Omnibus):                  0.508   Jarque-Bera (JB):                1.317
    Skew:                           0.169   Prob(JB):                        0.518
    Kurtosis:                       2.551   Cond. No.                         1.27
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

```python
    Feature	    VIF
0	const	    1.000000
1	Feature_1	1.056184
2	Feature_2	1.019085
3	Feature_3	1.037920
```
{% endcapture %}
{% include codeoutput.html content=ex %}  












#### How to Read the Summary Output

Look for:
- **coef** – estimated effect of each variable
- **std err** – uncertainty of that estimate
- $\mathbf{P > \vert t \vert}$ – This is the p-value for that coefficient

###### Example Interpretation
Here Feature 3 shows:

$$
P>\vert t \vert = 0.748 
$$

then there’s a 74.8% chance we’d see this effect even if there were **no** *true relationship*.

That’s far from significant! Consider dropping Feature 3.

On the other hand, if Feature 1 has:

$$
P>\vert t \vert = 0.000
$$

It’s highly significant, suggesting a genuine relationship with the target.

##### Remember
- p-values only suggests that there is likely a relationship between the feature and the target.
    - Small p-values $\ne$ large impact — a weak predictor can still be significant if the sample is large.
    - High p-values $\ne$ useless — sometimes they become significant when correlated variables are removed.
- The impact or "importance" of a feature is given by the coefficient
    - This is assuming to scaled the data so that ever feature has a mean of 0 and a standard deviation of 1.
    - This is done using the `StandardScaler()` function.


Statistical significance is a *clue*, not a command.  

Use it alongside model metrics and practical reasoning.



<!-- Reflection -->
<div style="
    background-color: #fff7e6;
    border-left: 6px solid #e28f41;
    padding: 10px;
    border-radius: 5px;
">
<b>Discussion:</b> 
    <ul>
      <li>Which features in your model are statistically significant?</li>
      <li>Do any coefficients have large magnitude but high p-value?</li>
      <li>How might correlated features influence these results?</li>
      <li>Should we *always* remove non-significant predictors?</li>
    </ul>
</div>

<br>






#### Example of building a MLR

Let's suppose we have multiple features and we want to figure out how to build a MLR model out of them.

First, let's construct some fake data we can play around with:

{% capture ex %}
```python
# --- Create a synthetic dataset ---
np.random.seed(42)
n = 100

# Independent variables (predictors)
X1 = np.random.normal(5, 2, n)         # moderately correlated with y
X2 = np.random.normal(10, 3, n)        # moderately correlated with y
X3 = np.random.normal(50, 10, n)       # weakly correlated with y
X4 = np.random.normal(50, 10, n)       # weakly correlated with y

# Target variable (depends mostly on X1 and X2)
y = 3.2*X1 + 1.7*X2 + np.random.normal(0, 8, n)

# Combine into DataFrame
df = pd.DataFrame({
    "Feature_1": X1,
    "Feature_2": X2,
    "Feature_3": X3,
    "Feature_4": X4,
    "Target": y
})

df.head()
```
{% endcapture %}
{% include codeinput.html content=ex %}  



{% capture ex %}
| Index | Feature_1 | Feature_2 | Feature_3 | Feature_4 | Target |
|------:|----------:|----------:|----------:|----------:|-------:|
| 0 | 5.993428 | 5.753888 | 53.577874 | 41.710050 | 16.205159 |
| 1 | 4.723471 | 8.738064 | 55.607845 | 44.398190 | 25.174817 |
| 2 | 6.295377 | 8.971856 | 60.830512 | 57.472936 | 35.439312 |
| 3 | 8.046060 | 7.593168 | 60.538021 | 56.103703 | 39.031622 |
| 4 | 4.531693 | 9.516143 | 36.223306 | 49.790984 | 27.078338 |

{% endcapture %}
{% include codeoutput.html content=ex %}  





Now we can 
- store the features as x_full,
- store the target as y, and 
- fit a MLR model using all of the features and see what the model looks like.

{% capture ex %}
```python
# --- Separate features and target ---
X_full = df[["Feature_1", "Feature_2", "Feature_3", "Feature_4"]]
X_full = sm.add_constant(X_full) # <- Adds the unknown constant (4 features + constant)
y = df["Target"]
```
{% endcapture %}
{% include codeinput.html content=ex %}  


Let's fit a MLR model using all of the features and see what the model looks like:

{% capture ex %}
```python
model_full = sm.OLS(y, X_full).fit()
print(model_full.summary())
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 Target   R-squared:                       0.474
    Model:                            OLS   Adj. R-squared:                  0.452
    Method:                 Least Squares   F-statistic:                     21.44
    Date:                                   Prob (F-statistic):           1.26e-12
    Time:                                   Log-Likelihood:                -350.43
    No. Observations:                 100   AIC:                             710.9
    Df Residuals:                      95   BIC:                             723.9
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        -11.2999      7.513     -1.504      0.136     -26.216       3.616
    Feature_1      2.9231      0.477      6.127      0.000       1.976       3.870
    Feature_2      2.2447      0.293      7.660      0.000       1.663       2.826
    Feature_3     -0.0691      0.078     -0.886      0.378      -0.224       0.086
    Feature_4      0.1996      0.095      2.093      0.039       0.010       0.389
    ==============================================================================
    Omnibus:                        0.508   Durbin-Watson:                   1.768
    Prob(Omnibus):                  0.776   Jarque-Bera (JB):                0.649
    Skew:                           0.141   Prob(JB):                        0.723
    Kurtosis:                       2.723   Cond. No.                         669.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
{% endcapture %}
{% include codeoutput.html content=ex %}  





From this report we can see that Feature 3 is unimportant to the present model:

- consider the p-values
- compare the size of the coeffcient estimations

**Note**: When building a model, only **change one thing at a time** and then rerun the model. 

We will only remove Feature 3 right now, and assess if there is anything else we would like to remove from there. 

{% capture ex %}
```python
# --- Separate features and target ---
X_full = df[["Feature_1", "Feature_2", "Feature_4"]]
X_full = sm.add_constant(X_full) # <- Adds the unknown constant (3 features + constant)
y = df["Target"]

model_full = sm.OLS(y, X_full).fit()
print(model_full.summary())
```
{% endcapture %}
{% include codeinput.html content=ex %}  


{% capture ex %}
```python
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 Target   R-squared:                       0.470
    Model:                            OLS   Adj. R-squared:                  0.454
    Method:                 Least Squares   F-statistic:                     28.39
    Date:                                   Prob (F-statistic):           3.15e-13
    Time:                                   Log-Likelihood:                -350.84
    No. Observations:                 100   AIC:                             709.7
    Df Residuals:                      96   BIC:                             720.1
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        -14.2968      6.701     -2.134      0.035     -27.598      -0.995
    Feature_1      2.8426      0.468      6.077      0.000       1.914       3.771
    Feature_2      2.2472      0.293      7.678      0.000       1.666       2.828
    Feature_4      0.1968      0.095      2.067      0.041       0.008       0.386
    ==============================================================================
    Omnibus:                        0.785   Durbin-Watson:                   1.773
    Prob(Omnibus):                  0.675   Jarque-Bera (JB):                0.908
    Skew:                           0.167   Prob(JB):                        0.635
    Kurtosis:                       2.673   Cond. No.                         431.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
{% endcapture %}
{% include codeoutput.html content=ex %}  





Notice the fit statistics are basically unchanged! 

We removed an entire feature and did not change the fit of the model. This means Feature 3 wasn't adding anything meaningful to the original model. 


Let's remove the Feature 4 and see what happends:

{% capture ex %}
```python
# --- Separate features and target ---
X_full = df[["Feature_1", "Feature_2"]]
X_full = sm.add_constant(X_full) # <- Adds the unknown constant (4 features + constant)
y = df["Target"]

model_full = sm.OLS(y, X_full).fit()
print(model_full.summary())
```
{% endcapture %}
{% include codeinput.html content=ex %} 

{% capture ex %}
```python
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 Target   R-squared:                       0.447
    Model:                            OLS   Adj. R-squared:                  0.435
    Method:                 Least Squares   F-statistic:                     39.13
    Date:                                   Prob (F-statistic):           3.47e-13
    Time:                                   Log-Likelihood:                -353.01
    No. Observations:                 100   AIC:                             712.0
    Df Residuals:                      97   BIC:                             719.8
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -3.1816      4.066     -0.782      0.436     -11.252       4.888
    Feature_1      2.6741      0.468      5.710      0.000       1.745       3.604
    Feature_2      2.2219      0.297      7.473      0.000       1.632       2.812
    ==============================================================================
    Omnibus:                        3.108   Durbin-Watson:                   1.798
    Prob(Omnibus):                  0.211   Jarque-Bera (JB):                2.892
    Skew:                           0.416   Prob(JB):                        0.235
    Kurtosis:                       2.954   Cond. No.                         56.0
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
{% endcapture %}
{% include codeoutput.html content=ex %}  





This time we had a larger impact on the overall fit! 

Deciding what to keep and remove is a bit of an art and depends on the kinds of things you are trying to model. Personally, I would leave Feature 4 out since it was cose to being insignificant and notice removing it make the constant very insignificant. 

Let's remove the constant as see what we get:

{% capture ex %}
```python
# --- Separate features and target ---
X_full = df[["Feature_1", "Feature_2"]]
y = df["Target"]

model_full = sm.OLS(y, X_full).fit()
print(model_full.summary())
```
{% endcapture %}
{% include codeinput.html content=ex %}  

{% capture ex %}
```python
                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                 Target   R-squared (uncentered):                   0.940
    Model:                            OLS   Adj. R-squared (uncentered):              0.939
    Method:                 Least Squares   F-statistic:                              770.2
    Date:                                   Prob (F-statistic):                    1.16e-60
    Time:                                   Log-Likelihood:                         -353.33
    No. Observations:                 100   AIC:                                      710.7
    Df Residuals:                      98   BIC:                                      715.9
    Df Model:                           2                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Feature_1      2.4350      0.354      6.874      0.000       1.732       3.138
    Feature_2      2.0331      0.173     11.723      0.000       1.689       2.377
    ==============================================================================
    Omnibus:                        2.652   Durbin-Watson:                   1.776
    Prob(Omnibus):                  0.265   Jarque-Bera (JB):                2.459
    Skew:                           0.383   Prob(JB):                        0.292
    Kurtosis:                       2.940   Cond. No.                         5.30
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
{% endcapture %}
{% include codeoutput.html content=ex %}  

Notice we are left with a fantastic model now! 

This really is an art! You should explor your models and keeps notes on how each one did and select the model that makes the most sense for your situation.







#### Forward and Backward Selection

The process of starting with **all available features** and then **removing them one by one** until you reach an acceptable model with the smallest number of features possible is called **backward selection** (or **backward elimination**).

- You begin with the *full model* (all predictors included).  
- You iteratively remove the **least significant** feature (often the one with the highest p-value or the smallest contribution to model performance).  
- You stop when removing another feature would noticeably worsen the model’s fit.  

This approach ensures you end up with a **parsimonious model** (a model that explains the data well using the fewest predictors necessary).

Conversely, **forward selection** starts from the opposite direction:

- Begin with **no features** in the model.  
- Add features one at a time, each time choosing the one that most improves model performance (or most significantly reduces error).  
- Stop when adding new features no longer yields meaningful improvement.

In practice, these methods are sometimes combined into a **stepwise selection** process. This is the process of adding and removing features iteratively to find a balance between simplicity and predictive power.


<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaway:</b> 

Feature selection isn’t just about accuracy, it’s about building a model that’s **interpretable, efficient, and generalizes well** to new data.
</div>












#### Colinearity Between Features

**Collinearity** (or **multicollinearity**) occurs when **two or more features in your dataset are highly correlated with each other**. In other words, one feature can be (mostly) predicted from another.

When this happens, the model struggles to determine which feature is actually responsible for the change in the target variable. The result is unstable and unreliable coefficients and thus an unstable model.

**Key symptoms of collinearity**:
- Coefficients have unexpected signs or magnitudes.
- Features that seem important appear statistically insignificant (high p-values).
- The model’s overall fit (e.g., $R^2$) looks good, but none of the individual predictors seem to matter.

**Why it matters**:
- It doesn’t usually harm overall predictive performance much, but it makes interpretation of coefficients meaningless.
- In extreme cases, it can cause the regression solution to become numerically unstable.

**Common ways to detect collinearity**:
- Look at the correlation matrix between features.
- Check the Variance Inflation Factor (VIF) — values above 5 (or 10) often indicate serious collinearity.

**Common ways to address it**:
- Remove or combine redundant features.
- Use dimensionality reduction (e.g., PCA).
- Apply regularization methods (e.g., Ridge or Lasso regression) that penalize redundant predictors.

**In short**:
Collinearity doesn’t mean your model is “wrong,” but it masks the true relationships between variables and makes your model harder to interpret.



{% capture ex %}
```python
# === Visual Model Selection ===

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Adjusted R² vs. Number of Features ---
features = ["Feature_1", "Feature_2", "Feature_3"]
adj_r2_scores = []

for i in range(1, len(features) + 1):
    for combo in itertools.combinations(features, i):
        X_subset = sm.add_constant(df[list(combo)])
        model = sm.OLS(df["Target"], X_subset).fit()
        adj_r2_scores.append((len(combo), model.rsquared_adj))

adj_r2_df = pd.DataFrame(adj_r2_scores, columns=["Num_Features", "Adj_R2"])
avg_scores = adj_r2_df.groupby("Num_Features")["Adj_R2"].mean()

plt.figure(figsize=(7, 5))
plt.plot(avg_scores.index, avg_scores.values, marker="o", linewidth=2)
plt.title("Model Complexity vs. Adjusted R²")
plt.xlabel("Number of Features")
plt.ylabel("Average Adjusted R²")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# --- Correlation Heatmap for Multicollinearity ---
plt.figure(figsize=(6, 5))
sns.heatmap(
    df[features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Predictors")
plt.show()


# --- VIF Table for Multicollinearity ---
# VIF answers a different question than correlation:
# "How redundant is each feature when predicted from the others?"
X_vif = sm.add_constant(df[features])

vif_table = pd.DataFrame({
    "Feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})

vif_table["VIF"] = vif_table["VIF"].round(2)
vif_table


```
{% endcapture %}
{% include codeinput.html content=ex %}  



{% capture ex %}
<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_55_0.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">

<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_55_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">

```python

    Feature	    VIF
0	const	    42.09
1	Feature_1	 1.06
2	Feature_2	 1.02
3	Feature_3	 1.04
```
{% endcapture %}
{% include codeoutput.html content=ex %}  








    


#### How to Choose a Good MLR Model

Building a good multiple linear regression (MLR) model is a balance between simplicity, explanatory power, and predictive performance.
The goal isn’t to fit every detail in the data, it’s to find the simplest model that explains the underlying pattern well.

1. **Start with Domain Knowledge**
- Use theory, intuition, or prior research to decide which features should matter.
- Avoid blindly throwing in every possible variable as this often leads to collinearity and overfitting.
- Think: Does this variable make sense to include? Could it logically influence the outcome?


2. **Check Feature Relationships**
- Use a correlation matrix and VIF values to see how features relate to each other and to the target.
- Remove redundant features (those that are highly correlated with each other).
- Standardize features when they’re on very different scales.

3. **Fit and Evaluate the Full Model**
- Start with all reasonable predictors and fit the model.
- Check:
    - Adjusted $R^2$
    - The p-values of coefficients (for significance)
    - Signs and magnitudes of coefficients (for reasonableness)
- Be cautious of high Adjusted $R^2$ with many insignificant predictors (often signals overfitting or collinearity).


4. **Simplify Using Model Selection Methods**

Use stepwise methods to balance performance and simplicity:
- Backward Selection: Start with all features, remove the least significant one at a time.
- Forward Selection: Start with no features, add the most significant one at a time.
- Stepwise Selection: Combines both approaches adaptively.
- Always verify that model changes are statistically justified (for example, using Adjusted R², AIC, or BIC).

5. **Assess Model Assumptions**

Good regression models follow these assumptions:
- Linearity: The relationship between predictors and target is roughly linear.
- Independence: Observations are independent.
- Normality: Residuals are approximately normally distributed.
- The spread of residuals is roughly constant.


6. **Validate the Model**
- You should have split your data into training and test sets and been working with the training set starting with step 2.
  - Remember, you never work with the test set, EVER! 
  - The only time you see it is when you are testing your model, not when you are exploring!
- If performance drops significantly on the test set, your model is likely overfitting.


7. **Keep It Interpretable**
- Simpler models are easier to explain and generalize better.
- Prefer a model with slightly lower accuracy but clear interpretability over one that’s complex and opaque.
- Remember: The best model isn’t the one with the most features, it’s the one that teaches you the most about the data.

**Summary**

A good MLR model is parsimonious (simple but powerful), statistically sound, and consistent with real-world reasoning.

This process is very much an art as well as a science. The best way to learn it is by doing and gaining experience! 







### Reflection: What Makes a "Good" Regression Model?

After this exercise, think about:
- Why might a model with fewer variables sometimes perform better?
- What do AIC, BIC, and Adjusted R² tell you, and how do they differ?
- What does a high VIF indicate about your predictors?
- How would you decide which variables to keep in a real dataset?

A *good* model isn’t the most complex one — it’s the one that explains the data well, uses interpretable predictors, and generalizes to new data.








## Summary — Linear & Multiple Linear Regression

We’ve explored how linear models form the foundation of machine learning:
from a single feature to multiple predictors working together.

### Linear Regression (Simple)
- Models the relationship between one feature and a continuous target.
- Learns a **line of best fit** by minimizing **Sum of Squared Residuals (SSR)**.
- Evaluated using **MSE**, **MAE**, **RMSE**, and **R²**.
- Residual plots help diagnose linearity, noise, and fit quality.

### Multiple Linear Regression (MLR)
- Extends linear regression to multiple features:
  $$
  y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \text{noise}
  $$
- Each coefficient shows how much the target changes with that feature, *holding other features constant*.
- MLR helps model more realistic systems but introduces new challenges:
  - **Multicollinearity:** predictors carrying overlapping information.
  - **Overfitting:** too many variables relative to data.
  - **Insignificant coefficients:** predictors that add noise, not signal.

### Model Evaluation & Selection

| Concept | What It Tells You | Best Value | Purpose |
|:--------|:------------------|:-----------|:---------|
| **MSE / RMSE / MAE** | How closely predictions match real values | Lower | Fit accuracy |
| **R²** | % of variance explained | Higher (≤ 1) | Model explanatory power |
| **Adjusted R²** | R² penalizing complexity | Higher | Model parsimony |
| **AIC / BIC** | Penalize unnecessary parameters | Lower | Model comparison |
| **p-values** | Which coefficients are statistically significant | < 0.05 | Feature selection |
| **VIF** | Detects redundant predictors | < 5 | Collinearity check |

<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b> 

- Linear regression is a *baseline* model — simple, interpretable, and fast.
- Adding features helps until it starts to hurt (overfitting).
- Simpler models often generalize better.
- Statistical significance and metrics should guide, not dictate, model choices.
- Always visualize and sanity-check your data and residuals.
</div>



# One-Cell Code for the Lecture

{% capture ex %}
```python
# =====================================================
# One-Cell: Linear and Multiple Linear Regression
# Covers: model fitting, evaluation metrics, adjusted R², AIC/BIC, and classification metrics
# =====================================================

# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import statsmodels.api as sm

sns.set(style="whitegrid", font_scale=1.1)
np.random.seed(42)

# ================================================================
# === SIMPLE LINEAR REGRESSION ===
# ================================================================

# Generate synthetic linear data
X_lin = np.linspace(0, 10, 50)
y_lin = 3 * X_lin + 5 + np.random.normal(0, 2, 50)

X_lin_reshaped = X_lin.reshape(-1, 1)
lin_model = LinearRegression().fit(X_lin_reshaped, y_lin)
y_lin_pred = lin_model.predict(X_lin_reshaped)

# Compute fit statistics
n = len(y_lin)
p = 1
mse = mean_squared_error(y_lin, y_lin_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_lin, y_lin_pred)
r2 = r2_score(y_lin, y_lin_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("=== Simple Linear Regression ===")
print(f"Equation: ŷ = {lin_model.intercept_:.2f} + {lin_model.coef_[0]:.2f}x")
print(f"MSE: {mse:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f} | Adjusted R²: {adj_r2:.3f}\n")

# Plot
plt.figure(figsize=(7,5))
plt.scatter(X_lin, y_lin, color='gray', alpha=0.6, label='Actual')
plt.plot(X_lin, y_lin_pred, color='red', linewidth=2, label='Predicted')
plt.title("Simple Linear Regression Fit")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()




# ================================================================
# === MULTIPLE LINEAR REGRESSION ===
# ================================================================

# Create synthetic multiple regression data
n_samples = 100
X1 = np.random.uniform(0, 10, n_samples)
X2 = np.random.uniform(0, 5, n_samples)
X3 = np.random.uniform(-3, 3, n_samples)
noise = np.random.normal(0, 2, n_samples)
y_multi = 2*X1 - 3*X2 + 1.5*X3 + 10 + noise

X_multi = np.column_stack((X1, X2, X3))
multi_model = LinearRegression().fit(X_multi, y_multi)
y_multi_pred = multi_model.predict(X_multi)

# Compute fit stats
n, p = X_multi.shape
mse_m = mean_squared_error(y_multi, y_multi_pred)
rmse_m = np.sqrt(mse_m)
mae_m = mean_absolute_error(y_multi, y_multi_pred)
r2_m = r2_score(y_multi, y_multi_pred)
adj_r2_m = 1 - (1 - r2_m) * (n - 1) / (n - p - 1)

# Compute AIC/BIC via statsmodels
X_multi_sm = sm.add_constant(X_multi)
model_sm = sm.OLS(y_multi, X_multi_sm).fit()

# Here is how you can get each of the measures
print("=== Multiple Linear Regression ===")
print(f"Coefficients: {dict(zip(['X1','X2','X3'], multi_model.coef_))}")
print(f"Intercept: {multi_model.intercept_:.3f}")
print(f"MSE: {mse_m:.3f} | RMSE: {rmse_m:.3f} | MAE: {mae_m:.3f} | R²: {r2_m:.3f} | Adjusted R²: {adj_r2_m:.3f}")
print(f"AIC: {model_sm.aic:.2f} | BIC: {model_sm.bic:.2f}\n")

# Plot actual vs predicted
plt.figure(figsize=(6,5))
plt.scatter(y_multi, y_multi_pred, color='navy', alpha=0.7)
plt.plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 'r--')
plt.title("Multiple Linear Regression: Actual vs. Predicted")
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.grid(alpha=0.3)
plt.show()


# Print out the report of the MLR
print(model_sm.summary())



```
{% endcapture %}
{% include codeinput.html content=ex %}  


    


{% capture ex %}
    === Simple Linear Regression ===
    Equation: ŷ = 5.13 + 2.88x
    MSE: 3.301 | RMSE: 1.817 | MAE: 1.482 | R²: 0.956 | Adjusted R²: 0.955


<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_60_1.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">


    === Multiple Linear Regression ===
    Coefficients: {'X1': 2.0654266856611034, 'X2': -2.9796050500407043, 'X3': 1.4278624368865993}
    Intercept: 9.827
    MSE: 2.609 | RMSE: 1.615 | MAE: 1.311 | R²: 0.963 | Adjusted R²: 0.962
    AIC: 387.67 | BIC: 398.10

<img
  src="{{ '/courses/machine-learning-foundations/images/lec03/output_60_3.png' | relative_url }}"
  alt=""
  style="display:block; margin:1.5rem auto; max-width:1000px; width:60%;">

```python
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.963
    Model:                            OLS   Adj. R-squared:                  0.962
    Method:                 Least Squares   F-statistic:                     842.7
    Date:                                   Prob (F-statistic):           8.46e-69
    Time:                                   Log-Likelihood:                -189.84
    No. Observations:                 100   AIC:                             387.7
    Df Residuals:                      96   BIC:                             398.1
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          9.8268      0.444     22.114      0.000       8.945      10.709
    x1             2.0654      0.056     36.670      0.000       1.954       2.177
    x2            -2.9796      0.113    -26.338      0.000      -3.204      -2.755
    x3             1.4279      0.103     13.928      0.000       1.224       1.631
    ==============================================================================
    Omnibus:                        0.992   Durbin-Watson:                   1.723
    Prob(Omnibus):                  0.609   Jarque-Bera (JB):                0.830
    Skew:                           0.223   Prob(JB):                        0.660
    Kurtosis:                       2.977   Cond. No.                         17.1
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
{% endcapture %}
{% include codeoutput.html content=ex %}  

