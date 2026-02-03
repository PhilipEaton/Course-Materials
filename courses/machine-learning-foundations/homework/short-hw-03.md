---
layout: jupyternotebook
title: Machine Learning Foundations – Project 03
course_home: /courses/machine-learning-foundations/
nav_section: homework
nav_order: 2
---

# Project 3: Linear and Multiple Linear Regression


## *Teaching computers how to draw a line.*

In this project you’ll apply Linear Regression and Multiple Linear Regression to model relationships between numerical variables.

You’ll learn how to evaluate regression models, interpret coefficients, and examine residuals to test how well a linear model fits your data.

---

## Learning Objectives

By the end of this project, you will be able to:
- **Identify and frame a regression problem** — recognize when a question involves predicting a *continuous* outcome rather than a category.
- **Apply simple and multiple linear regression models** to explore relationships between variables and make numerical predictions.
- **Interpret model coefficients** to understand the direction and strength of associations between predictors and outcomes.
- **Evaluate regression model performance** using appropriate metrics such as  $R^2$, Mean Squared Error (MSE), and Mean Absolute Error (MAE).
- **Visualize model behavior** through *Residual* plots to assess fit and detect potential issues (e.g., bias, heteroscedasticity, outliers).
- **Compare models** — understand how adding predictors can affect accuracy and generalization.
- **Reflect on model assumptions and bias**, explaining where linear regression may or may not be an appropriate tool for real-world data.

---

## Instructions

1. **Choose a dataset**: Pick one dataset from the list below — all are built into `scikit-learn` or `seaborn`, so you don’t need to download anything.

   Use one of the following datasets:





   - **Diabetes Progression (medical, smaller dataset)**  
     **Target:** disease progression (quantitative)  
     **Predictors**: age, BMI, blood pressure, and lab measurements.     
     ```python
     from sklearn.datasets import load_diabetes
     data_original = load_diabetes(as_frame=True)
     df = data_original.frame
     ```

   - **Auto MPG (fuel efficiency)**  
     **Target:** mpg (miles per gallon)  
     **Predictors**: cylinders, displacement, horsepower, weight, acceleration, model year, origin.   
     ```python
     import seaborn as sns  
     df = sns.load_dataset("mpg").dropna(subset=["mpg"])
     ```

   - **Tips (restaurant bills and tipping behavior)**  
     **Target:** tip (amount left)  
     **Predictors:** total_bill, size, sex, smoker, day, time.  
     ```python
     import seaborn as sns  
     df = sns.load_dataset("tips").dropna(subset=["tip"])
     ```

   - **Diamonds (price prediction — large dataset)**  
     **Target:** price  
     **Predictors:** carat, cut, color, clarity, depth, table, x, y, z.  
     ```python
     import seaborn as sns  
     df = sns.load_dataset("diamonds").dropna(subset=["price"])
     ```


2. **Update and run the code**: Use the Linear & Multiple Linear Regression example below as your template.

   **Your goal**: 
   Modify it so that it correctly loads, processes, and analyzes your chosen dataset with a **continuous target**.

   Add comments to each block (indicated in the code below) to explain what is happening.

   **You’ll need to**:
   - **Load your dataset** and clearly set **X (features)** and **y (continuous target)**.
   - **Update** the multiple linear regression model to drop insignificant features (one at a time!).

3. **Analyze and Report**: Write a short report (including key plots where appropriate!) interpreting the results of your analysis and what they reveal about the dataset you chose to study
    Your report should read in a **semi-professional** tone, similar to a technical summary you might provide to a customer who asked you to build this model. The goal is to clearly explain:
    - Which features appear most predictive?
    - How well does the model fit (look at residuals and R²)?
    - Were any assumptions of linear regression violated (linearity, constant variance, outliers)?
    - What would you try next (interaction terms, polynomial features, etc.)?  

    Your notebook should produce the following:
    - **Model details & comparisons:**
    - [ ] Simple Linear Regression (1 predictor): metrics + a brief note on the single coefficient
    - [ ] Multiple Linear Regression (all predictors): metrics + a table of coefficients

    - **Diagnostics & interpretability:**
    - [ ] Residual plot with y=0 reference line
    - [ ] Statsmodels OLS summary for coefficient significance (p-values) and adjusted $R^2$

    - **Core metrics (test set):**
    - [ ] R² (coefficient of determination)
    - [ ] MSE (mean squared error)
    - [ ] MAE (mean absolute error)





### What to Submit

- A semi-professional report deailing your findings, written for a layperson.
    - Some details of what you could include are given above and must include are given below. 
- An indepth reflection about the project.
    - Suggestions on what you can reflect on are below.


1. **A clean, runnable Jupyter notebook** that:
    - [ ] loads and briefly describes your dataset (continuous target variable),
    - [ ] prepares and splits the data (train/test),
    - [ ] fits at least one *Simple* and one *Multiple* Linear Regression model,
    - [ ] includes R², MSE, and MAE for each model,
    - [ ] shows your *Residuals* plot,
    - [ ] uses clear block comments explaining each code step,
    - [ ] runs top-to-bottom without errors.

2. **A short written report (2–4 pages, PDF format)** that explains your project.
    See What you should include in Setp 3 above. 

    For further consideration, you should think about, and possibly add to your report or your reflection section (wink wink), comments about:
    - [ ] What question or relationship you investigated,  
    - [ ] How you prepared and analyzed the data,  
    - [ ] Which model performed best and why (Remember to run a simple linear regression for all of the features!),  
    - [ ] Note and patterns or outliers in residuals,  
    - [ ] Any violations of model assumptions (look at residuals),  

> **Reminder:** Your notebook should demonstrate a clear, step-by-step workflow, not just working code.  
> Use comments, titles, and plots to tell the *story* of your analysis.


## The Code

The cell below contains a complete example demonstrating both linear and multiple linear regression in action.

Use it as a template to guide your own work.


{% capture ex %}
```python
# --- Imports ---

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


# --- Visualization Styling ---
sns.set(style="whitegrid", palette="muted", font_scale=1.1)  # Nice default theme for plots


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
# Add a comment explaining what the next section of code does
data_original = fetch_california_housing(as_frame=True)
df = data_original.frame.dropna()
target_name = 'MedHouseVal'

# Add a comment explaining what the next section of code does
df = df.dropna()

# Add a comment explaining what the next section of code does
df_encoded = pd.get_dummies(df, drop_first=True)

# Add a comment explaining what the next section of code does
X = df_encoded.drop(columns = [target_name])
y = df_encoded[target_name]
# --------------------------------------------------------------------




# --------------------------------------------------------------------
# --- This is just a fun plot for this data.
#        You can see the outline of Cali. :D
# --------------------------------------------------------------------
plt.scatter(df["Longitude"], df["Latitude"])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Location")
plt.show()

# Here is a cool, different way to make a plot :D
df.plot(kind="scatter", x="Longitude", y="Latitude", grid=True, alpha=0.2)
plt.show()
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
sns.pairplot(df_encoded,
             hue=y.name,
             vars=X.columns.tolist())
plt.suptitle("Feature Relationships", y=1.02) # <- Update this!
plt.show()


# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Add a comment explaining what this section of code is doing. ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------

### Add your comments here ###
feature = X.columns[0]
X_train_simple = X_train_scaled[:, [0]]
X_test_simple = X_test_scaled[:, [0]]

### Add your comments here ###
simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)
y_pred_simple = simple_model.predict(X_test_simple)

### Add your comments here ###
print("\n=== Simple Linear Regression ===")
print(f"Feature: {feature}")
print(f"R²: {r2_score(y_test, y_pred_simple):.3f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_simple):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_simple):.3f}")
print(f"Coefficient: {simple_model.coef_[0]:.3f}")
print(f"Intercept: {simple_model.intercept_:.3f}")

# Plot: Actual vs Predicted
plt.scatter(X_test[feature], y_test, label="Actual")
plt.plot(X_test[feature], y_pred_simple, color="red", label="Predicted")
plt.xlabel(feature)
plt.ylabel("Median House Value")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# Plot: residuals
residuals = y_test - y_pred_simple

plt.figure(figsize=(6, 4))
plt.scatter([range(len(residuals))], residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Index")
plt.ylabel("Residuals")
plt.title("Residual Plot – Simple Linear Regression")
plt.show()
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# --- Add a comment explaining what this AND THE NEXT cell  ---
#        combined have you doing.                           ---
# --------------------------------------------------------------------

# --------------------------------------------------------------
# Convert scaled data into a DataFrame with proper column names
# --------------------------------------------------------------
X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

# Ensure y_train has matching index
y_train_aligned = y_train.copy()
y_train_aligned.index = X_train_df.index

# Add constant term for intercept
X_train_sm = sm.add_constant(X_train_df)

# --------------------------------------------------------------
# Fit the initial Multiple Linear Regression model
# --------------------------------------------------------------
initial_mlr_model = sm.OLS(y_train_aligned, X_train_sm).fit()
print(initial_mlr_model.summary())

# --------------------------------------------------------------
# Identify insignificant predictors (p > 0.05)
# --------------------------------------------------------------
p_values = initial_mlr_model.pvalues
insignificant = p_values[p_values > 0.05].index.tolist()
insignificant = [var for var in insignificant if var != "const"]  # keep constant

print("\n Insignificant features (p > 0.05):")
print(insignificant)



# --------------------------------------------------------------
# Remove MOST insignificant predictor and refit model
## Repeat until all predictors are significant.
# --------------------------------------------------------------
X_train_reduced = X_train_df.drop(columns="Population")
X_train_reduced = sm.add_constant(X_train_reduced)

reduced_model = sm.OLS(y_train_aligned, X_train_reduced).fit()

print("\n=== Reduced Model Summary ===")
print(reduced_model.summary())

# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------

# For the initial model
y_pred = initial_mlr_model.predict(X_train_sm)

r2 = r2_score(y_train_aligned, y_pred)
mse = mean_squared_error(y_train_aligned, y_pred)
mae = mean_absolute_error(y_train_aligned, y_pred)
rmse = np.sqrt(mse)

print("\n=== Initial Model Performance ===")
print(f"R²:   {r2:.3f}")
print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")


# For the reduced model
y_pred_reduced = reduced_model.predict(X_train_reduced)

r2_red = r2_score(y_train_aligned, y_pred_reduced)
mse_red = mean_squared_error(y_train_aligned, y_pred_reduced)
mae_red = mean_absolute_error(y_train_aligned, y_pred_reduced)
rmse_red = np.sqrt(mse_red)

print("\n=== Reduced Model Performance ===")
print(f"R²:   {r2_red:.3f}")
print(f"MSE:  {mse_red:.3f}")
print(f"RMSE: {rmse_red:.3f}")
print(f"MAE:  {mae_red:.3f}")



# --------------------------------------------------------------------
# --- Add a comment explaining what this section of code is doing. ---
# --------------------------------------------------------------------
residuals = y_train_aligned - y_pred_reduced

plt.figure(figsize=(6, 4))
plt.scatter([range(len(residuals))], residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Index")
plt.ylabel("Residuals")
plt.title("Residual Plot – Multiple Linear Regression")
plt.show()
```
{% endcapture %}
{% include codeinput.html content=ex %}
