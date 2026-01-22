---
layout: default
title: Machine Learning Foundations – Homework 01
course_home: /courses/machine-learning-foundations/
nav_section: homework
nav_order: 1
---


{% capture ex %}
```python

```
{% endcapture %}
{% include codeinput.html content=ex %}

{% capture ex %}

{% endcapture %}
{% include codeoutput.html content=ex %}



# Project 1: Re-engineering a Machine Learning Workflow


---
## Learning Objectives

By the end, students will be able to:
- Identify which parts of code are dataset-specific (e.g., feature names, encoding).
- Modify preprocessing to fit a new dataset structure.
- Verify that models train successfully.
- Interpret the accuracy metric.
- Reflect on how data characteristics affect model performance.

---

As the saying goes: **“Good programmers write code; great programmers reuse and adapt it.”**

Real data scientists rarely write machine learning pipelines from scratch. Instead, they start from a working example and adapt it. This requires them to understand what to change and why in the code they are adapting. They need not be able to write efficient code all on their own.

We are not training to be ML engineers that build and implement algorithms from first principles. We’re training to become practitioners who can:
- understand what common ML models are doing,
- interpret results clearly, and
- apply those models to new data to inform decisions.



## Instructions


1. **Choose a dataset**  
   - Use one of the following data sets:
{% capture ex %}
#### Iris (the flowers)
 <div style="margin-left: 30px">
     
**Target:** species  
 ```python
 from sklearn.datasets import load_iris
 data_original = load_iris(as_frame=True)
 ```
 </div> 


 #### Wine (multiclass, 3 classes) — slightly harder
 <div style="margin-left: 30px">
     
 **Target**: wine cultivar
 ```python
 from sklearn.datasets import load_wine  
 data_original = load_wine(as_frame=True)
 ```
 </div>

 #### Breast Cancer Wisconsin (binary) — interpretable, imbalanced-ish
 <div style="margin-left: 30px">
     
 **Target**: diagnosis (0 = malignant, 1 = benign)
 ```python
 from sklearn.datasets import load_breast_cancer  
 data_original = load_breast_cancer(as_frame=True)
 ```
 </div> 

 #### Digits (multiclass, 10 classes)
 <div style="margin-left: 30px">
     
 **Target**: digit label (0–9)
 ```python
 from sklearn.datasets import load_digits  
 data_original = load_digits(as_frame=True)
 ```
 </div>

#### Titanic (binary) - Takes a bit of effort to get working
<div style="margin-left: 30px">

**Target**: survived (0 = No, 1 = Yes)
```python
import seaborn as sns  
data_original = sns.load_dataset("titanic").dropna(subset=['survived'])
```
 </div> 
{% endcapture %}
{% include codeinput.html content=ex %}
        
 

2. **Replace the data in the notebook**  
   - Update the data loading cell in the code below to import your new dataset.  
   - Adjust variable names and column references (`X`, `y`, feature names) as needed.

3. **Run through the full pipeline**  
   - Preprocessing
       -   Scaling, encoding, etc.
   - Model training  
   - Evaluation and visualization

4. **Add comments above each major block**  
   - Your comments don’t need to be detailed! Just a quick note explaining what the cell/block of code does.
   - I want to know that you know what the block of code is doing!
		- I am not concerned with what each line of code does!
   - Example:
     ```python
     # Split the data into training and test sets
     X_train, X_test, y_train, y_test = train_test_split(...)
     ```

5. **Summarize your results**  
   - Write a short report (shouldn't need more than 600 words for this) on the following:
        - What did you have to change in the code compared to the Penguins example?
        - Which model performed best? Any surprises?
        - One next step you’d try with more time.
             - Google around and see if you can find ML models we did not use here!
		- Which of the models are you most excited/interested in learning more about?



### What to Submit

1. A clean, runnable notebook (Jupyter or Google Colab) that:
    - [ ] loads your chosen dataset,
    - [ ] prepares features/target (encodes if needed),
    - [ ] performs train/test split (use stratify=y),
    - [ ] scales features,
    - [ ] trains the same models done in class (k-NN, Logistic Regression, Naïve Bayes, Decision Tree, Random Forest),
    - [ ] prints accuracy + classification report (with correct class names),
    - [ ] includes the model comparison bar plot, and
    - (optional) shows decision boundaries with the provided helper on a PCA projection.

2. A short (≤600 words) report, as a PDF.
    - [ ] What did you have to change in the code compared to the Penguins example?
    - [ ] Which model performed best? Any surprises?
    - [ ] What part of the pipeline did you find most intuitive?
    - [ ] What part felt the hardest to adapt, and why?
    - [ ] One next step you’d try with more time.
	- [ ] Which of the models are you most excited/interested in learning more about?


### Tips & Common Pitfalls

- Use separate encoders per column (e.g., sex_encoder, embarked_encoder) to avoid overwriting .classes_.
- For classification reports, either:
  - pass labels=np.unique(y_train) and the correct target_names, or
  - omit target_names and print numeric labels.
- Use train_test_split(..., stratify=y) to ensure all classes appear in train/test.
- If x-axis labels overlap on plots: widen the figure and/or rotate tick labels.
- Keep output professional: remove exploratory prints/plots not requested in the deliverables.


## The Code


### Step 0: Import needed Libraries

Run the following block of code.


```python
# ===============================================================
# === 0. Import Libraries ===
# ===============================================================

# --- Core Data Science Libraries ---
import pandas as pd                     # Data manipulation & table handling (like Excel in Python)
import numpy as np                      # Numerical operations & efficient table/array computations

# --- Visualization Libraries ---
import matplotlib.pyplot as plt          # Basic plotting (line plots, histograms, scatter plots)
import seaborn as sns                    # Statistical data visualization, built on top of matplotlib

# --- Scikit-learn: Core Machine Learning Toolkit ---
from sklearn.model_selection import train_test_split                # Splits data into training and testing sets
from sklearn.preprocessing import StandardScaler, LabelEncoder      # Normalize numeric data & encode categories
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Model evaluation metrics

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

```

---

### Step 1a: Load your chosen the data set

In the demo we loaded Palmer Penguins with Seaborn. For this project, replace that loader with the code for the dataset you selected below.

After loading, we’ll keep using a common variable name ("df" for dataframe) for the features table so the rest of the notebook works with minimal changes.

> What you need to do:
> 1. Copy the loader block for your dataset (Iris, Wine, Breast Cancer, Digits, or Titanic) from above.
> 2. Confirm the dataframe variable is named `df`.
> 3. Verify you have loaded the correct data.

<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #8EC9DC;
    padding: 12px;
    border-radius: 6px;
">
<b style="color:#1b4965;">Professional Practice:</b>  
<br><br>
You should alway try to write code/notebooks that work with other data with as minimal required changes as possible. This can be done by using general variables names like df, X, and y.  
<br><br>
This takes practice, but it is worth it!
</div>

#### Example
If you selected the dataset (this is made up) **crabs**, you would use a code that looks something like this
>##### Copy the data load code given above!
>```python
> from sklearn.datasets import penguins
> data_original = sns.load_dataset("penguins")
> ```
>##### Keep a common dataframe variable name for the rest of the notebook
>```python
> df = data_original.frame 
> ```


```python
# Add a comment explaining what the next section of code does
from sklearn.datasets import penguins
data_original = sns.load_dataset("penguins")

# Add a comment explaining what the next section of code does
df = data_original.frame  # includes both features and the 'target' column

```


---

### Step 1b: Clean and Prepare the Data

Now that your dataset is loaded into the variable `df`, we need to **inspect** it. Your goals:

1) Identify the **target** column (what you want to predict).  
2) Identify the **feature** columns (the inputs used to predict).  
3) Decide whether any columns (features or target) need to be **encoded** into numeric form.

**Important**: Do not change the data in this step. We are just inspecting the data.

#### What to check

Run these quick inspections and answer the questions below.



```python
# Peek at the first rows (what columns exist? what do values look like?)
df.head()
```


```python
# Data types and missing values
df.info()
```


```python
# How many unique values in each column? (helps spot categorical columns)
df.nunique().sort_values()
```


```python
# Optional: look at the unique values of any column you suspect is categorical
# Example: df['sex'].unique()
```

#### Questions to consider

**Target**:
- Which column is the target you’re trying to predict? (e.g., target, species, survived)
- Is the target numeric already (0/1/2, etc.) or text (e.g., "setosa")?

**Features**:
- Which columns will you use as features? List them.
- Are there any categorical (text) features (e.g., sex, embarked)?

**Missing values**:
- Do you see any NaNs?
    - If yes, how will you handle them for this project (drop rows vs. simple fill)?
- Is any Encoding needed?
    - Recall, if a feature or the target is text (strings like "Male", "Adelie"), it must be encoded to numbers before modeling. If everything is already numeric, you do not need any encoding.

#### Decide and plan your updates

Based on your inspection, write a short plan:
<div style="margin-left: 30px">
    
Target column: _____ (Does this need encoded?)

Features: [_____ , _____ , _____]

Categorical features to encode (if any): [_____ , _____]

Missing values plan: (e.g., “drop rows with NaN in features we use”)
</div>

You will apply this plan in the next step when you define X and y, perform any encoding that’s needed, and then proceed with train/test split and scaling.

For now, let's just drop any rows with incomplete data (this answers one question above :D):


```python
# Add a comment explaining what the next section of code does
df = df.dropna()

```

---

### Step 1c: Encode Your Data (If Needed)

Before we can train our models, **all features and the target must be numeric**. Most machine learning algorithms can’t handle text directly (e.g., `"Male"`, `"Setosa"`, `"Yes"`).  

Now that you’ve inspected your data in Step 1b, it’s time to update the code below if encoding is required.

### What you need to decide

1. **Does the target need encoding?**  
   - If your target column is *text* (like `"Adelie"` or `"setosa"`), you’ll need to convert it to numbers.  
   - If it’s already numeric (like `0`, `1`, `2`), no changes are needed.

2. **Do any features need encoding?**  
   - Look back at your inspection results.  
   - If any features are text (for example, `"sex"`, `"embarked"`, `"island"`), they must be converted to numeric form.

3. **How will you encode?**  
   - Use the **encoding section from our demo code** as a guide.  
   - You can use `LabelEncoder` for simple categorical features, or `pd.get_dummies()` if a column has several categories.  
   - Use **separate encoders** for each categorical column so their `.classes_` attributes stay distinct.
   - Do not be surprised if you do not need to encode anything for this project.


#### The Code:

Below is what you’ve see  from the in class demo (Lecture 01). Read it carefully and decide what changes (if any) are required.

**Remember**: The data is stored in a variable called `df`.


```python
# This is a slick function that will encode all of the non-numerical columns for you
df_encoded = pd.get_dummies(df, drop_first=True)
```

And we can inspect the dataframe to make sure everything was updated correctly.


```python
# Display the first few rows again to confirm our cleaned and encoded data.
df_encoded.head()
# Notice the additional encoded columns at the end.
```


---

### Step 1d: Select Your Features and Target

Now that your data is clean and numeric, it’s time to tell the model **what it should learn**.

In every machine learning task, we separate our dataset into:

- **Features (`X`)** → the input variables the model will use to make predictions  
- **Target (`y`)** → the output variable the model is trying to predict  

### Your task

Use your dataset inspection from Steps 1b and 1c to decide:

1. Which column is your **target** (the thing you want to predict)?  
   - Example: `species`, `target`, `diagnosis`, or `survived`
2. Which columns are your **features** (the inputs)?  
   - Example: `bill_length_mm`, `flipper_length_mm`, `body_mass_g` (for Penguins)
3. Exclude your target from the feature list!  
   - The model shouldn’t “cheat” by seeing the answer during training.


#### The code:


```python
# Add a comment explaining what the next section of code does

X = df_encoded.drop(columns = ['species_enc']) # <- drops the "target" column.
### Notice we are using the encoded values for the non-numerical data.

# Add a comment explaining what the next section of code does
y = df_encoded['species_enc']
```


---


### Step 1e: Split Data into Training and Test Sets

Before we can train our models, we need to **divide our data** into two groups:

- **Training set:** used by the model to *learn* patterns  
- **Test set:** used to *evaluate* how well the model generalizes to new, unseen data

This separation helps prevent and detect **overfitting**; when a model memorizes the training data instead of learning useful relationships.

#### The Code:



```python
# Add a comment explaining what the next section of code does
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Add a comment explaining what the next section of code does
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

```

---

### Step 2: Quick Visual Check

Before modeling, make a few simple plots to **see** structure in your data.

### What to look for
- **Separation:** Do classes form distinct groups in 2D scatter plots?
- **Trends/shape:** Linear vs. curved relationships; tight vs. diffuse clouds.
- **Outliers:** Obvious points far from the rest.
- **Class balance:** Any class much smaller than the others?
- **Feature usefulness:** Which axes seem to separate classes best?

### Guiding questions
1. Which 1–2 features look most predictive, and why?
2. Any outliers you’d flag?
3. Based on the plots, which model family might work well (linear boundary, tree-based, distance-based)?

#### The Code:



```python
# Add a comment explaining what the next section of code does
sns.pairplot(df_encoded,
             hue=y.name,
             vars=X.columns.tolist())
plt.suptitle("Penguin Feature Relationships", y=1.02) # <- Update this!
plt.show()

```

---




### Step 3: Train and Evaluate Multiple Models

We’ll test several popular algorithms used throughout this course:
- k-Nearest Neighbors (k-NN)
- Logistic Regression
- Naïve Bayes
- Decision Tree
- Random Forest

Each model will:
1. Be trained on the same dataset  
2. Make predictions on the test set  
3. Report its accuracy and classification metrics


#### The Code:



```python
# Add a comment explaining what the next section of code does
models = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),               # k-Nearest Neighbors: compares penguins to their closest neighbors
    "Logistic Regression": LogisticRegression(max_iter=1000),  # Logistic Regression: predicts categories based on probabilities
    "Naïve Bayes": GaussianNB(),                               # Naïve Bayes: uses probability and feature independence assumptions
    "Decision Tree": DecisionTreeClassifier(random_state=42),  # Decision Tree: splits data into “if–then” rules
    "Random Forest": RandomForestClassifier(random_state=42)   # Random Forest: combines many trees for stronger predictions
}

```

#### and we can run and test the accuracy of our models using this:


```python
# Create an empty list where we’ll store each model’s name and accuracy score.
results = []

# Loop through each model in our dictionary.
# 'name' is the string (like "k-NN"), and 'model' is the actual model object.
for name, model in models.items():
    # Train the model using the training data.
    model.fit(X_train_scaled, y_train)
    
    # Use the trained model to predict species for the test data.
    y_pred = model.predict(X_test_scaled)
    
    # Calculate how accurate the model’s predictions were.
    acc = accuracy_score(y_test, y_pred)
    
    # Save the model name and accuracy in our results list for later comparison.
    results.append((name, acc))
    
    # Print the model name and accuracy score, rounded to 3 decimal places.
    print(f"\n{name} Accuracy: {acc:.3f}")
    print("\n")
    
    # Print a detailed report showing how well the model performed for each species.
    # This includes precision, recall, F1-score, and support.
    # Print a detailed report showing how well the model performed for each species.
    print(classification_report(
            y_test,
            y_pred,
            labels=np.unique(y_train)))
    print("\n")
    print("\n")
```

#### Please run the next section of code to define the helper function for plotting.


```python
# ===============================================================
# Helper Function: Visualize 2D Decision Boundaries
# ===============================================================
from matplotlib.colors import ListedColormap

def plot_model_boundaries(model, model_name, X_scaled, y, cmap="viridis"):
    """
    Visualize the decision boundaries created by a classification model
    using PCA for dimensionality reduction.

    Parameters
    ----------
    model : sklearn estimator
        A trained classification model (e.g., LogisticRegression()).
    model_name : str
        A short name for the model (used for the plot title).
    X_scaled : array-like
        Scaled feature matrix (e.g., X_train_scaled).
    y : array-like
        True labels corresponding to X_scaled.
    cmap : str
        Colormap used for background decision regions.
    """

    # --- Step 1. Reduce features to 2D with PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # --- Step 2. Create a 2D meshgrid covering the projected feature space
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # --- Step 3. Train model on the 2D PCA projection
    model.fit(X_pca, y)

    # --- Step 4. Predict each point on the grid to map out decision regions
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # --- Step 5. Plot the decision boundaries and training data
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, cmap=ListedColormap(['#a1dab4','#41b6c4','#225ea8']), alpha=0.6)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, edgecolor='k', s=40)
    plt.title(f"{model_name} Decision Boundaries")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
```

#### Run this set of code to see the decision boundaries. 


```python
plot_model_boundaries(KNeighborsClassifier(n_neighbors=5), "k-Nearest Neighbors", X_train_scaled, y_train)
plot_model_boundaries(LogisticRegression(max_iter=1000), "Logistic Regression", X_train_scaled, y_train)
plot_model_boundaries(GaussianNB(), "Naïve Bayes", X_train_scaled, y_train)
plot_model_boundaries(DecisionTreeClassifier(max_depth=4, random_state=42), "Decision Tree", X_train_scaled, y_train)
plot_model_boundaries(RandomForestClassifier(random_state=42), "Random Forest", X_train_scaled, y_train)

```

---


### Step 4: Compare Model Performance

Let’s visualize the accuracy of each model side by side.

```python
# Turn our list of results into a small DataFrame for easy plotting
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

# Make the figure slightly larger for readability
plt.figure(figsize=(8, 5))

# Create a bar plot comparing accuracy by model.
# 'hue="Model"' ensures each bar has its own color (required in Seaborn ≥0.14)
sns.barplot(x="Model", y="Accuracy", hue="Model", data=results_df,
            palette="crest", legend=False)

# Set axis limits and titles for clarity
plt.ylim(0, 1)
plt.title("Model Comparison on Penguin Classification")
plt.xlabel("Model")
plt.ylabel("Accuracy")

# Rotate x-axis labels so they don't overlap
plt.xticks(rotation=30, ha='right')

plt.tight_layout()  # Adjusts spacing so labels and titles fit nicely
plt.show()
```

---



### Step 4 (continued): Visualize a Decision Tree

Decision trees can be visualized to show how the model splits data step by step.  

Here’s a simplified version of our trained tree.

```python
# Create a new, larger figure so the tree fits clearly on the screen
plt.figure(figsize=(15, 8))

# Use scikit-learn's built-in 'plot_tree' function to visualize our trained decision tree model.
# - models["Decision Tree"] accesses the specific model we trained earlier in our loop.
# - feature_names = X.columns labels the tree’s splits with the actual feature names (e.g., bill length, body mass).
# - class_names = species_encoder.classes_ shows the species names at the leaf nodes.
# - filled=True adds color to the nodes to help distinguish classes visually.
# - fontsize=8 keeps the labels readable without being too large.
plot_tree(models["Decision Tree"],
          feature_names=X.columns,
          filled=True,
          fontsize=8)

# Add a descriptive title to the plot
plt.title("Decision Tree Visualization")

# Display the finished plot
plt.show()

```


---

### Step 5: Try Clustering with k-Means and PCA

Now let’s remove the labels and see if the computer can **discover** the species on its own using clustering.

We’ll use:
- **k-Means Clustering** to form 3 clusters  
- **PCA** (Principal Component Analysis) to reduce 6 dimensions into 2 for visualization

```python
# Create a KMeans clustering model.
# - n_clusters=3 tells the algorithm to group data into 3 clusters (we expect 3 penguin species)
# - random_state=42 ensures reproducibility (same random initialization each run)
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model on the scaled training data and get the predicted cluster for each sample.
# Since k-Means is unsupervised, there are no "labels"—it simply finds patterns in the data.
clusters = kmeans.fit_predict(X_train_scaled)

# Use Principal Component Analysis (PCA) to reduce our multi-dimensional data down to 2 dimensions.
# This allows us to visualize the clustering results on a simple x–y plot.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Create a scatter plot of the data in its 2D PCA projection.
# Each point represents one penguin, and its color shows which cluster it was assigned to.
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=50)

# Add a title and axis labels to make the plot informative.
plt.title("k-Means Clustering with PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Display the final visualization.
plt.show()


```


---

## One-Cell “All-at-Once” Pipeline

**Goal:** Create a single code cell that runs your entire ML workflow end-to-end and produces the requested outputs in a clean, readable order. Think of it as a mini “report generator.”

<div align="center">

**This should be as simple as copying and pasting the code**
**from the cells you modified above into a single cell below.**
</div>



### What your one cell must do (in order)
1. **Imports**  
   - Import the libraries you actually use.  

2. **Data Load**  
   - Load the dataset you chose (Iris, Wine, Breast Cancer, Digits, or Titanic).  

3. **Encode (if needed)**  
   - If your target or any features are text, encode them.  
   - Keep encoders separate (e.g., `sex_encoder`) if you need class names later.

4. **Define Features/Target**  
   - `X = ...`, `y = ...` (exclude the target from `X`!).

5. **Split & Scale**  
   - `train_test_split(..., stratify=y)`  
   - Standardize numeric features (fit on train, transform train/test).

6. **Train Models & Evaluate**  
   - Train the model set from class: k-NN, Logistic Regression, Naïve Bayes, Decision Tree, Random Forest.  
   - **Print** accuracy and a classification report (with correct class names).  

7. **Visuals**  
   - Bar plot comparing model accuracy.  
   - (optional) Decision boundary plot(s) via the helper on PCA-projected data.  
   - A small tree visualization.

### Professionalism reminder
Per the syllabus: only the **requested deliverables** should appear when this cell runs. Extra/leftover output (debug prints, intermediate tables) will incur the automatic deduction.

This means the following cell should **not have** code like `df.head()`. 



```python
# --- Imports & config ---

# Your code
#   here

# --- Load data (df) ---

# Your code
#   here

# --- Encode if needed (brief, only columns that require it) ---

# Your code
#   here

# --- Define X, y ---

# Your code
#   here

# --- Split & scale ---

# Your code
#   here

# --- Train models & collect metrics ---

# Your code
#   here

# --- Plots ---

# Your code
#   here
```
