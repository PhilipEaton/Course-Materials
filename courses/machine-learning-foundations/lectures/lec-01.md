---
layout: default
title: Machine Learning Foundations – Lecture 01
course_home: /courses/machine-learning-foundations/
nav_section: lectures
nav_order: 1
---

# Lecture 01: Introduction to Machine Learning

## System Set-up

Let's take some time to make sure everyone has their coding space set up. 

**Easiest way: Anaconda (All-in-One)**:

**Why**: Easiest install; ships with Python, Jupyter, and most libraries we’ll use.

1. Download & install Anaconda
    - Go to: [https://www.anaconda.com/download] → choose your OS → Python 3.x
    - During install:
        - Windows: allow the installer to add Anaconda to PATH (okay for this course).
        - macOS (Intel/Apple Silicon): pick the default installer for your chip (M1/M2 = Apple Silicon).
2. Launch Jupyter Notebook
    - Open Anaconda Navigator → click Jupyter Notebook, or
    - From a terminal: `jupyter notebook`
3. Create a course environment (recommended but optional)
    - Open Anaconda Prompt / Terminal and run:
        
        > ```python
        > conda create -n dssa-ml python=3.11 -y
        > conda activate dssa-ml
        > conda install -y numpy pandas matplotlib seaborn scikit-learn jupyterlab
        > ```
    - Then start Jupyter: `jupyter notebook` (or `jupyter lab` if you prefer JupyterLab)


**Optional way - Use and IDLE: VS Code + Python Extension**

**Why**: Nice editor + notebooks in one app.
- Install VS Code → Extensions → Python (Microsoft).
- Open a folder → create a `.ipynb` or `.py` → make sure the kernel is your `dssa-ml` env.

**Quick Verification**

Run the follow cell:


```python
import sys, platform
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

print("✅ Python OK:", sys.version)
print("✅ Platform:", platform.platform())
print("✅ numpy:", np.__version__)
print("✅ pandas:", pd.__version__)
print("✅ matplotlib:", plt.matplotlib.__version__)
print("✅ seaborn:", sns.__version__)
print("✅ scikit-learn:", sklearn.__version__)

# Tiny smoke test plot
import numpy as np
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x))
plt.title("Jupyter is working!")
plt.show()

```

    ✅ Python OK: 3.12.4 | packaged by Anaconda, Inc. | (main, Jun 18 2024, 10:07:17) [Clang 14.0.6 ]
    ✅ Platform: macOS-14.6.1-arm64-arm-64bit
    ✅ numpy: 1.26.4
    ✅ pandas: 2.2.2
    ✅ matplotlib: 3.8.4
    ✅ seaborn: 0.13.2
    ✅ scikit-learn: 1.4.2



<img
  src="{{ '/courses/machine-learning-foudations/images/lec01/output_1_1.png' | relative_url }}"
  alt="Image of a sine wave to help check that python has been installed properly."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">
    


If everything is a green check mark ✅, then you are good to go! Skip down to **"Can We Predict a Penguin's Species?"**.

If something failed, then let's cover some common errors:

- “**Kernel not found / won’t start**” → In Jupyter, select Kernel → Change Kernel → `dssa-ml` (or your env name).
- **ImportError for a library** → Install into the active environment:
    > ```python
    > conda activate dssa-ml
    > conda install seaborn scikit-learn  # or: pip install seaborn scikit-learn
    > ```
- **Command not found** (jupyter/conda) → Close and reopen your terminal; on Windows use Anaconda Prompt.
- **Apple Silicon (M1/M2) oddities** → Prefer conda installs over pip for scientific libibraries.
- **Colab** → If a package is missing, install in a cell:
    > ```python
    > !pip install <package>
    > ```

<h1 style="
    color: white;
    background-color: #4bbe7e;
    padding: 15px;
    border-radius: 10px;
    text-align: left;
">
"Can We Predict a Penguin’s Species?"
</h1>

Welcome to your first Machine Learning (ML) adventure!

In this demo, we’ll explore the **Palmer Penguins** dataset. The `pengiun` and `iris` data sets are friendly, visual, and are often used for in-class examples and demonstrations.

Our goal: **build ML models that can *predict the species of a penguin*** from its physical measurements.

That is, using formal data science language:

- **Features**: the physical measurements
- **Target**: species of a penguin

Today will searve as an example of the tools we will use in class and will give us a feel for the entire ML workflow that we’ll use throughout this course:

0. Import needed Libraries
2. Load and clean a dataset  
3. Visualize relationships  
4. Train several ML models and Evaluate their performance
5. Compare the ML models performance  
6. Try an unsupervised approach with clustering and PCA  
7. Reflect on what we learned

<h2 style="
    color: white;
    background-color: #e28f41;
    padding: 10px;
    border-radius: 8px;
">
Step 0: Import needed Libraries and Functions
</h2>

Run the following cell to load in the required libraries and functions for this notebook.


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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, adjusted_rand_score  # Model evaluation metrics

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

<!-- Subsection Header -->
<h3 style="
    color: white;
    background-color: #f4b942;
    padding: 8px;
    border-radius: 6px;
">
Instructor Note: Why So Many Imports?
</h3>

You might be wondering why we’re importing so many things from `sklearn` individually instead of just writing something like:

> ```python
> import sklearn as sk 
> ```

That’s a great question — and the short answer is that Scikit-learn (sklearn) is a **very large** library made up of many smaller modules.

When we write:

> ```python
> from sklearn.preprocessing import StandardScaler
> ```

we’re only loading the specific tool we need (here, the scaler for normalizing data that we will be using a lot).

If we only wrote:

> ```python
> import sklearn as sk 
> ```

we would still need to call functions like this:

> ```python
> sklearn.preprocessing.StandardScaler()
> ```

which makes coding annoying and the code itself harder to read/understand.

Importing the way we did above makes our code:
- **Clearer**: we see where each tool comes from
- **Easier to learn**: you’ll start to recognize which modules handle what (e.g., sklearn.tree, sklearn.metrics)
- **More efficient**: Python doesn’t load the entire library unnecessarily

So, while it looks like a lot of imports, this way helps us learn and keeps everything organized!


<h2 style="
    color: white;
    background-color: #e28f41;
    padding: 10px;
    border-radius: 8px;
">
Step 1a: Load the Data
</h2>

We’ll use Seaborn’s built-in version of the **Palmer Penguins** dataset.  
Each row describes a penguin, with measurements such as bill length, flipper length, body mass, and more.

Let’s load it and take a quick peek.



```python
# Load the "Palmer Penguins" dataset directly from Seaborn's built-in collection
penguins = sns.load_dataset("penguins")

# Display the first few rows of the dataset (by default, the first 5)
# This helps us preview the data and understand what each column looks like
penguins.head()

```




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
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>



The snippet of code:

> ```python
> penguins.head()
> ```

returns the first 5 rows of the data for us to inspect. 

You should always do this when you first load data. It helps you understand the structure of the dataset, see what the column names are, and get a sense of the types of data you’re working with (e.g., numbers, text, or a mix of both).


<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #8EC9DC;
    padding: 12px;
    border-radius: 6px;
">
<b style="color:#1b4965;">Professional Practice:</b>  
<br><br>
In data science, it’s considered best practice to <b>separate exploratory code</b> from your final, deliverable code.  
Exploratory commands such as <code>.head()</code>, <code>.info()</code>, or quick diagnostic plots are essential while you’re
getting to know your data. However, they should not appear in the version you hand off to a client, supervisor, or production system.
<br><br>
Your final notebook or script should:
<ul>
<li>Generate <b>only the outputs your client or stakeholder requested</b>.</li>
<li>Run <b>cleanly from start to finish</b> with no extra inspection cells or debugging output.</li>
<li>Be <b>reproducible</b>—anyone should be able to rerun it and obtain the same results from <b>another</b> computer.</li>
</ul>
Think of it like publishing a scientific paper: you can take all the notes you want during analysis, but the version you submit should be polished, clear, and professional.
</div>


<h2 style="
    color: white;
    background-color: #e28f41;
    padding: 10px;
    border-radius: 8px;
">
Step 1b: Clean and Prepare the Data
</h2>

Before we can use the dataset in machine learning models, we need to:
- Remove rows with missing values.
- Convert text columns (like `species` and `sex`) into numeric form.

First, let's drop the rows that are missing data:


```python
# Drop any rows that have missing values (NaN = “not a number”).
# This helps prevent errors later when training our models.
penguins = penguins.dropna()

# Display the first few rows of the dataset (by default, the first 5)
penguins.head()
```




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
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.3</td>
      <td>20.6</td>
      <td>190.0</td>
      <td>3650.0</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>



Notice the fourth row that was previously filled with `NaN` is gone! 

Now, we need to create a label encoder, which will convert text in the data (like "Male", "Female", or "Adelie") into numeric values that the machine learning models can understand.


```python
# Start by opening the encoder and storing it with the name label_enc.
# Create and fit separate label encoders for each categorical variable
species_encoder = LabelEncoder()
island_encoder  = LabelEncoder()
sex_encoder     = LabelEncoder()


# Now we can transform (or convert or encode) each of the text columns into numbers.
# Each unique category (e.g., species, island names, and sex) becomes a numeric code.
penguins['species_enc'] = species_encoder.fit_transform(penguins['species'])
penguins['island_enc']  = island_encoder.fit_transform(penguins['island'])
penguins['sex_enc']     = sex_encoder.fit_transform(penguins['sex'])
```


```python
# Display the first few rows again to confirm our cleaned and encoded data.
penguins.head()
# Notice the additional encoded columns at the end.
```




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
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
      <th>species_enc</th>
      <th>island_enc</th>
      <th>sex_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.3</td>
      <td>20.6</td>
      <td>190.0</td>
      <td>3650.0</td>
      <td>Male</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We need to define the **features** and **target** in the data.
- **Features** are the data we will be using to make decisions.
- The  **Target** is the classification, number, etc. we are trying to predict given the features.


```python
# Define our “features” (or "independent variables" or "predictors") as X — the input data used to make predictions.
# These are the penguins’ measurable traits like bill size, flipper length, etc.
X = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
              'body_mass_g', 'island_enc', 'sex_enc']]
# Notice we are using the encoded values for the non-numerical data.

# Define our “target” (y) — the thing we want to predict.
# Here, it’s the penguin’s species.
y = penguins['species_enc']

```

Lastly, split your dataset into a ***training* set** and a ***test* set**.

- **Training set** (≈70–80%) — used to fit/train your models.
- **Test set** (≈20–30%) — held out until the end to evaluate how well the trained model generalizes (accuracy, consistency).

> Tip: For classification, use a stratified split so the class proportions in train and test are similar.
> This can be done in the following manner:
> ```python
> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
> ```

Here we will not stratify, but we could if we wanted to! 


```python
# Split our data into two parts:
#   80% for training (to teach the model)
#   20% for testing (to check how well it learned)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the numeric features so they’re on a similar range (important for distance-based models).
# This prevents larger measurements (like body mass) from dominating smaller ones.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

<!-- Subsection Header -->
<h3 style="
    color: white;
    background-color: #f4b942;
    padding: 8px;
    border-radius: 6px;
">
One Stop Shop
</h3>

Below is the complete code for managing the data, assuming you have inspected the data and are encoding everything you need to encode.


```python
# Load the "Palmer Penguins" dataset directly from Seaborn's built-in collection
data = sns.load_dataset("penguins")

# Drop any rows that have missing values (NaN = “not a number”).
# This helps prevent errors later when training our models.
data = data.dropna()

# Start by opening the encoder and storing it with the name label_enc.
# Create and fit separate label encoders for each categorical variable
species_encoder = LabelEncoder()
island_encoder  = LabelEncoder()
sex_encoder     = LabelEncoder()


# Now we can transform (or convert or encode) each of the text columns into numbers.
# Each unique category (e.g., species, island names, and sex) becomes a numeric code.
data['species_enc'] = species_encoder.fit_transform(data['species'])
data['island_enc']  = island_encoder.fit_transform(data['island'])
data['sex_enc']     = sex_encoder.fit_transform(data['sex'])

# Define our “features” (or "independent variables" or "predictors") as X — the input data used to make predictions.
# These are the penguins’ measurable traits like bill size, flipper length, etc.
X = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
              'body_mass_g', 'island_enc', 'sex_enc']]
# Notice we are using the encoded values for the non-numerical data.

# Define our “target” (y) — the thing we want to predict.
# Here, it’s the penguin’s species.
y = data['species_enc']

# Split our data into two parts:
#   80% for training (to teach the model)
#   20% for testing (to check how well it learned)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the numeric features so they’re on a similar range (important for distance-based models).
# This prevents larger measurements (like body mass) from dominating smaller ones.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

<h2 style="
    color: white;
    background-color: #e28f41;
    padding: 10px;
    border-radius: 8px;
">
Step 2: Explore the Data (after cleaning)
</h2>

Let’s visualize a few relationships between features and species.  

**Why plot first?**  
The human eye is *exceptionally* good at spotting patterns, clusters, trends, outliers, and boundaries, even when they’re subtle. With only a handful of features, simple plots can quickly reveal relationships that can guide our modeling choices.

**What to look for**
- **Separation:** Do data points form distinct clouds in a scatter plot
    - e.g., to species cluster in a plot of bill depth vs. bill length?
- **Shape of relationships:** Linear trend or curved? Tight or diffuse?
- **Outliers and anomalies:** Any points far from the pack that might influence a model.
- **Class imbalance:** Are there far fewer points for one species?
    - Having one group dominate could hurt the generalizability of our models.
- **Feature usefulness:** Which axes seem to separate species most cleanly?

**Caveats (your eyes are great—but not perfect)**
- **Illusions of separation:** With few points, randomness can *look* meaningful. This is why we still use models and do not just make models from by-eye observations.
- **Multiple comparisons:** In a big grid of plots, something could look “good” by chance.
- **Scale effects:** Distance-based models (like k-NN, k-means) are sensitive to the scales of data.
    - We’ll always standardize the features to make comparisons fair and to make models easier to interpret.
- **High dimensions:** Patterns that look clean in 2D can vanish in higher dimensions or *vice versa*.
- **Correlation ≠ causation:** Visual association is a clue, not proof.

**Connect to our modeling steps**
We will learn that:
- clear visual boundaries suggest **logistic regression** or **decision trees** may do well.
- compact, circular/spherical clusters often suit **k-means**; elongated or uneven clusters may not.
- strongly differing scales or variances remind us to **standardize** before k-NN/k-means.
- overlapping blobs might benefit from **nonlinear** models or **feature engineering**.


**As you look at the plots below, ask yourself:**
1. Which two features best separate the species?
2. Do you see outliers that might need attention?
3. If you had to pick just *one* feature to predict species, which would it be—and why?




```python
sns.pairplot(penguins,
             hue="species", # <- Target
             vars=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"] # <- Features
            )
plt.suptitle("Penguin Feature Relationships", y=1.02)
plt.show()

```


    
![png](output_22_0.png)
    


<h2 style="
    color: white;
    background-color: #e28f41;
    padding: 10px;
    border-radius: 8px;
">
Step 3: Train and Evaluate Model(s)
</h2>

Today we will test several popular algorithms used throughout this course:
- k-Nearest Neighbors (k-NN)
- k-Means
- Logistic Regression
- Naïve Bayes
- Decision Tree
- Random Forest

Each model will:
1. Be trained on the same training dataset.
2. Make predictions on the same test set.  
3. Report its accuracy and classification metrics.



```python
# Create a dictionary that links each model’s name (as a string)
# to the actual model object from scikit-learn.
# This lets us loop through several models easily without repeating code.
models = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),               # Instance-based classifier using nearest neighbors
    "Logistic Regression": LogisticRegression(max_iter=1000),  # Predicts category probabilities
    "Naïve Bayes": GaussianNB(),                               # Probabilistic model assuming feature independence
    "Decision Tree": DecisionTreeClassifier(random_state=42),  # Rule-based splitting model
    "Random Forest": RandomForestClassifier(random_state=42)  # Ensemble of trees for stronger predictions
}

```


```python
# --- Initialize results list for summary table ---
results = []

# Loop through each model in our dictionary.
for name, model in models.items():
    
    print(f"\n=== {name} ===")
    
    # Train the model using the training data.
    model.fit(X_train_scaled, y_train)

    # Predict the labels for the test data.
    y_pred = model.predict(X_test_scaled)

    # Calculate fit metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")  # weighted handles class imbalance

    # Save results for later summary
    results.append((name, acc, f1))

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Weighted F1-Score: {f1:.3f}\n")
    print(classification_report(
        y_test,
        y_pred,
        labels=np.unique(y_train),
        target_names=species_encoder.classes_  # Hard-coded for penguin species
    ))

# --- Convert results to DataFrame for comparison ---
results_df = pd.DataFrame(results, columns=["Model", "Accuracy (or ARI)", "F1-Score"])
print("\n=== Model Comparison Summary ===")
display(results_df)

```

    
    === k-NN ===
    Accuracy: 0.985
    Weighted F1-Score: 0.985
    
                  precision    recall  f1-score   support
    
          Adelie       1.00      0.97      0.98        31
       Chinstrap       0.93      1.00      0.96        13
          Gentoo       1.00      1.00      1.00        23
    
        accuracy                           0.99        67
       macro avg       0.98      0.99      0.98        67
    weighted avg       0.99      0.99      0.99        67
    
    
    === Logistic Regression ===
    Accuracy: 1.000
    Weighted F1-Score: 1.000
    
                  precision    recall  f1-score   support
    
          Adelie       1.00      1.00      1.00        31
       Chinstrap       1.00      1.00      1.00        13
          Gentoo       1.00      1.00      1.00        23
    
        accuracy                           1.00        67
       macro avg       1.00      1.00      1.00        67
    weighted avg       1.00      1.00      1.00        67
    
    
    === Naïve Bayes ===
    Accuracy: 0.896
    Weighted F1-Score: 0.900
    
                  precision    recall  f1-score   support
    
          Adelie       1.00      0.77      0.87        31
       Chinstrap       0.65      1.00      0.79        13
          Gentoo       1.00      1.00      1.00        23
    
        accuracy                           0.90        67
       macro avg       0.88      0.92      0.89        67
    weighted avg       0.93      0.90      0.90        67
    
    
    === Decision Tree ===
    Accuracy: 0.985
    Weighted F1-Score: 0.985
    
                  precision    recall  f1-score   support
    
          Adelie       1.00      0.97      0.98        31
       Chinstrap       0.93      1.00      0.96        13
          Gentoo       1.00      1.00      1.00        23
    
        accuracy                           0.99        67
       macro avg       0.98      0.99      0.98        67
    weighted avg       0.99      0.99      0.99        67
    
    
    === Random Forest ===
    Accuracy: 1.000
    Weighted F1-Score: 1.000
    
                  precision    recall  f1-score   support
    
          Adelie       1.00      1.00      1.00        31
       Chinstrap       1.00      1.00      1.00        13
          Gentoo       1.00      1.00      1.00        23
    
        accuracy                           1.00        67
       macro avg       1.00      1.00      1.00        67
    weighted avg       1.00      1.00      1.00        67
    
    
    === Model Comparison Summary ===



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
      <th>Model</th>
      <th>Accuracy (or ARI)</th>
      <th>F1-Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>k-NN</td>
      <td>0.985075</td>
      <td>0.985229</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Naïve Bayes</td>
      <td>0.895522</td>
      <td>0.899955</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Decision Tree</td>
      <td>0.985075</td>
      <td>0.985229</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Random Forest</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


#### Understanding Classification Metrics

When we evaluate a machine learning model, we don’t just care about how many predictions were right overall. We also want to know how the model got things right (and wrong).

The classification report provides four key metrics for each class:

| Metric                   | What It Means                                                                                                                                                               | Why It Matters                                                                                                                            |
| :----------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| **Precision**            | Of all the data *predicted* as a given class, what percent were correct?  <br>*“When the model says YES, how often is it right?”*                        | High precision means fewer false positives. This is important when false alarms are costly (e.g., diagnosing a disease that isn’t there). |
| **Recall (aka Sensitivity)** | Of all the data that truly belong to a given class, how many were correctly identify?  <br>*“When something is truly YES, how often does the model catch it?”* | High recall means fewer false negatives. This matters when missing something is costly (e.g., failing to detect fraud).                   |
| **F1-Score**             | The **harmonic mean** of precision and recall. It balances the two, giving a single score that penalizes extreme imbalances.                                                | Useful when you want a single number to capture both precision and recall performance.                                                    |
| **Support**              | The number of true instances of each class in the dataset.                                                                                                                  | Helps you see whether each class has a lot of examples (balanced) or just a few (imbalanced).                                             |

<div style="
    background-color: #E6F2FA;
    border-left: 6px solid #8EC9DC;
    padding: 14px;
    border-radius: 6px;
">
<b style="color:#1b4965;">Professional Practice:</b>  
<br><br>
That <b>harmonic mean</b> in the F1-score isn’t just a fancy choice—it serves an important purpose.  
Unlike the regular (arithmetic) average (add things up and divide by the number of things), the harmonic mean <b>penalizes extreme differences</b> between precision and recall.  
<br><br>
If precision is <i>perfect</i> but recall is <i>terrible</i>, the F1-score drops dramatically, reflecting that your model isn't actually useful.
<br><br>

<b>Mathematically:</b>  
<div style="background-color: white; border-radius: 4px; padding: 8px; margin: 8px 0;">
$$
F_1 = 2 \times \left(\frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\right)
$$
</div>

<br>
To see why this matters, compare the two averages:
<ul style="margin-top: 6px; margin-bottom: 6px;">
<li><b>Arithmetic Mean:</b> 
    <b>Mathematically:</b>  
<div style="background-color: white; border-radius: 4px; padding: 8px; margin: 8px 0;">
$$
    \frac{(1.0 + 0.2)}{2} = 0.6
$$
</div></li>
<li><b>Harmonic Mean:</b> 
    <b>Mathematically:</b>  
<div style="background-color: white; border-radius: 4px; padding: 8px; margin: 8px 0;">
$$
    2 \times \left(\frac{1.0 × 0.2}{1.0 + 0.2}\right) = 0.33
$$
</div></li>
</ul>

The harmonic mean pulls the score down more strongly when one value is low, forcing the model to perform well on both <b>precision</b> and <b>recall</b> to get a high $F_1$.
<br><br>
In professional data science practice, this makes $F_1$-score the go-to metric for <b>imbalanced datasets</b>—  like detecting rare diseases or fraudulent transactions—where simple accuracy can be misleading.
</div>



The following is a helpful function that can be used to viziualize 2D decision boundaries. 

Run the following cell the load the helper function:


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

Now we can apply this function to the different models we trained to see how they are making their decisions. 



```python
plot_model_boundaries(KNeighborsClassifier(n_neighbors=5), "k-Nearest Neighbors", X_train_scaled, y_train)
plot_model_boundaries(LogisticRegression(max_iter=1000), "Logistic Regression", X_train_scaled, y_train)
plot_model_boundaries(GaussianNB(), "Naïve Bayes", X_train_scaled, y_train)
plot_model_boundaries(DecisionTreeClassifier(max_depth=4, random_state=42), "Decision Tree", X_train_scaled, y_train)
plot_model_boundaries(RandomForestClassifier(random_state=42), "Random Forest", X_train_scaled, y_train)

```


    
![png](output_30_0.png)
    



    
![png](output_30_1.png)
    



    
![png](output_30_2.png)
    



    
![png](output_30_3.png)
    



    
![png](output_30_4.png)
    


<h2 style="
    color: white;
    background-color: #e28f41;
    padding: 10px;
    border-radius: 8px;
">
Step 3 (continued): Visualize a Decision Tree
</h2>

Decision trees can be visualized to show how the model splits data step by step.  

Here’s a simplified version of our trained tree:



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
          class_names=species_encoder.classes_,
          filled=True,
          fontsize=8)

# Add a descriptive title to the plot
plt.title("Decision Tree Visualization")

# Display the finished plot
plt.show()

```


    
![png](output_32_0.png)
    


<h2 style="
    color: white;
    background-color: #e28f41;
    padding: 10px;
    border-radius: 8px;
">
Step 4: Compare Model Performance
</h2>

Let’s visualize the accuracy and $F_1$-score of each model side by side.



```python
# Turn our list of results into a small DataFrame for easy plotting
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1-Score"])

# Make the figure slightly larger for readability
plt.figure(figsize=(8, 5))

## Bar plot of the Accuracy of each model
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


## Bar plot of the F1-score of each model
# Make the figure slightly larger for readability
plt.figure(figsize=(8, 5))

# Create a bar plot comparing accuracy by model.
# 'hue="Model"' ensures each bar has its own color (required in Seaborn ≥0.14)
sns.barplot(x="Model", y="F1-Score", hue="Model", data=results_df,
            palette="crest", legend=False)

# Set axis limits and titles for clarity
plt.ylim(0, 1)
plt.title("Model Comparison on Penguin Classification")
plt.xlabel("Model")
plt.ylabel("F1-Score")

# Rotate x-axis labels so they don't overlap
plt.xticks(rotation=30, ha='right')

plt.tight_layout()  # Adjusts spacing so labels and titles fit nicely
plt.show()
```


    
![png](output_34_0.png)
    



    
![png](output_34_1.png)
    


<h2 style="
    color: white;
    background-color: #e28f41;
    padding: 10px;
    border-radius: 8px;
">
Step 5: Try Clustering with k-Means and PCA
</h2>

Now let’s remove the labels and see if the computer can **discover** the species on its own using clustering.

We’ll use:
- **k-Means Clustering** to form 3 clusters since we know there are 3 species.
- **PCA** (Principal Component Analysis) to reduce 6 dimensions into 2 for visualization.
    - Do not worry too much about this. We will talk about it later.



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

# --- Set up figure for side-by-side comparison ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left plot: k-Means cluster assignments ---
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="Set2", s=50)
axes[0].set_title("k-Means Clusters (Unsupervised)")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

# --- Right plot: True species labels ---
scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap="viridis", s=50)
axes[1].set_title("True Species Labels")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")


# --- Add title ---
fig.suptitle("Comparing k-Means Clusters to True Species Labels (PCA Projection)", fontsize=14, y=1.02)
fig.subplots_adjust(wspace=0.3, top=0.88)  # manually adjust spacing for best fit

plt.show()



```


    
![png](output_36_0.png)
    


As we can see k_means does not do a great job! 

<h2 style="
    color: white;
    background-color: #e28f41;
    padding: 10px;
    border-radius: 8px;
">
Step 6: Wrap-Up
</h2>

- We successfully trained and evaluated multiple machine learning models.
- Even simple algorithms like **k-NN** and **Naïve Bayes** performed well.
- **Random Forests** often yield top accuracy on structured datasets like this one.
- **Clustering and PCA** can reveal natural structure even without labels.


<div style="
    background-color: #f0f7f4;
    border-left: 6px solid #4bbe7e;
    padding: 10px;
    border-radius: 5px;
">
<b>Key Takeaways:</b> 

- ML is not magic! It’s pattern recognition through math and logic.
- We’ll explore each of these methods in detail throughout the course.
- You now have a full end-to-end example of how ML workflows look in practice!
</div>

