---
layout: jupyternotebook
title: Machine Learning Foundations – Project 06
course_home: /courses/machine-learning-foundations/
nav_section: homework
nav_order: 6
---

# Project 6: Build Your Own Adventure


### Overview

Use a dataset of your choosing to answer a meaningful question using one or more supervised or unsupervised machine-learning methods.

---

### Learning Objectives

By the end of this project, you will be able to:
- Independently source and prepare a dataset suitable for ML analysis.
- Identify and clearly state a research or application question.
- Select, justify, and implement an appropriate ML method (classification, regression, or clustering).
- Evaluate model performance with appropriate metrics and validation.
- Interpret and communicate results through visualizations and clear reasoning.

---

## Instructions

1. **Choose a dataset**: Find a dataset from the internet, or maybe your own materials. Here are some helpful locations to find reliable data:

Choosing an interesting dataset is one of the most important (and fun!) parts of this project.  You may need to Google or GAI you way to figureing out how to load your data into your Notebook. Let me know if you need any help!

Below are some reliable places to find datasets that work well for machine learning analysis.


**Kaggle Datasets**
- <https://www.kaggle.com/datasets>  
Kaggle hosts thousands of community datasets covering every topic imaginable — from health and finance to music, sports, and social media.  
You can search by keywords (e.g., “climate,” “education,” “nutrition,” “energy”) and download the data as `.csv` files.  
Many come with descriptions and context.


**UCI Machine Learning Repository**
- <https://archive.ics.uci.edu/ml/index.php>  
A classic source of clean, ready-to-analyze datasets.  
Examples include wine quality, breast cancer, heart disease, and student performance datasets — perfect for testing models like Decision Trees, Random Forests, and Logistic Regression.


**Google Dataset Search**
- https://datasetsearch.research.google.com/  
A search engine for open datasets across the web — useful if you want something more niche or large-scale.
(Be sure the data is in a usable format like .csv or .xlsx.)


**Data.gov (U.S. Government Open Data)**
- https://www.data.gov/  
Tons of public datasets about health, education, environment, and more — great for projects with social or scientific themes.
Examples:
- COVID-19 statistics
- Renewable energy production
- Local weather and climate records


**Other Fun & Specialized Sources**

- FiveThirtyEight Datasets (https://github.com/fivethirtyeight/data): Datasets behind their news analyses (sports, politics, culture).

- Our World in Data (https://ourworldindata.org/): Long-term, global data on climate, population, health, and inequality.

- Awesome Public Datasets (GitHub) (https://github.com/awesomedata/awesome-public-datasets): A huge collection of curated data links by topic.


**Tips for Selecting Your Data**

- Choose something that interests you personally — sports, movies, weather, biology, etc.
- Aim for 100–10,000 rows and 3–15 columns — large enough to find patterns, but small enough to handle in memory.
- Make sure your target variable (y) is well-defined (a category for classification or a number for regression).
- Prefer datasets with column headers and minimal missing data.
- If possible, pick a dataset that could support more than one model type.

**Note**: ***Do not use*** datasets that contain sensitive or personal information (e.g., names, emails, or health records). Stick to public, anonymized data.



2. **Your Code**: You should use code we have built up throughout the other projects and adapt them to your new data set and question(s). You should accomplish the following with said code:

    - [ ] Train one or more models (justify your choices).
    - [ ] Evaluate with relevant metrics.
    - [ ] Include at least one interpretability visualization (e.g., feature importances, coefficients, cluster map).

    Your notebook should run top-to-bottom **without errors**, setting a `random_state` where applicable for reproducibility, and should produce all of the information and plots needed for your report.

3. **Report and Reflection**:

    **Write a Report** (with key plots). Focus your discussion on:

    - What did you learn about the data?
        - What was your data set and where was it from?
        - What question did you set out to answer or what did you set out to accomplish?
            - What predictive models were you hoping to create?
        - What models did you use and why?
        - How reliable are your results?
        - What were your results/deliverables?
    - What were the limitations or biases in the data or methods?
    - What would you explore next with more time?

    **Write a Reflection.**
    - This is your chance to reflect on this project and on the course itself. 
    - What did learn? 
    - What did you wish you learned better? 
    - How was the class? How could it be improved?

    This is your time to speak! 

    If you would prefer to do this an anonymous manner, please complete the course evaluation through the school. 