# Drug Persistency Classification Model

## Problem Description
Understanding drug persistency is crucial for pharmaceutical companies. This project aims to develop a machine learning model to predict drug persistency based on physician prescription data.

## Business Objective
ABC Pharma is seeking to automate the process of identifying drug persistency. The goal of creating this classification model is to offer insights into factors impacting drug persistency, aiming to enhance drug adherence and optimize patient outcomes.

## Project Lifecycle

- **Problem & Data Understanding**: 1 week
- **Data Cleaning & Transformation**: 1 week
- **Exploratory Data Analysis (EDA)**: 1 week
- **Model Selection**: 1 week
- **Model Building**: 1 week
- **Preparing Final Report**: 1 week
- **Total Duration**: 6 weeks

## Data Source
The dataset has been generously provided by ABC Pharma. It comprises patient demographics, physician specialty, clinical indicators, disease/treatment specifics, and adherence data. Spanning over 100,000 anonymized patient records, this dataset is primed for deep analysis.

## Data Cleaning

The provided Python code illustrates the data cleaning process: handling unknown values, dropping irrelevant columns, encoding categorical variables, imputing missing data, and saving the cleaned dataset for further processing.

```python
import pandas as pd
import numpy as np
# ... [rest of the data cleaning code]
```

## EDA Highlights

- Factors influencing drug persistency are likely a mix of genetic, cultural, and regional elements.
- Women have a higher likelihood of being affected than men.
- There's a prominent correlation between fracture incidents and glucose levels during treatment.
- Persistent patients often get Dexa scans.
- Patients with Vitamin D insufficiency (especially women) exhibit the most significant risk factor.
  
## Model Exploration

Given the dataset's intricacy, a variety of models from different families (Linear, Ensemble, Boosting) are recommended. Exploring these will help to better understand the dataset and optimize prediction accuracy.

### Proposed Models:

1. **Logistic Regression**
   - `max_iter`: 1000
   - **ROC-AUC Score**: 0.8723773078872916

2. **Random Forest**
   - `n_estimators`: 1000
   - **ROC-AUC Score**: 0.8572641876266296

3. **SVM**
   - `probability`: true
   - **ROC-AUC Score**: 0.8727125545797969

4. **AdaBoost**
   - `base_estimator`: DecisionTreeClassifier(max_depth=1)
   - `n_estimators`: 1000
   - `learning_rate`: 1.0
   - **ROC-AUC Score**: 0.8720420611947864

## Feature Selection
Chi-Squared testing was leveraged for feature selection to pinpoint the top 5 impactful features. The scores represent each feature's influence, with `Dexa_Freq_During_Rx` outperforming others by a considerable margin.

## Deployment

The chosen AdaBoost model, fine-tuned with feature selection, is now live and hosted on Heroku. Check out the prediction tool [here](https://drug-persistency-predictor.herokuapp.com/).
