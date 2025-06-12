# Churn_Prediction_Analysis
# Telco Customer Churn Prediction and Analysis

## Project Overview

This project focuses on predicting customer churn in a telecommunications company. By leveraging historical customer data, we build and evaluate machine learning models to identify customers at high risk of churning. The goal is to provide actionable insights and empower the company to develop targeted retention strategies, ultimately reducing customer attrition and improving customer lifetime value.

The project encompasses the entire data science lifecycle:
* **Data Acquisition & Understanding:** Sourcing and initial exploration of the dataset, including SQL integration simulation.
* **Data Wrangling & Preprocessing:** Cleaning, transforming, and preparing the data for modeling.
* **Exploratory Data Analysis (EDA) & Feature Selection:** In-depth analysis to uncover patterns and identify key drivers of churn.
* **Machine Learning Model Building & Evaluation:** Training various classification models, hyperparameter tuning, and performance assessment.
* **Recommendations:** Translating model insights into actionable business strategies.

### Dataset

The dataset used in this project is the "Telco Customer Churn" dataset, which contains information about a fictional telecommunications company's customers and their churn status. It includes various demographic, service-related, and billing details.

**Key Features include:**
* **Demographic Info:** Gender, SeniorCitizen, Partner, Dependents
* **Service Info:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
* **Account Info:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
* **Target Variable:** Churn (Yes/No - converted to 1/0)
## Project Structure

The project is organized into several Jupyter notebooks, each covering a specific phase of the data science pipeline:

* `1.0_Data_Acquisition_and_Initial_Wrangling.ipynb`: Focuses on loading the raw data, understanding its structure, and simulating initial data extraction and inspection using SQLite.
* `2.0_Data_Wrangling_and_Preprocessing.ipynb`: Details the steps involved in cleaning, transforming, and preparing the data. This includes handling missing values, encoding categorical features, and creating new features like `TenureGroup`.
* `3.0_EDA_and_Feature_Selection.ipynb`: Conducts extensive Exploratory Data Analysis to visualize distributions, relationships between features, and correlations with the 'Churn' target. Key insights are derived to inform feature selection.
* `4.0_Machine_Learning_Model_Building_and_Evaluation.ipynb`: Covers the training and evaluation of various machine learning classification models (Logistic Regression, Decision Tree, Random Forest), hyperparameter tuning using GridSearchCV, and feature importance analysis.

**Other Files:**
* `Data/`: This directory contains the processed datasets (`telco_churn_processed.csv`, `telco_churn_pre_encoding.csv`, `feature_importances.csv`) and the saved visualizations from the notebooks.
* `models/`: This directory stores the trained machine learning model (`best_churn_prediction_model.joblib`).

## Methodology & Analysis Highlights

### Data Acquisition & Initial Wrangling
* Loaded data from CSV and simulated SQLite database integration to mimic real-world data sourcing.
* Initial data inspection revealed column data types and potential issues like `TotalCharges` containing empty strings.

### Data Wrangling & Preprocessing
* Handled `TotalCharges` inconsistencies by replacing empty strings with '0' and converting to numeric.
* Transformed `Churn` and `SeniorCitizen` to binary (0/1).
* Converted `gender` to `gender_Male` (binary).
* Categorized `tenure` into `TenureGroup` bins for better analysis of customer loyalty periods.
* Applied One-Hot Encoding to all other categorical features to prepare data for machine learning models.

### Exploratory Data Analysis (EDA) & Feature Selection

**Key Findings:**
* **Churn Rate:** The dataset showed an overall churn rate of approximately **26.54%**, indicating a significant challenge for the telecom company.
* **Contract Type:** Customers on **Month-to-month contracts** exhibited significantly higher churn rates compared to those on one-year or two-year contracts.
* **Internet Service:** Customers with **Fiber optic internet service** showed a higher propensity to churn than those with DSL or no internet service.
* **Online Security & Tech Support:** Customers *without* online security or tech support services were more likely to churn, highlighting the importance of these add-ons for retention.
* **Payment Method:** **Electronic check** users had a noticeably higher churn rate.
* **Tenure:** Newer customers (lower tenure) showed higher churn rates, while long-term customers were more stable.
* **Monthly Charges:** Churners tended to have higher monthly charges on average.

**Visualizations (Examples - *Replace with actual image paths after saving them from notebooks*):**
* **Distribution of Customer Churn:**
    ![Churn Distribution](Data/churn_distribution.png)
* **Churn Rate by Contract Type:**
    ![Churn Rate by Contract Type](Data/categorical_churn_rates.png) <!-- Ensure you have this specific plot, or pick another representative categorical plot -->
* **Monthly Charges Distribution by Churn:**
    ![Monthly Charges Distribution](Data/numerical_feature_histograms.png) <!-- Or a specific plot for MonthlyCharges if separate -->
* **Correlation Heatmap:**
    ![Correlation Heatmap](Data/correlation_heatmap.png)

### Machine Learning Model Building & Evaluation

We trained and evaluated three classification models to predict churn: Logistic Regression, Decision Tree, and Random Forest. Data was split into 80% training and 20% testing, using stratification to maintain churn class balance.

**Evaluation Metrics Focused On:**
* **Accuracy:** Overall correctness of predictions.
* **Precision:** Proportion of positive identifications that were actually correct (minimizing false positives for churn).
* **Recall (Sensitivity):** Proportion of actual positives that were correctly identified (minimizing false negatives for churn, i.e., not missing potential churners).
* **F1-Score:** Harmonic mean of Precision and Recall.
* **ROC AUC Score:** Measures the model's ability to distinguish between churners and non-churners.

**Model Performance Summary (on Test Set):**

| Model              | Accuracy | ROC AUC | Precision (Churn=1) | Recall (Churn=1) | F1-Score (Churn=1) |
|--------------------|----------|---------|---------------------|------------------|--------------------|
| Logistic Regression| 0.8013   | 0.8424  | 0.66                | 0.52             | 0.58               |
| Decision Tree      | 0.7303   | 0.6406  | 0.49                | 0.43             | 0.46               |
| **Random Forest (Best)**| **0.8034** | **0.8450** | **0.67** | **0.54** | **0.60** |
*(Note: These values are based on the outputs provided in your `4.0_Machine_Learning_Model_Building_and_Evaluation.ipynb` snippet for a general RandomForestClassifier. The values for the *tuned* Random Forest might differ slightly and should be used from your notebook's final output.)*

**Hyperparameter Tuning:**
* `GridSearchCV` was employed to optimize the Random Forest model, searching for the best combination of `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` to maximize ROC AUC.
* The **Best Random Forest Model** (after tuning) showed slightly improved or comparable performance, demonstrating the value of optimization.

**Feature Importance:**
The Random Forest model provided insights into the most important features driving churn. The top features include:
* `Contract_Month-to-month`
* `tenure`
* `InternetService_Fiber optic`
* `TotalCharges`
* `MonthlyCharges`
* `PaymentMethod_Electronic check`
* `OnlineSecurity`
* `TechSupport`

**Feature Importance Visualization:**
![Feature Importance](Data/feature_importance_rf.png)

## Actionable Business Recommendations

Based on the EDA and the machine learning model's insights, the following recommendations are proposed to mitigate customer churn:

1.  **Target Month-to-Month Customers:** Develop specific retention programs (e.g., promotional discounts for signing longer-term contracts, loyalty bonuses) for customers on month-to-month contracts, as they are the highest churn risk group.
2.  **Improve Fiber Optic Service Satisfaction:** Investigate the reasons for high churn among Fiber Optic internet users. This could involve network stability improvements, better customer support for technical issues, or competitive pricing adjustments.
3.  **Enhance Security & Tech Support Offerings:** Promote and perhaps bundle online security and tech support services. Customers lacking these services are more prone to churn, suggesting they may feel unsupported or insecure.
4.  **Optimize Electronic Check Payment Experience:** Analyze the friction points for customers using electronic checks. This might involve exploring more convenient or secure payment options, or providing incentives for alternative payment methods.
5.  **Proactive Engagement for New Customers:** Implement early intervention programs for new customers (low tenure), as they are more likely to churn. This could include welcome calls, service setup assistance, or special introductory offers.
6.  **Address High Monthly Charges:** Review pricing strategies, especially for customers with high monthly bills. Consider personalized plans or value-added services to justify costs for high-spending customers.
7.  **Monitor Customer Activity and Service Usage:** Continuously monitor service usage patterns and overall customer activity. A sudden drop in usage or a spike in support calls could be early indicators of churn.

By focusing on these key areas, the telecom company can proactively address the factors leading to churn and improve customer retention.

## Future Work

* Explore more advanced machine learning models (e.g., XGBoost, LightGBM, Neural Networks) to potentially improve prediction accuracy.
* Implement A/B testing for proposed retention strategies to measure their effectiveness.
* Integrate the model into a real-time prediction system for immediate churn risk assessment.
* Conduct a deeper dive into specific churn reasons through qualitative data (customer feedback, call transcripts) if available.

