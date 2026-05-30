# Census Income Prediction Machine Learning Pipeline

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline designed to predict whether an individual's annual income exceeds \$50,000 based on demographic attributes extracted from 1994 Census data (e.g., age, education, occupation, marital status, and hours worked per week).

## 📊 Data Pipeline Workflow
The project implements rigorous preprocessing methods to ensure high-quality training conditions:
1. **Data Cleansing**: Identified hidden missing values masked as `"?"`, converted them into `NaN`, and eliminated incomplete records. Handled structural duplicate entries.
2. **Outlier Mitigation**: Filtered extreme numerical deviations across features such as `age` and `hours-per-week` using the **Interquartile Range (IQR)** method.
3. **Categorical Encoding**: Transformed descriptive non-numeric strings into structured numerical matrices using `LabelEncoder`.
4. **Data Partitioning**: Stratified features into an 80% training set and a 20% validation pool.

## 🛠️ Mathematical Approach for Outlier Treatment
Outliers were identified using boundaries defined as:
$$IQR = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times IQR$$
$$\text{Upper Bound} = Q_3 + 1.5 \times IQR$$
Any observations outside $[\text{Lower Bound}, \text{Upper Bound}]$ were removed to prevent distortion in regression and split dynamics.

## 📈 Supervised Learning Performance
Three distinct algorithms were evaluated using classification accuracy:
* **Logistic Regression**: Serves as the baseline linear classifier.
* **Decision Tree Classifier**: Fine-tuned over progressive tree depths to control variance and overfitting.
* **Random Forest Classifier**: Evaluated as an ensemble bagging method.

## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone [https://github.com/subhankar0296/census-income-prediction.git](https://github.com/subhankar0296/census-income-prediction.git)
   cd census-income-prediction
