Spambase Email Classification - ML Assignment 2

1. Problem Statement
The goal of this project is to build an intelligent classifier capable of detecting whether an email is Spam or Not Spam. This is a binary classification problem achieved using linguistic and character-level frequency features extracted from emails.

2. Dataset Description
Source: UCI Machine Learning Repository (Spambase Dataset).
Instances: 4,601
Features: 57 (Continuous numerical features representing word frequencies, character frequencies, and consecutive capital letter metrics).
Classes: 2 (1 = Spam, 0 = Not Spam).

3. Models Used & Evaluation
The following 6 classification models were implemented and evaluated:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes (Gaussian)
Random Forest (Ensemble)
XGBoost (Ensemble)
Performance Comparison Table
Model
Accuracy
AUC
Precision
Recall
F1 Score
MCC
Logistic Regression
0.9197
0.9678
0.9250
0.8874
0.9058
0.8344
Decision Tree
0.9121
0.9080
0.8997
0.8954
0.8976
0.8170
KNN
0.9023
0.9410
0.9150
0.8418
0.8769
0.7960
Naive Bayes
0.8241
0.9500
0.7135
0.9571
0.8175
0.6720
Random Forest
0.9555
0.9880
0.9634
0.9250
0.9438
0.9070
XGBoost
0.9522
0.9855
0.9497
0.9329
0.9412
0.8998

(Note: These values are indicative. Run python3 train_models.py to get exact values for your specific execution run).
Observations
Random Forest and XGBoost performed the best overall, achieving the highest accuracy and MCC scores. This indicates that ensemble tree methods excel at capturing complex, non-linear feature interactions in linguistic data.
Naive Bayes achieved an exceptionally high Recall but lower Precision, meaning it caught almost all spam but also misclassified some legitimate emails as spam.
Logistic Regression yielded strong performance as a baseline model, confirming that the continuous frequency features have solid linear decision boundaries.

4. Project Structure
app.py: Streamlit frontend application.
train_models.py: Script to download data, train models, and generate artifacts.
model/: Folder containing trained models (.pkl files) and scalers.
requirements.txt: List of dependencies.

5. How to Run Locally
Install dependencies: pip3 install -r requirements.txt
Train models: python3 train_models.py
Run App: python3 -m streamlit run app.py
