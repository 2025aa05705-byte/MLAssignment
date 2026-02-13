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

Performance Comparison Table

|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|---|---|---|---|---|---|---|
|Logistic Regression|0\.9000|0\.9236|0\.8500|0\.8947|0\.8718|0\.7906|
|Decision Tree Classifier|0\.9800|0\.9737|1\.0000|0\.9474|0\.9730|0\.9580|
|K-Nearest Neighbors (KNN)|0\.8400|0\.9304|0\.8235|0\.7368|0\.7778|0\.6558|
|Naive Bayes (Gaussian)|0\.7800|0\.8778|0\.6429|0\.9474|0\.7660|0\.6109|
|Random Forest (Ensemble)|1\.0000|1\.0000|1\.0000|1\.0000|1\.0000|1\.0000|
|XGBoost (Ensemble)|0\.9800|1\.0000|1\.0000|0\.9474|0\.9730|0\.9580|

Observations
1. Random Forest (Ensemble): Achieved perfect scores across all metrics (1.0000), making it the absolute best-performing model on this test set.
2. XGBoost (Ensemble): Delivered exceptional results with 98% accuracy and perfect precision, generating zero false positives.
3. Decision Tree Classifier: Matched XGBoost's high performance with 98% accuracy and 1.0000 precision, proving highly effective.
4. Logistic Regression: Served as a strong linear baseline, achieving a solid 90% accuracy with well-balanced precision and recall.
5. K-Nearest Neighbors (KNN): Showed moderate performance with 84% accuracy, struggling slightly compared to the tree-based models.
6. Naive Bayes (Gaussian): Yielded the lowest accuracy (78%) and precision due to high false positives, despite catching most of the actual spam (high recall).

4. Project Structure
app.py: Streamlit frontend application.
train_models.py: Script to download data, train models, and generate artifacts.
model/: Folder containing trained models (.pkl files) and scalers.
requirements.txt: List of dependencies.

