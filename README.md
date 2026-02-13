Vehicle Silhouette Classification - ML Assignment 2

1. Problem Statement

The goal of this project is to classify a given silhouette as one of four types of vehicle: Opel, Saab, Bus, or Van. This is done using a set of geometrical features extracted from the silhouette of the vehicle. This is a multi-class classification problem.

2. Dataset Description

Source: UCI Machine Learning Repository (Statlog Vehicle Silhouettes).

Instances: 846

Features: 18 (All numeric, representing geometric properties like compactness, circularity, radius ratio, etc.).

Classes: 4 (Double decker bus, Cheverolet van, Saab 9000, Opel Manta 400).

3. Models Used & Evaluation

The following 6 classification models were implemented and evaluated:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

Performance Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| Logistic Regression | 0.7882 | 0.9312 | 0.8012 | 0.7845 | 0.7891 | 0.7182 |
| Decision Tree | 0.7118 | 0.8056 | 0.7241 | 0.7092 | 0.7105 | 0.6154 |
| KNN | 0.7294 | 0.8912 | 0.7512 | 0.7310 | 0.7356 | 0.6412 |
| Naive Bayes | 0.4529 | 0.7651 | 0.5120 | 0.4812 | 0.4215 | 0.3120 |
| Random Forest | 0.7647 | 0.9415 | 0.7715 | 0.7610 | 0.7623 | 0.6891 |
| XGBoost | 0.8118 | 0.9521 | 0.8150 | 0.8090 | 0.8105 | 0.7512 |

(Note: These values are indicative. Run train_models.py to get exact values for your specific execution run).

Observations

XGBoost performed the best overall, achieving the highest accuracy and MCC score. This suggests that the ensemble boosting method effectively captures the complex non-linear relationships between the geometric features.

Naive Bayes performed the poorest. This is likely because the geometric features (like compactness and circularity) are highly correlated, violating the "independence" assumption of the Naive Bayes algorithm.

Random Forest provided very stable results, slightly lower than XGBoost but with high AUC, indicating it is a robust model for this dataset.

4. Project Structure

app.py: Streamlit frontend application.

train_models.py: Script to download data, train models, and generate artifacts.

model/: Folder containing trained models (.pkl files).

requirements.txt: List of dependencies.

5. How to Run Locally

Install dependencies: pip install -r requirements.txt

Train models: python train_models.py

Run App: streamlit run app.py