# 01. Traditional Machine Learning Overview

## 📌 Lecture Overview

This lecture introduces **Traditional Machine Learning** as a framework for learning patterns from data and generalizing them to new inputs.  
Instead of manually designing rules, models are trained to capture statistical relationships and make predictions or decisions based on data.

---

## 📖 Full Article

- English  
  https://zeromathai.com/en/dl-traditional-ml-overview-en/

- Korean  
  https://zeromathai.com/dl-traditional-ml-overview/

👉 Other languages are available on the website

---

## 📚 Table of Contents

- Concept of Machine Learning  
- Learning Paradigms in Machine Learning  
- Families of Machine Learning Algorithms  
- Evaluation Metrics and Interpretation  

---

## 1. Concept of Machine Learning

Machine Learning is an empirical approach where models learn patterns from data.

- Learns relationships between input and output  
- Builds statistical models instead of rule-based logic  
- Generalizes to unseen data  

👉 Key idea:  
A model does not memorize data — it **learns patterns** that apply beyond the training set.

---

## 2. Learning Paradigms in Machine Learning

Machine Learning differs based on how learning signals are provided.

### 🔹 Supervised Learning
- Uses labeled data (input + correct output)
- Learns to minimize prediction error
- Used for:
  - Classification
  - Regression

---

### 🔹 Unsupervised Learning
- No labels provided
- Discovers hidden patterns or structure

Used for:
- Clustering
- Dimensionality Reduction

---

### 🔹 Reinforcement Learning
- Agent interacts with an environment
- Learns actions that maximize reward

👉 Core idea:  
Trial → Feedback → Improvement

Used in:
- Robotics
- Game AI
- Sequential decision-making

---

## 3. Families of Machine Learning Algorithms

Machine Learning algorithms can be grouped by how they solve problems.

---

### 🔹 Clustering Algorithms
- Group similar data points together
- No labels required

Examples:
- K-means
- Hierarchical Clustering

Metrics:
- Silhouette Score
- Homogeneity
- Completeness

---

### 🔹 Classification Algorithms
- Predict discrete categories

Examples:
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM

Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

### 🔹 Regression Algorithms
- Predict continuous values

Examples:
- Linear Regression
- Polynomial Regression
- Ridge / Lasso

Metrics:
- MSE
- RMSE
- MAE
- R²

---

### 🔹 Ensemble Learning
- Combines multiple models

Goal:
- Improve accuracy
- Reduce variance and bias

Methods:
- Bagging
- Boosting

---

## 4. Evaluation Metrics and Interpretation

Model evaluation depends on the problem context.

---

### 🔹 Classification Metrics
- Accuracy: overall correctness  
- Precision: correctness of positive predictions  
- Recall: coverage of actual positives  
- F1-score: balance of precision and recall  
- ROC-AUC: threshold-independent performance  

---

### 🔹 Regression Metrics
- MSE: average squared error  
- RMSE: error in original scale  
- MAE: absolute error  
- R²: explanatory power  

---

### 🔹 Clustering Metrics
- Silhouette Score: cluster separation  
- Homogeneity: same-class grouping  
- Completeness: all class members grouped  
- V-measure: balance of homogeneity and completeness  

---

## 🎯 Key Insight

Traditional Machine Learning provides the foundation for modern AI:

- Defines how models learn from data  
- Introduces core problem types  
- Establishes evaluation frameworks  

👉 These ideas directly lead to:
- Deep Learning
- Representation Learning
- Generative AI

---

## 🔗 Navigation

← Previous: (coming soon)  
→ Next: [Foundations of AI and Deep Learning](02-foundations-of-ai.md)
