# Intern Performance Prediction & Analytics System

## 📌 Project Overview

This project predicts intern performance based on task completion, consistency, and engagement metrics using Machine Learning.


## 🚀 Sprint 1 – Data Foundation
- Defined dataset structure
- Created synthetic dataset (40 interns)
- Performed data cleaning and preprocessing
- Applied feature scaling and train-test split


## 🤖 Sprint 2 – Model Development
- Trained Machine Learning models
- Used:
  - XGBoost (Classification)
  - Gradient Boosting Regressor (Regression)
- Compared model performance
- Selected best model based on accuracy


## 📊 Results

- XGBoost Accuracy: 1.00  
- Regression Accuracy: 1.00  
- Mean Squared Error: ~0  


## 🛠️ Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost


### Note:
The high accuracy is due to the use of a synthetic dataset generated using rule-based logic. 
Since the model learns the same pattern used to create the data, it achieves near-perfect performance. 
In real-world scenarios, the accuracy may vary.


## ⚙️ Sprint 3 – Optimization & Evaluation
- Applied advanced feature engineering
- Created new features:
  - efficiency
  - engagement_index
  - performance_intensity
- Improved model performance using optimized dataset
- Evaluated using classification metrics

### 📊 Model Performance:
- Accuracy: 1.00  
- Precision: 1.00  
- Recall: 1.00  
- F1-Score: 1.00  

### 📉 Confusion Matrix:

[[4 0]
[0 4]]

## 🚀 Sprint 4 – Model Deployment

- Converted trained ML model into a REST API using FastAPI
- Built an interactive web interface using Jinja2 templates
- Enabled real-time prediction with user input form
- Implemented feature engineering inside API pipeline
- Integrated trained model (.pkl) with backend for live predictions

### 🌐 Deployment Features:
- Live Prediction API endpoint (`/predict`)
- Supports real-time intern performance classification
- Dynamic feature generation before prediction
- User-friendly UI for easy interaction

### 🧠 Backend Workflow:
1. User inputs data via web form  
2. API processes input and generates engineered features  
3. Data is scaled using StandardScaler  
4. XGBoost model predicts performance  
5. Output displayed as Low / Medium / High  

### ⚡ API Status:
- Local API running successfully  
- Ready for cloud deployment (Render / Railway)

### 📌 Example Output:
- Input: Intern performance metrics  
- Output: Performance Level (Low / Medium / High)

---

## 🎯 Final Outcome

The system successfully integrates:
- Machine Learning Model  
- Feature Engineering Pipeline  
- Real-Time Prediction API  
- Interactive User Interface  

This project demonstrates an end-to-end AI solution from data processing to deployment.

## 🌐 Live Demo
👉 https://karshavarthini-intern-performance-prediction.hf.space
