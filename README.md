# Phishing URL Detector 🧠🔗

A machine learning project to detect **phishing URLs** using multiple classification models — **Support Vector Machine (SVM)**, **Neural Network**, and **Naive Bayes** — based on URL features.  
Developed as part of the **CDS 303 Data Mining** course at George Mason University.

---

## 📊 Project Overview

Phishing is a cyberattack method where fake websites impersonate legitimate ones to steal personal or sensitive data.  
Our goal was to build models that can **automatically classify URLs as either “phishing” or “legitimate.”**

We used the **PhiUSIIL Phishing URL dataset**, containing over **235,000 URLs** and **56 features**, focusing on key traits such as:
- Number of Images
- Number of Self Redirects
- Largest Line of Code
- Number of External References

---

## 🧩 CRISP-DM Workflow

1. **Business Understanding** – Identify how machine learning can detect phishing attempts.  
2. **Data Understanding** – Explore patterns and relationships among URL-based features.  
3. **Data Preparation** – Handle outliers, scale numeric data, and address class imbalance.  
4. **Modeling** – Train and compare multiple classifiers.  
5. **Evaluation** – Validate results using cross-validation, confusion matrices, and ROC curves.  
6. **Deployment** – Propose integration into a browser extension or company system.

---

## ⚙️ Models & Results

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:---------:|:----------:|:-------:|:----------:|
| **Support Vector Machine (SVM)** | **98.0%** | 0.97 | 0.98 | 0.98 |
| **Neural Network** | **98.6%** | 0.9868 | 0.9816 | 0.9842 |
| **Naive Bayes** | **96.2%** | 0.945 | 0.969 | 0.957 |

✅ **Best Model:** SVM (RBF kernel, C = 1.0, gamma = scale, probability = True)



