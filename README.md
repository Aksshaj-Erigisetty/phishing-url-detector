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

---
## 🗂 Repository Structure
```
phishing-url-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│ ├── raw/ # store raw CSVs (ignored in git)
│ └── processed/ # scaled and cleaned training data
│
├── src/
│ ├── data_preprocess.py # scales features and prepares processed CSV
│ ├── train.py # trains SVM and saves best_model.pkl
│ ├── predict.py # generates predictions from trained model
│ └── svm_experiment.py # your original SVM script
│
├── notebooks/
│ ├── Naive-Bayes-Final.ipynb
│ └── Neural-Network-Final.ipynb
│
└── reports/
├── Final_Presentation.pdf
└── Final_Report.pdf
```
---

---

## 📊 Datasets Used

This project uses two datasets for building and testing phishing detection models:

### 1️⃣ PhiUSIIL Phishing URL Dataset (Main)
- **File:** `PhiUSIIL_Phishing_URL_Dataset.csv`
- **Source:** [UCI Machine Learning Repository](https://doi.org/10.1016/j.cose.2023.103545)
- **Size:** 235,796 rows × 56 columns  
- **Description:**  
  This dataset contains features extracted from real-world URLs and their corresponding labels indicating whether they are **phishing (0)** or **legitimate (1)**.  
  It includes structural, lexical, and behavior-based URL attributes such as:
  - Number of Images  
  - Number of Self References  
  - Largest Line of Code  
  - Number of External References  
  - URL Length, Domain Age, and more

### 2️⃣ Custom Phishing URLs Dataset
- **File:** `phishing urls.xlsx`
- **Description:**  
  A smaller, manually curated dataset used for **testing feature extraction** and verifying the model pipeline during development.  
  It includes sample phishing and legitimate URLs with a simplified set of columns for quick experimentation.

---

✅ Both datasets can be stored under:
```
├── PhiUSIIL_Phishing_URL_Dataset.csv
└── phishing urls.xlsx
```

---

## 🧠 Key Insights

During the analysis of the PhiUSIIL Phishing URL dataset, several strong patterns emerged that differentiate phishing websites from legitimate ones.

### 1️⃣ Technical Observations
- **Number of Self-Redirects:**  
  Phishing websites often redirect to themselves multiple times, making it difficult for users to trace the true destination.
- **Largest Line of Code:**  
  Many phishing pages have unusually long and inefficient code lines, which may be used to obfuscate malicious scripts.
- **External References:**  
  Phishing websites tend to have a high number of external links pointing to suspicious or unrelated domains.
- **Number of Images:**  
  Either too many or too few images — both extremes — are often indicators of fake websites.

### 2️⃣ Model Findings
- **Support Vector Machine (SVM)** delivered the most consistent accuracy at **98%**.
- **Neural Network** slightly outperformed SVM in recall but required more computation time.
- **Naive Bayes** offered a lightweight baseline model with **96%** accuracy.
- Addressing **outliers** and **scaling** numeric features with **Min-Max normalization** significantly improved results.

### 3️⃣ Practical Implications
- These results show that organizations can build **fast and reliable phishing detectors** using only URL-based features.
- This approach can be integrated into **browser extensions** or **email filters** to automatically warn users about suspicious links.

---


