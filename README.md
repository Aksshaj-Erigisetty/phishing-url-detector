# Phishing URL Detector 

A machine learning project to detect **phishing URLs** using multiple classification models — **Support Vector Machine (SVM)**, **Neural Network**, and **Naive Bayes** — based on URL features.  
Developed as part of the **CDS 303 Data Mining** course at George Mason University.

---

## Project Overview

Phishing is a cyberattack method where fake websites impersonate legitimate ones to steal personal or sensitive data.  
Our goal was to build models that can **automatically classify URLs as either “phishing” or “legitimate.”**

We used the **PhiUSIIL Phishing URL dataset**, containing over **235,000 URLs** and **56 features**, focusing on key traits such as:
- Number of Images
- Number of Self Redirects
- Largest Line of Code
- Number of External References

---

## CRISP-DM Workflow

1. **Business Understanding** – Identify how machine learning can detect phishing attempts.  
2. **Data Understanding** – Explore patterns and relationships among URL-based features.  
3. **Data Preparation** – Handle outliers, scale numeric data, and address class imbalance.  
4. **Modeling** – Train and compare multiple classifiers.  
5. **Evaluation** – Validate results using cross-validation, confusion matrices, and ROC curves.  
6. **Deployment** – Propose integration into a browser extension or company system.

---

## Models & Results

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:---------:|:----------:|:-------:|:----------:|
| **Support Vector Machine (SVM)** | **98.0%** | 0.97 | 0.98 | 0.98 |
| **Neural Network** | **98.6%** | 0.9868 | 0.9816 | 0.9842 |
| **Naive Bayes** | **96.2%** | 0.945 | 0.969 | 0.957 |

**Best Model:** SVM (RBF kernel, C = 1.0, gamma = scale, probability = True)

---
## Repository Structure
```
phishing-url-detector/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   |
│   ├── PhiUSIIL_Phishing_URL_Dataset.csv
│   └── phishing urls.xlsx
│   
│
│
├── notebooks/
│   ├── Naive-Bayes-Final 1.ipynb
│   └── Neutral-Network-Final 1.ipynb
│
└── reports/
    ├── Team 2- CDS 303 FINAL Report 1.pdf
    └── CDS 303-Team 2 Final Presentation 1.pdf

```
---

---

## Datasets Used

This project uses two datasets for building and testing phishing detection models:

### PhiUSIIL Phishing URL Dataset (Main)
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

### Custom Phishing URLs Dataset
- **File:** `phishing urls.xlsx`
- **Description:**  
  A smaller, manually curated dataset used for **testing feature extraction** and verifying the model pipeline during development.  
  It includes sample phishing and legitimate URLs with a simplified set of columns for quick experimentation.

---

Both datasets can be stored under:
```
├── PhiUSIIL_Phishing_URL_Dataset.csv
└── phishing urls.xlsx
```

---

## Key Insights

During the analysis of the PhiUSIIL Phishing URL dataset, several strong patterns emerged that differentiate phishing websites from legitimate ones.

### Technical Observations
- **Number of Self-Redirects:**  
  Phishing websites often redirect to themselves multiple times, making it difficult for users to trace the true destination.
- **Largest Line of Code:**  
  Many phishing pages have unusually long and inefficient code lines, which may be used to obfuscate malicious scripts.
- **External References:**  
  Phishing websites tend to have a high number of external links pointing to suspicious or unrelated domains.
- **Number of Images:**  
  Either too many or too few images — both extremes — are often indicators of fake websites.

### Model Findings
- **Support Vector Machine (SVM)** delivered the most consistent accuracy at **98%**.
- **Neural Network** slightly outperformed SVM in recall but required more computation time.
- **Naive Bayes** offered a lightweight baseline model with **96%** accuracy.
- Addressing **outliers** and **scaling** numeric features with **Min-Max normalization** significantly improved results.

### Practical Implications
- These results show that organizations can build **fast and reliable phishing detectors** using only URL-based features.
- This approach can be integrated into **browser extensions** or **email filters** to automatically warn users about suspicious links.

---

---

##  Evaluation & Visualization

The models were evaluated using several key metrics to ensure reliability and robustness.

### Evaluation Metrics
Each model was tested using **5-fold Stratified Cross-Validation**, ensuring both phishing and legitimate URLs were represented evenly across folds.

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:---------:|:----------:|:-------:|:----------:|
| **SVM (RBF Kernel)** | **98.0%** | 0.97 | 0.98 | 0.98 |
| **Neural Network** | **98.6%** | 0.9868 | 0.9816 | 0.9842 |
| **Naive Bayes** | **96.2%** | 0.945 | 0.969 | 0.957 |

**Best Model:**  
Support Vector Machine (RBF kernel) — offered the best balance between speed, accuracy, and generalization.

---

### Confusion Matrix (SVM)

|                | Predicted Phishing | Predicted Legitimate |
|----------------|-------------------:|---------------------:|
| **Actual Phishing** | True Positive (97%) | False Negative (2%) |
| **Actual Legitimate** | False Positive (1%) | True Negative (99%) |

- High true-positive and true-negative rates indicate that the SVM model is very effective at distinguishing phishing URLs.  
- False positives remain minimal, meaning few legitimate sites are incorrectly flagged.

---

### ROC Curve & AUC

- The **Receiver Operating Characteristic (ROC)** curve for SVM shows an **AUC ≈ 0.99**, indicating excellent separability between phishing and legitimate URLs.  
- Both the Neural Network and Naive Bayes models also achieved strong AUC values (above 0.95).  
- The ROC visualization confirms that the SVM’s probability predictions are highly reliable.

---

### Cross-Validation Summary

| Metric | Mean | Std. Deviation |
|:-------|:----:|:---------------:|
| **Precision** | 0.979 | ±0.002 |
| **Recall** | 0.981 | ±0.001 |
| **F1-score** | 0.980 | ±0.002 |

Cross-validation results show that the model is **consistent and stable** across different subsets of data, confirming that it is not overfitting.

---

### Visualization Highlights

From the analysis and reports:
- **Violin plots** showed skewed feature distributions but clear separability between classes.  
- **Correlation matrix** identified key predictors like `NoOfImage`, `LineOfCode`, and `NoOfExternalRef`.  
- **Pair plots** visually confirmed clusters of phishing vs. legitimate URLs.  
- **ROC curve and bar plots** validated the strong predictive performance of SVM.

---

*All plots and visuals can be found in the Jupyter notebooks and presentation slides (`/notebooks` and `/reports`).*

---

---

## 📘 Dataset Reference

> **Prasad, A. & Chandra, S. (2024).** *PhiUSIIL Phishing URL (Website)* [Dataset].  
> UCI Machine Learning Repository.  
> DOI: [10.1016/j.cose.2023.103545](https://doi.org/10.1016/j.cose.2023.103545)

This dataset contains over **235,000 URLs** with 56 extracted features describing URL behavior, structure, and content.  
It is widely used for research on phishing detection using machine learning models.

Additionally, a smaller custom dataset (`phishing urls.xlsx`) was used for testing and feature validation.

---


### ⭐ Acknowledgment
This project was developed as part of **CDS 303 – Data Mining** coursework.  
Special thanks to my teammates and instructors for their collaboration and guidance.

---
