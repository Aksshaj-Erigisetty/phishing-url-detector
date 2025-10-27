# Phishing URL Detector

**Brief:** A machine-learning project to detect phishing URLs using feature engineering and classification models.  
**Author:** Anantha Aksshaj Erigisetty (Aksshaj)

## Contents
- `data/` — raw and processed datasets (not all raw data included here)
- `notebooks/` — EDA and modeling notebooks
- `src/` — python scripts for preprocessing, training, and inference
- `models/` — saved model artifacts
- `requirements.txt` — Python dependencies

## Project overview
1. **Data**: `data/raw/phishing_data.csv` — URL list with `label` column (1 = phishing, 0 = benign).
2. **Feature engineering**: extract URL-based features (length, special chars, entropy, domain age if available, host tokens, presence of IP, suspicious TLDs, URL shortening, etc.).
3. **Models**: baseline (Logistic Regression), tree-based (RandomForest/XGBoost), and an ensemble.
4. **Evaluation**: ROC-AUC, Precision-Recall, F1-score; show confusion matrices and top features.
5. **Deliverables**: reproducible notebook, training script `src/train.py`, and inference script `src/predict.py`.

## Quick start
```bash
git clone https://github.com/yourusername/phishing-detector.git
cd phishing-detector
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# preprocess
python src/data_preprocess.py --input data/raw/phishing_data.csv --output data/processed/phishing_processed.csv
# train
python src/train.py --data data/processed/phishing_processed.csv --model models/best_model.pkl
# predict (example)
python src/predict.py --model models/best_model.pkl --url "http://example-suspicious.com"
