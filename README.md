# Anomaly Detector

A Python tool that detects anomalous financial transactions 
using two statistical methods: z-score and IQR outlier detection.

## Dataset

Credit Card Fraud Detection dataset (Kaggle — MLG-ULB)
- 284,807 transactions
- 492 fraudulent (0.172% of total)
- Features are PCA-transformed for confidentiality

## Methods

### Z-score
Flags transactions where the amount is more than 3 standard 
deviations from the mean. Simple and fast.

### IQR (Interquartile Range)
Flags transactions outside 1.5x the interquartile range. 
More robust to extreme outliers than z-score.

## How to run

git clone git@github.com:YOUR-USERNAME/anomaly-detector.git
cd anomaly-detector
python -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib scikit-learn

Download creditcard.csv from Kaggle and place in root folder, then:

python explore.py
python detector.py

## What I found

The z-score method flagged 4076 transactions, correctly 
identifying 492 of the 492 fraudulent ones.
The IQR method flagged 11 transactions, correctly 
identifying  fraudulent ones.

Neither method alone is sufficient — real fraud detection 
combines statistical methods with machine learning models 
trained on labelled data.

## Tech stack

- Python 3.12
- pandas — data loading and manipulation
- numpy — statistical calculations
- matplotlib — visualisation

## Next steps

- Implement Isolation Forest (scikit-learn) for ML-based detection
- Compare precision and recall across all three methods
- Build a combined ensemble approach