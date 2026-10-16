import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('creditcard.csv')

# Z-score method
mean = df['Amount'].mean()
std = df['Amount'].std()
df['zscore'] = (df['Amount'] - mean) / std

threshold = 3
df['zscore_flag'] = (df['zscore'].abs() > threshold).astype(int)

print("=== Z-Score Method ===")
print(f"Transactions flagged: {df['zscore_flag'].sum()}")
print(f"Actual fraudulent: {df['Class'].sum()}")

flagged = df[df['zscore_flag'] == 1]
correct = flagged[flagged['Class'] == 1]
print(f"Correctly flagged fraud: {len(correct)}")
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['iqr_flag'] = ((df['Amount'] < lower) | (df['Amount'] > upper)).astype(int)

print("\n=== IQR Method ===")
print(f"Transactions flagged: {df['iqr_flag'].sum()}")

flagged_iqr = df[df['iqr_flag'] == 1]
correct_iqr = flagged_iqr[flagged_iqr['Class'] == 1]
print(f"Correctly flagged fraud: {len(correct_iqr)}")
output = df[df['zscore_flag'] == 1][['Amount', 'Class', 'zscore']].copy()
output.columns = ['Amount', 'Actually_Fraud', 'Z_Score']
output = output.sort_values('Z_Score', ascending=False)
output.to_csv('flagged_transactions.csv', index=False)
print("\nResults saved to flagged_transactions.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df[df['Class'] == 0]['Amount'], bins=50, color='steelblue', alpha=0.7)
axes[0].set_title('Normal Transactions')
axes[0].set_xlabel('Amount')
axes[0].set_ylabel('Count')

axes[1].hist(df[df['Class'] == 1]['Amount'], bins=50, color='crimson', alpha=0.7)
axes[1].set_title('Fraudulent Transactions')
axes[1].set_xlabel('Amount')

plt.tight_layout()
plt.savefig('distribution.png')