import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('creditcard.csv')

print(f"Total transactions: {len(df)}")
print(f"Fraudulent transactions: {df['Class'].sum()}")
print(f"Fraud rate: {df['Class'].mean() * 100:.3f}%")
print(f"\nColumns: {list(df.columns)}")
print(f"\nAmount stats:")
print(df['Amount'].describe())
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df[df['Class'] == 0]['Amount'], bins=50, color='blue', alpha=0.7)
plt.title('Normal transactions')
plt.xlabel('Amount')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(df[df['Class'] == 1]['Amount'], bins=50, color='red', alpha=0.7)
plt.title('Fraudulent transactions')
plt.xlabel('Amount')

plt.tight_layout()
plt.show()