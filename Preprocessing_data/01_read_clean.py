# 01_read_clean.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn
INPUT_PATH = "../Dataset/Output/Customer_data_with_ID.csv"
OUTPUT_DIR = "../Dataset/Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#%% 1. Đọc dữ liệu
df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip().str.replace('\xa0', ' ', regex=False)
if 'Restaurnat Rating' in df.columns:
    df.rename(columns={'Restaurnat Rating': 'Restaurant Rating'}, inplace=True)

print("Dữ liệu ban đầu:", df.shape)

#%% 2. Xử lý missing
numeric_cols = [
    'Age', 'Family size', 'Restaurant Rating', 'Delivery Rating',
    'No. of orders placed', 'Delivery Time', 'Order Value'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

cat_cols = [c for c in df.columns if c not in numeric_cols + ['CustomerID']]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

#%% 3. Xử lý outlier
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

plt.figure(figsize=(12, 5))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot sau khi xử lý outlier")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% 4. Lưu dữ liệu gốc
df_raw_path = os.path.join(OUTPUT_DIR, "df_raw_dashboard.csv")
df.to_csv(df_raw_path, index=False)
print(f"Đã lưu: {df_raw_path}")