# eda_preprocessing.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

# Path
INPUT_PATH = "./data/Customer_data_with_ID.csv"
OUTPUT_DIR = "./Data_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#%% - 1. Đọc dữ liệu
""" 1. Đọc dữ liệu """
df = pd.read_csv(INPUT_PATH)

# Làm sạch tên cột
df.columns = df.columns.str.strip().str.replace('\xa0', ' ', regex=False)

# Nếu có lỗi chính tả cũ, sửa
if 'Restaurnat Rating' in df.columns:
    df.rename(columns={'Restaurnat Rating': 'Restaurant Rating'}, inplace=True)

print("Dữ liệu ban đầu:", df.shape)

#%% - 2. Xử lý missing
""" 2. Xử lý missing """
# Danh sách numeric ban đầu
numeric_cols = [
    'Age', 'Family size', 'Restaurant Rating', 'Delivery Rating',
    'No. of orders placed', 'Delivery Time', 'Order Value'
]

# Bảo đảm convert sang numeric (lỗi thì NaN)
# Điền missing cho numeric bằng mean
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

# Với categorical (những cột còn lại, TRỪ CustomerID)
cat_cols = [c for c in df.columns if c not in numeric_cols + ['CustomerID']]
# Điền mode cho categorical
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

#%% - 3. Xử lý outlier
""" 3. Xử lý outliers bằng IQR """
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
print("Outliers clipped.")

# Vẽ boxplot trên df (dữ liệu đã clean) để kiểm tra
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot sau khi xử lý outlier")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% - # 4. EDA cơ bản
""" 4. EDA: Thống kê, mô tả, tương quan (numeric gốc) """
print(df[numeric_cols].describe())
# Phân phối các biến số gốc (trước chuẩn hoá)
plt.figure(figsize=(14, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Phân phối {col}')
plt.tight_layout()
plt.show()

# Ma trận tương quan chỉ cho các biến định lượng gốc
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Ma trận tương quan (numeric)')
plt.show()

#%% - 5. Tạo bản raw (dashboard)
""" 5. Lưu dữ liệu gốc (dashboard) """
df_raw_path = os.path.join(OUTPUT_DIR, "df_raw_dashboard.csv")
df.to_csv(df_raw_path, index=False)
print(f"Đã lưu file raw: {df_raw_path}")

#%% - 6. Mã hóa ordinal cho các cột ý kiến (Likert)
""" 6. Mã hóa Likert """
# Map chuẩn hóa (đưa về lowercase luôn)
likert_map = {
    'strongly disagree': 1,
    'disagree': 2,
    'neutral': 3,
    'agree': 4,
    'strongly agree': 5
}
likert_cols = [
    'Ease and convenient', 'Self Cooking', 'Health Concern',
    'Late Delivery', 'Poor Hygiene', 'Bad past experience',
    'More Offers and Discount'
]

# Hạ hết về lowercase, bỏ ký tự đặc biệt, khoảng trắng lạ
for col in likert_cols:
    df[col] = (
        df[col].astype(str).str.lower()
        .str.replace(r'[^a-z ]', '', regex=True).str.strip()
    )
    df[col + '_encoded'] = df[col].map(likert_map)

# Báo lỗi nếu còn giá trị chưa mã hoá
    missing = df[df[col + '_encoded'].isna()][col].unique()
    if len(missing) > 0:
        print(f"Cột {col} còn giá trị chưa mã hoá:", missing)
    else:
        print(f"{col} đã mã hoá xong")
#%% - 7. Mã hoá ordinal
"""# 7. Mã hoá ordinal """
map_wait = {'30 minutes': 1, '45 minutes': 2, '60 minutes': 3, 'more than 60 minutes': 4}
map_influence = {'no': 1, 'maybe': 2, 'yes': 3}
df['Maximum wait time_encoded'] = df['Maximum wait time'].astype(str).str.lower().map(map_wait)
df['Influence of rating_encoded'] = df['Influence of rating'].astype(str).str.lower().map(map_influence)

#%% - 8. Dummy encoding cho biến phân loại
""" 8. One-hot encoding """
cat_cols = ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications',
            'Frequently used Medium', 'Frequently ordered Meal category', 'Perference']
cat_cols = [c for c in cat_cols if c in df.columns]

#Giữ các cột Likert đã mã hoá (đã có *_encoded)
# nhưng loại bỏ bản text gốc trước khi one-hot
drop_cols = likert_cols + ['Maximum wait time', 'Influence of rating']
df_encoded = df.drop(columns=drop_cols, errors='ignore')

#One-hot encode các biến phân loại
df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

#Đảm bảo kiểu dữ liệu boolean → int
df_encoded = df_encoded.replace({True: 1, False: 0})

encoded_path = os.path.join(OUTPUT_DIR, "df_encoded.csv")
df_encoded.to_csv(encoded_path, index=False)
print(f"Đã lưu df_encoded.csv ({df_encoded.shape}) — chỉ gồm số, sẵn sàng để scale")

#%% - 9. Kiểm tra VIF
""" 9. Kiểm tra tương quan & VIF """
numeric_for_vif = numeric_cols + [c + '_encoded' for c in likert_cols] + [
    'Maximum wait time_encoded', 'Influence of rating_encoded'
]

# Chỉ giữ cột có trong dataframe
numeric_for_vif = [c for c in numeric_for_vif if c in df_encoded.columns]
X = df_encoded[numeric_for_vif].dropna()
vif = pd.DataFrame({
    "Variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("Bảng VIF:")
print(vif.sort_values(by="VIF", ascending=False))

#%% - 10. Scale và tạo file chỉ chứa numeric
""" 10. Scale dữ liệu (đảm bảo chỉ gồm cột numeric để mô hình/cluster dùng) """
scaler = StandardScaler()

# Chỉ scale trên các cột numeric (sau khi encode)
df_scaled_numeric = df_encoded.copy()
df_scaled_numeric[numeric_for_vif] = scaler.fit_transform(df_scaled_numeric[numeric_for_vif])

# Lưu scaler
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# Chỉ giữ lại CustomerID + các cột numeric & encoded
cols_for_model = ['CustomerID'] + numeric_for_vif + [
    c for c in df_encoded.columns if any(prefix in c for prefix in [
        'Gender_', 'Marital Status_', 'Occupation_',
        'Educational Qualifications_', 'Frequently used Medium_',
        'Frequently ordered Meal category_', 'Perference_'
    ])
]

df_scaled_model = df_scaled_numeric[cols_for_model]

# Lưu lại
scaled_path = os.path.join(OUTPUT_DIR, "df_scaled_model.csv")
df_scaled_model.to_csv(scaled_path, index=False)

print(f"Đã lưu df_scaled_model.csv ({df_scaled_model.shape}), chỉ chứa biến numeric/encoded")
