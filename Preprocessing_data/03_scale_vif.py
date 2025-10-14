# 03_scale_vif.py
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

OUTPUT_DIR = "../Dataset/Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_encoded = pd.read_csv(os.path.join(OUTPUT_DIR, "df_encoded.csv"))

#%% 1. Xác định cột numeric cho VIF
numeric_cols = [
    'Age', 'Family size', 'Restaurant Rating', 'Delivery Rating',
    'No. of orders placed', 'Delivery Time', 'Order Value'
]
likert_cols = [
    'Ease and convenient', 'Self Cooking', 'Health Concern',
    'Late Delivery', 'Poor Hygiene', 'Bad past experience',
    'More Offers and Discount'
]
numeric_for_vif = numeric_cols + [c + '_encoded' for c in likert_cols] + [
    'Maximum wait time_encoded', 'Influence of rating_encoded'
]
numeric_for_vif = [c for c in numeric_for_vif if c in df_encoded.columns]

#%% 2. Kiểm tra VIF
X = df_encoded[numeric_for_vif].dropna()
vif = pd.DataFrame({
    "Variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("=== Bảng VIF ===")
print(vif.sort_values(by="VIF", ascending=False))

#%% 3. Scale dữ liệu
scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[numeric_for_vif] = scaler.fit_transform(df_scaled[numeric_for_vif])
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

#%% 4. Chọn cột model
cols_for_model = ['CustomerID'] + numeric_for_vif + [
    c for c in df_encoded.columns if any(prefix in c for prefix in [
        'Gender_', 'Marital Status_', 'Occupation_',
        'Educational Qualifications_', 'Frequently used Medium_',
        'Frequently ordered Meal category_', 'Perference_'
    ])
]
df_scaled_model = df_scaled[cols_for_model]

scaled_path = os.path.join(OUTPUT_DIR, "df_scaled_model.csv")
df_scaled_model.to_csv(scaled_path, index=False)
print(f"Đã lưu: {scaled_path}")