# 02_encoding.py
import pandas as pd
import os

OUTPUT_DIR = "../Dataset/Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Đọc file raw
df = pd.read_csv(os.path.join(OUTPUT_DIR, "df_raw_dashboard.csv"))

#%% 1. Mã hoá Likert
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

for col in likert_cols:
    df[col] = (
        df[col].astype(str).str.lower()
        .str.replace(r'[^a-z ]', '', regex=True).str.strip()
    )
    df[col + '_encoded'] = df[col].map(likert_map)
    missing = df[df[col + '_encoded'].isna()][col].unique()
    if len(missing) > 0:
        print(f"Cột {col} còn giá trị chưa mã hoá:", missing)

#%% 2. Mã hoá ordinal
map_wait = {'30 minutes': 1, '45 minutes': 2, '60 minutes': 3, 'more than 60 minutes': 4}
map_influence = {'no': 1, 'maybe': 2, 'yes': 3}
df['Maximum wait time_encoded'] = df['Maximum wait time'].astype(str).str.lower().map(map_wait)
df['Influence of rating_encoded'] = df['Influence of rating'].astype(str).str.lower().map(map_influence)

#%% 3. One-hot encoding
cat_cols = ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications',
            'Frequently used Medium', 'Frequently ordered Meal category', 'Perference']
cat_cols = [c for c in cat_cols if c in df.columns]
drop_cols = likert_cols + ['Maximum wait time', 'Influence of rating']

df_encoded = df.drop(columns=drop_cols, errors='ignore')
df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
df_encoded = df_encoded.replace({True: 1, False: 0}).infer_objects(copy=False)

encoded_path = os.path.join(OUTPUT_DIR, "df_encoded.csv")
df_encoded.to_csv(encoded_path, index=False)
print(f"Đã lưu: {encoded_path}")