import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# --- PATH ---
DATA_DIR = "./Data_output"
os.makedirs(DATA_DIR, exist_ok=True)

df_scaled_model = pd.read_csv(os.path.join(DATA_DIR, "df_scaled_model.csv"))
df_raw = pd.read_csv(os.path.join(DATA_DIR, "df_raw_dashboard.csv"))

# Giữ lại CustomerID
ids = df_scaled_model['CustomerID']
df_cluster = df_scaled_model.drop(columns=['CustomerID'], errors='ignore')

print(f"Dữ liệu scale dùng để phân cụm: {df_cluster.shape}")

#%% - 2. PCA CHO NHÓM BIẾN LIKERT
groupA = ['Ease and convenient_encoded', 'Self Cooking_encoded', 'Health Concern_encoded']
groupB = ['Poor Hygiene_encoded', 'Bad past experience_encoded', 'Late Delivery_encoded']
groupC = ['More Offers and Discount_encoded', 'Influence of rating_encoded']

# Kiểm tra đủ cột
for group_name, group_cols in zip(['A', 'B', 'C'], [groupA, groupB, groupC]):
    missing = [c for c in group_cols if c not in df_cluster.columns]
    if missing:
        print(f"Thiếu cột ở nhóm {group_name}: {missing}")

# PCA n=1 cho từng nhóm
df_cluster['pca_convenience'] = PCA(n_components=1, random_state=42).fit_transform(df_cluster[groupA])
df_cluster['pca_service_issue'] = PCA(n_components=1, random_state=42).fit_transform(df_cluster[groupB])
df_cluster['pca_deal_sensitive'] = PCA(n_components=1, random_state=42).fit_transform(df_cluster[groupC])

print("Đã tạo 3 thành phần PCA đại diện cho nhóm Likert.")

#%% - 3. CHỌN BIẾN PHÂN CỤM
features = [
    'Age', 'Family size', 'Restaurant Rating', 'Delivery Rating',
    'No. of orders placed', 'Delivery Time', 'Order Value',
    'pca_convenience', 'pca_service_issue', 'pca_deal_sensitive'
]
X_cluster = df_cluster[features].copy()
X_scaled = X_cluster.values

#%% - 4. CHỌN SỐ CỤM HỢP LÝ (Elbow & Silhouette)
inertias, sil_scores = [], []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# --- Vẽ biểu đồ Elbow & Silhouette ---
plt.figure(figsize=(7,4))
plt.plot(K, inertias, 'bo-', color='#644E94')
plt.xlabel('Số cụm (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method — chọn số cụm hợp lý')
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
plt.plot(K, sil_scores, 'ro-', color='#C6ABC5')
plt.xlabel('Số cụm (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method — đánh giá độ tách cụm')
plt.grid(True)
plt.show()

for k in [2, 3, 4]:
    print(f"k={k}, Silhouette={silhouette_score(X_scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)):.3f}")

#%% - 5. VẼ PHÂN CỤM 2D (PCA)
for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    plt.figure(figsize=(6,5))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=labels,
        palette=['#644E94', '#C6ABC5', '#9282AA', '#FAE4F2'][:k], s=50, edgecolor='black'
    )
    plt.title(f'Phân cụm PCA 2D với K={k}')
    plt.xlabel('Thành phần chính 1')
    plt.ylabel('Thành phần chính 2')
    plt.legend(title='Cụm')
    plt.grid(True)
    plt.show()

#%% - 6. PHÂN CỤM CHÍNH THỨC VỚI K=3
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans_final.fit_predict(X_scaled)
df_cluster['CustomerID'] = ids

print("Đã gán nhãn cụm thành công!")

#%% - 7. PROFILE CỤM
cluster_profile = (
    df_cluster.groupby('cluster')[features]
    .mean().round(3)
)
cluster_profile['Count'] = df_cluster['cluster'].value_counts().sort_index().values

print("Profile cụm:")
print(cluster_profile)

# Lưu
df_cluster.to_csv(os.path.join(DATA_DIR, "df_cluster_full.csv"), index=False)
cluster_profile.to_csv(os.path.join(DATA_DIR, "cluster_profile_scaled.csv"))
print("Đã lưu df_cluster_full.csv + cluster_profile_scaled.csv")

#%% - 8. MÔ TẢ CỤM (RAW + PCA)
df_merged = pd.merge(df_raw, df_cluster[['CustomerID', 'cluster']], on='CustomerID', how='left')

numeric_features = ['Age', 'Family size', 'Restaurant Rating', 'Delivery Rating',
                    'No. of orders placed', 'Delivery Time', 'Order Value']
pca_features = ['pca_convenience', 'pca_service_issue', 'pca_deal_sensitive']

profile_real = df_merged.groupby('cluster')[numeric_features].mean().round(2)
pca_means = df_cluster.groupby('cluster')[pca_features].mean().round(2)
profile_real = pd.concat([profile_real, pca_means], axis=1)
profile_real['Count'] = df_cluster['cluster'].value_counts().sort_index().values

# Gắn nhãn PCA
def label_pca(v, thresholds=(-0.3, 0.3)):
    return "Thấp" if v <= thresholds[0] else "Cao" if v >= thresholds[1] else "Trung bình"

desc_profile = profile_real.copy()
for col in pca_features:
    desc_profile[col] = profile_real[col].apply(label_pca)

rename_map = {
    'pca_convenience': 'Mức độ coi trọng sự tiện lợi',
    'pca_service_issue': 'Vấn đề dịch vụ',
    'pca_deal_sensitive': 'Nhạy cảm ưu đãi/đánh giá'
}
desc_profile.rename(columns=rename_map, inplace=True)

desc_profile.to_csv(
    os.path.join(DATA_DIR, "cluster_characteristics_descriptive.csv"),
    index=False,
    encoding="utf-8-sig"   #đảm bảo tiếng Việt không lỗi khi mở bằng Excel
)
print("Đã lưu cluster_characteristics_descriptive.csv")

#%% - 9. PIE CHART
summary = df_cluster['cluster'].value_counts().sort_index().reset_index()
summary.columns = ['Cluster', 'Số lượng']

plt.figure(figsize=(8,8))
colors = ['#FAE4F2', '#C6ABC5', '#644E94']
wedges, texts, autotexts = plt.pie(
    summary['Số lượng'], labels=None,
    autopct=lambda pct: '{:.1f}%\n({:.0f})'.format(pct, pct/100.*summary['Số lượng'].sum()),
    startangle=140, colors=colors, textprops={'fontsize':11, 'fontweight':'bold'}
)
for t in autotexts: t.set_color('black')
plt.title('Phân bố khách hàng theo cụm (K=3)', fontsize=14, pad=20)
plt.legend(title="Cụm", labels=[f"Cluster {i}" for i in summary['Cluster']],
           loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()
