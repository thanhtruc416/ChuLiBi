# ============================================
# Frame09_ExpectedLoss_PyCharm_FIXED_v2.py
# ============================================
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    roc_auc_score, r2_score, mean_squared_error, mean_absolute_error, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

# ============================================
# Global Settings
# ============================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

sns.set_style("whitegrid")
custom_palette = ['#644E94', '#9B86C2', '#E3D4E0']
bg_color = "#F9F7FB"
text_color = "#2E2E2E"

PATH = "../Dataset/Output/"
FILE_CLUSTER = os.path.join(PATH, "df_cluster_full.csv")
FILE_CHURN = os.path.join(PATH, "churn_predictions_preview.csv")
FILE_RAW= os.path.join(PATH, "df_raw_dashboard.csv")
df_raw= pd.read_csv(FILE_RAW)

# ============================================
# STEP 1. Load + Preprocess Data
# ============================================
def load_and_preprocess():
    df_cluster = pd.read_csv(FILE_CLUSTER)
    df_churn = pd.read_csv(FILE_CHURN)
    print(f"[INFO] Cluster: {df_cluster.shape} | Churn: {df_churn.shape}")

    # Handle missing values
    num_cols = df_cluster.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())

    cat_cols = df_cluster.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        df_cluster[col] = df_cluster[col].fillna(df_cluster[col].mode()[0])

    print("[INFO] Missing values handled successfully.")

    # Normalize cluster column
    cluster_col = [c for c in df_cluster.columns if 'cluster' in c.lower()]
    df_cluster = df_cluster.rename(columns={cluster_col[0]: 'cluster'})
    print("[INFO] Standardized cluster column → 'cluster'")

    # Merge
    df = pd.merge(df_churn, df_cluster, left_on="Customer_ID", right_on="CustomerID", how="left")
    if 'cluster_x' in df.columns and 'cluster_y' in df.columns:
        df = df.rename(columns={'cluster_y': 'cluster'}).drop(columns=['cluster_x'])
    elif 'cluster_y' in df.columns:
        df = df.rename(columns={'cluster_y': 'cluster'})
    elif 'cluster_x' in df.columns:
        df = df.rename(columns={'cluster_x': 'cluster'})

    df = df.sort_values("CustomerID").reset_index(drop=True)
    df_cluster = df_cluster.sort_values("CustomerID").reset_index(drop=True)
    return df, df_cluster

# ============================================
# STEP 2. Train Churn Classification
# ============================================
def train_churn_model(df, df_cluster):
    leak_vars = ["proba_churn", "proba_churn_model", "ExpectedLoss_pred"]
    leak_vars = [v for v in leak_vars if v in df_cluster.columns or v in df.columns]
    print(f"[INFO] Removing potential leakage vars: {leak_vars}")

    X_no_leak = df_cluster.drop(columns=[c for c in leak_vars if c in df_cluster.columns], errors='ignore')
    feature_cols = [c for c in X_no_leak.columns if c not in ["CustomerID", "cluster"]]
    X = X_no_leak[feature_cols]
    y = df["pred_churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2,
                                                        random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                    random_state=SEED, stratify=y_temp)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "Logistic": LogisticRegression(max_iter=1000, n_jobs=1, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=SEED),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=SEED, n_jobs=1)
    }

    results = []
    for name, model in models.items():
        auc_scores = []
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_va)[:, 1]
            auc_scores.append(roc_auc_score(y_va, y_pred_proba))
        results.append({"Model": name, "Mean_AUC": np.mean(auc_scores)})

    df_result = pd.DataFrame(results)
    print("\nKFold AUC Results:")
    print(df_result)

    best_model_name = df_result.loc[df_result["Mean_AUC"].idxmax(), "Model"]
    print(f"Best churn model: {best_model_name}")

    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    df["proba_churn_model"] = best_model.predict_proba(X)[:, 1]

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(6, 5), facecolor=bg_color)
    plt.plot(fpr, tpr, label=f'{best_model_name} (AUC={roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.3f})',
             color=custom_palette[0])
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate", color=text_color)
    plt.ylabel("True Positive Rate", color=text_color)
    plt.title("ROC Curve - Churn Model", color=text_color)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df, X_train  # trả về X_train thật

# ============================================
# STEP 3. Expected Loss Calculation
# ============================================
def compute_expected_loss(df, X_train):
    scaler = MinMaxScaler()
    scaler.fit(df.loc[X_train.index, ["Order Value"]])  # fit đúng train
    df["OrderValue_scaled"] = scaler.transform(df[["Order Value"]])
    df["ExpectedLoss_score"] = df["proba_churn_model"] * df["OrderValue_scaled"]
    print("[INFO] ExpectedLoss_score calculated successfully.")
    return df

# ============================================
# STEP 4. Dual Expected Loss Models
# ============================================
def dual_expected_loss(df, df_cluster, PATH="../Dataset/Output/"):
    sns.set_style("whitegrid")
    custom_palette = {
        "High Behavior + High Value": "#4B2E83",
        "High Behavior Only": "#7A5BA8",
        "High Value Only": "#B39CD0",
        "Low Risk": "#E3D4E0"
    }

    bg_color = "#F9F7FB"
    text_color = "#2E2E2E"
    plt.rcParams.update({
        "axes.facecolor": bg_color,
        "figure.facecolor": bg_color,
        "text.color": text_color,
        "axes.labelcolor": text_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "axes.edgecolor": "#DDDDDD"
    })

    print("[STEP] Building Dual Expected Loss models...")

    # --- FIX LEAKAGE ---
    drop_leak = ["ExpectedLoss_score", "OrderValue_scaled", "proba_churn_model"]
    df_model = df.drop(columns=[c for c in drop_leak if c in df.columns], errors='ignore')

    target_col = "ExpectedLoss_score"
    y_full = df[target_col]

    cols_full = [c for c in df_model.columns if c not in ["CustomerID", "Customer_ID", "cluster"]]
    cols_noorder = [c for c in cols_full if not any(x in c.lower() for x in ["order", "price", "value"])]

    X_full = df_model[cols_full]
    X_noorder = df_model[cols_noorder]

    def encode_df(df_in):
        df_out = df_in.copy()
        for c in df_out.columns:
            if df_out[c].dtype == "object" or df_out[c].dtype.name == "category":
                le = LabelEncoder()
                df_out[c] = le.fit_transform(df_out[c].astype(str))
        return df_out

    X_full = encode_df(X_full)
    X_noorder = encode_df(X_noorder)

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED)
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_noorder, y_full, test_size=0.2, random_state=SEED)

    rf_full = RandomForestRegressor(random_state=SEED, n_jobs=1)
    rf_noorder = RandomForestRegressor(random_state=SEED, n_jobs=1)
    rf_full.fit(Xf_train, yf_train)
    rf_noorder.fit(Xn_train, yn_train)

    y_pred_full = rf_full.predict(Xf_test)
    y_pred_noorder = rf_noorder.predict(Xn_test)

    def evaluate(y_true, y_pred, model_name):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"[INFO] {model_name} — R²={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")
        return r2, rmse, mae

    r2_full, rmse_full, mae_full = evaluate(yf_test, y_pred_full, "ExpectedLoss_full")
    r2_no, rmse_no, mae_no = evaluate(yn_test, y_pred_noorder, "ExpectedLoss_noOrder")
    corr = np.corrcoef(y_pred_full, y_pred_noorder)[0, 1]
    print(f"Correlation between Full vs No-Order predictions: {corr:.4f}")

    # --- Dual map ---
    df_dual_map = pd.DataFrame({
        "ExpectedLoss_full_pred": y_pred_full,
        "ExpectedLoss_noOrder_pred": y_pred_noorder
    }, index=Xf_test.index)
    df_dual_map["cluster"] = df_cluster.loc[df_dual_map.index, "cluster"].values

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_dual_map[["ExpectedLoss_full_pred", "ExpectedLoss_noOrder_pred"]]),
        columns=["ExpectedLoss_full_pred", "ExpectedLoss_noOrder_pred"],
        index=df_dual_map.index
    )
    df_scaled["cluster"] = df_dual_map["cluster"]
    df_scaled["risk_segment"] = np.select(
        [
            (df_scaled["ExpectedLoss_noOrder_pred"] >= 0.6) & (df_scaled["ExpectedLoss_full_pred"] >= 0.6),
            (df_scaled["ExpectedLoss_noOrder_pred"] >= 0.6),
            (df_scaled["ExpectedLoss_full_pred"] >= 0.6)
        ],
        ["High Behavior + High Value", "High Behavior Only", "High Value Only"],
        default="Low Risk"
    )

    # --- Scatter plot ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_scaled,
        x="ExpectedLoss_noOrder_pred",
        y="ExpectedLoss_full_pred",
        hue="risk_segment",
        palette=custom_palette,
        s=70, alpha=0.9, edgecolor="white", linewidth=0.6
    )
    plt.axvline(0.6, color='#C9BFE5', linestyle='--', lw=1)
    plt.axhline(0.6, color='#C9BFE5', linestyle='--', lw=1)
    plt.title("Dual Risk Map: Behavioral vs Financial Expected Loss", fontsize=13, weight='bold', color=text_color)
    plt.xlabel("Behavioral Risk (No-Order Model)", fontsize=11)
    plt.ylabel("Financial Risk (Full Model)", fontsize=11)
    plt.legend(title="Risk Segment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    seg_summary = df_scaled["risk_segment"].value_counts(normalize=True).mul(100).round(1)
    print("Tỷ lệ khách hàng theo nhóm rủi ro (%):")
    print(seg_summary)

    # --- Cluster heatmap ---
    plt.figure(figsize=(8, 5))
    heatmap_data = df_scaled.groupby("cluster")[["ExpectedLoss_full_pred", "ExpectedLoss_noOrder_pred"]].mean()
    cmap = sns.color_palette(list(custom_palette.values()), as_cmap=True)
    sns.heatmap(heatmap_data, annot=True, cmap=cmap)
    plt.title("Mean Expected Loss per Cluster", fontsize=13, weight='bold', color=text_color)
    plt.tight_layout()
    plt.show()

    # === SAVE OUTPUTS ===
    os.makedirs(PATH, exist_ok=True)

    # Cluster-level summary
    heatmap_data.to_csv(os.path.join(PATH, "expected_loss_by_customer.csv"), index=False)

    # === Gắn dự đoán ExpectedLoss_pred ===
    df_pred_full = df.copy()
    df_pred_full["ExpectedLoss_pred"] = np.nan
    df_pred_full.loc[Xf_test.index, "ExpectedLoss_pred"] = y_pred_full

    cols_top = [
        c for c in [
            "CustomerID", "cluster", "Order Value",
            "proba_churn_model", "OrderValue_scaled",
            "ExpectedLoss_score", "ExpectedLoss_pred"
        ] if c in df_pred_full.columns
    ]

    # --- Lấy dữ liệu gốc để có "Order Value" thật ---
    if "Order Value" in df_raw.columns:
        df_pred_full = df_pred_full.drop(columns=["Order Value"], errors="ignore")
        df_pred_full = df_pred_full.merge(
            df_raw[["CustomerID", "Order Value"]],
            on="CustomerID", how="left"
        )

    # --- Tính ExpectedLoss thực tế (Order Value × proba_churn_model) ---
    df_pred_full["ExpectedLoss_real"] = (
            df_pred_full["Order Value"] * df_pred_full["proba_churn_model"]
    )

    # --- Chọn và sắp top theo ExpectedLoss_real ---
    df_top50 = df_pred_full[cols_top + ["ExpectedLoss_real"]].dropna(subset=["ExpectedLoss_real"])
    df_top50 = df_top50.sort_values("ExpectedLoss_real", ascending=False).head(50)
    df_top50.to_csv(os.path.join(PATH, "expected_loss_top50.csv"), index=False)

    # --- Chuẩn bị file hiển thị cho UI ---
    df_display = df_top50.copy()

    # Chuyển các xác suất sang %
    df_display["Churn Probability (%)"] = (df_display["proba_churn_model"] * 100).round(1)
    df_display["Expected Loss Score (%)"] = (df_display["ExpectedLoss_score"] * 100).round(1)

    # ExpectedLoss_real = thiệt hại kỳ vọng thật (theo thang 1–3)
    df_display["Expected Loss (Value 1–3)"] = df_display["ExpectedLoss_real"].round(2)

    # Giữ cột cần hiển thị
    df_display = df_display[[
        "CustomerID", "cluster", "Order Value",
        "Churn Probability (%)", "Expected Loss Score (%)",
        "Expected Loss (Value 1–3)"
    ]]

    # Đổi tên thân thiện
    df_display = df_display.rename(columns={
        "CustomerID": "Customer ID",
        "cluster": "Cluster",
        "Order Value": "Order Value (1–3)"
    })

    # Xuất file
    df_display.to_csv(os.path.join(PATH, "expected_loss_top50_display.csv"), index=False)

    print(f"[INFO] Saved: {PATH}/expected_loss_by_customer.csv")
    print(f"[INFO] Saved: {PATH}/expected_loss_top50.csv")
    print(f"[INFO] Saved readable version for UI → expected_loss_top50_display.csv")
    print("[DONE] Dual Expected Loss analysis complete.")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    df, df_cluster = load_and_preprocess()
    df, X_train = train_churn_model(df, df_cluster)
    df = compute_expected_loss(df, X_train)
    dual_expected_loss(df, df_cluster)
