# File: Frame09_ExpectedLoss_FIXED_split.py
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

# ---------- GLOBAL ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

sns.set_style("whitegrid")
bg_color = "#F9F7FB"
text_color = "#2E2E2E"
PATH = "../Dataset/Output/"
FILE_CLUSTER = os.path.join(PATH, "df_cluster_full.csv")
FILE_CHURN = os.path.join(PATH, "churn_predictions_preview.csv")
FILE_RAW = os.path.join(PATH, "df_raw_dashboard.csv")

# load df_raw here (used by display); if not exists will raise early
if os.path.exists(FILE_RAW):
    df_raw = pd.read_csv(FILE_RAW)
else:
    df_raw = None

# ---------- STEP 1: load & preprocess ----------
def load_and_preprocess():
    df_cluster = pd.read_csv(FILE_CLUSTER)
    df_churn = pd.read_csv(FILE_CHURN)
    print(f"[INFO] Cluster: {df_cluster.shape} | Churn: {df_churn.shape}")

    # fill missing in cluster
    num_cols = df_cluster.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())
    cat_cols = df_cluster.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        df_cluster[col] = df_cluster[col].fillna(df_cluster[col].mode()[0])

    # normalize cluster column name
    cluster_col = [c for c in df_cluster.columns if 'cluster' in c.lower()]
    if cluster_col:
        df_cluster = df_cluster.rename(columns={cluster_col[0]: 'cluster'})

    # merge churn + cluster
    df = pd.merge(df_churn, df_cluster, left_on="Customer_ID", right_on="CustomerID", how="left")
    # normalize cluster column if duplicated
    if 'cluster_x' in df.columns and 'cluster_y' in df.columns:
        df = df.rename(columns={'cluster_y': 'cluster'}).drop(columns=['cluster_x'])
    elif 'cluster_y' in df.columns:
        df = df.rename(columns={'cluster_y': 'cluster'})
    elif 'cluster_x' in df.columns:
        df = df.rename(columns={'cluster_x': 'cluster'})

    df = df.sort_values("CustomerID").reset_index(drop=True)
    df_cluster = df_cluster.sort_values("CustomerID").reset_index(drop=True)
    return df, df_cluster

# ---------- STEP 2: train churn ----------
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
    # predict proba for all rows in X (original index from df_cluster)
    df["proba_churn_model"] = best_model.predict_proba(X)[:, 1]

    # ROC plot (only as diagnostic)
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(6, 5), facecolor=bg_color)
    plt.plot(fpr, tpr, label=f'{best_model_name} (AUC={roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title("ROC Curve - Churn Model")
    plt.tight_layout()
    plt.show()

    return df, X_train

# ---------- STEP 3: compute expected loss (score) ----------
def compute_expected_loss(df, X_train):
    scaler = MinMaxScaler()
    # fit scaler on training rows' Order Value to mimic your original approach
    scaler.fit(df.loc[X_train.index, ["Order Value"]])
    df["OrderValue_scaled"] = scaler.transform(df[["Order Value"]])
    df["ExpectedLoss_score"] = df["proba_churn_model"] * df["OrderValue_scaled"]
    print("[INFO] ExpectedLoss_score calculated successfully.")
    return df

# ---------- STEP 4A: train dual expected-loss models (returns train/test splits too) ----------
def dual_expected_loss_train(df, df_cluster, SEED=SEED):
    print("[STEP] Training Dual Expected Loss models...")

    # prepare df_model with leakage columns dropped
    drop_leak = ["ExpectedLoss_score", "OrderValue_scaled", "proba_churn_model"]
    df_model = df.drop(columns=[c for c in drop_leak if c in df.columns], errors='ignore')

    target_col = "ExpectedLoss_score"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in df")

    y_full = df[target_col]

    cols_full = [c for c in df_model.columns if c not in ["CustomerID", "Customer_ID", "cluster"]]
    cols_noorder = [c for c in cols_full if not any(x in c.lower() for x in ["order", "price", "value"])]

    X_full = df_model[cols_full].copy()
    X_noorder = df_model[cols_noorder].copy()

    # encode categoricals
    def encode_df(df_in):
        df_out = df_in.copy()
        for c in df_out.columns:
            if df_out[c].dtype == "object" or df_out[c].dtype.name == "category":
                le = LabelEncoder()
                df_out[c] = le.fit_transform(df_out[c].astype(str))
        return df_out

    X_full = encode_df(X_full)
    X_noorder = encode_df(X_noorder)

    # split (keep index)
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED)
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_noorder, y_full, test_size=0.2, random_state=SEED)

    rf_full = RandomForestRegressor(random_state=SEED, n_jobs=1)
    rf_noorder = RandomForestRegressor(random_state=SEED, n_jobs=1)
    rf_full.fit(Xf_train, yf_train)
    rf_noorder.fit(Xn_train, yn_train)

    # evaluate (fixed RMSE)
    def evaluate(y_true, y_pred, name):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"[INFO] {name} — R²={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")
        return r2, rmse, mae

    evaluate(yf_test, rf_full.predict(Xf_test), "ExpectedLoss_full")
    evaluate(yn_test, rf_noorder.predict(Xn_test), "ExpectedLoss_noOrder")

    return {
        "rf_full": rf_full,
        "rf_noorder": rf_noorder,
        "X_full": X_full,
        "X_noorder": X_noorder,
        "Xf_test": Xf_test,
        "Xn_test": Xn_test,
        "yf_test": yf_test,
        "yn_test": yn_test
    }

# ---------- STEP 4B: plot dual map and cluster heatmap using TEST predictions (to match original) ----------
def dual_expected_loss_plot(train_out, df_cluster, save_path=PATH):
    rf_full = train_out["rf_full"]
    rf_noorder = train_out["rf_noorder"]
    Xf_test = train_out["Xf_test"]
    Xn_test = train_out["Xn_test"]

    # predictions on test (these are what original dual map used)
    y_pred_full = rf_full.predict(Xf_test)
    y_pred_noorder = rf_noorder.predict(Xn_test)

    # build df_dual indexed by Xf_test.index (use Xf_test.index for both preds — note original used that index)
    # We'll align by index length: use Xf_test.index for full preds, and for noorder we will align by position
    df_dual_map = pd.DataFrame({
        "ExpectedLoss_full_pred": y_pred_full,
        "ExpectedLoss_noOrder_pred": y_pred_noorder
    }, index=Xf_test.index)

    # map cluster from df_cluster using same index (if same index/ordering as df originally)
    try:
        df_dual_map["cluster"] = df_cluster.loc[df_dual_map.index, "cluster"].values
    except Exception:
        # fallback: take first N clusters (positional)
        df_dual_map["cluster"] = df_cluster["cluster"].values[:len(df_dual_map)]

    # scale to 0-1 for plotting
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_dual_map[["ExpectedLoss_full_pred", "ExpectedLoss_noOrder_pred"]])
    df_scaled = pd.DataFrame(scaled, columns=["ExpectedLoss_full_pred", "ExpectedLoss_noOrder_pred"], index=df_dual_map.index)
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

    # scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_scaled,
        x="ExpectedLoss_noOrder_pred", y="ExpectedLoss_full_pred",
        hue="risk_segment", palette={
            "High Behavior + High Value": "#4B2E83",
            "High Behavior Only": "#7A5BA8",
            "High Value Only": "#B39CD0",
            "Low Risk": "#E3D4E0"
        },
        s=70, alpha=0.9, edgecolor="white", linewidth=0.6
    )
    plt.axvline(0.6, color='#C9BFE5', linestyle='--', lw=1)
    plt.axhline(0.6, color='#C9BFE5', linestyle='--', lw=1)
    plt.title("Dual Risk Map: Behavioral vs Financial Expected Loss")
    plt.tight_layout()
    plt.show()

    # heatmap (mean per cluster)
    heatmap_data = df_scaled.groupby("cluster")[["ExpectedLoss_full_pred", "ExpectedLoss_noOrder_pred"]].mean()
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, cmap="Purples")
    plt.title("Mean Expected Loss per Cluster")
    plt.tight_layout()
    plt.show()

    # save heatmap summary for UI
    os.makedirs(save_path, exist_ok=True)
    heatmap_data.to_csv(os.path.join(save_path, "expected_loss_cluster_summary.csv"), index=True)
    print(f"[INFO] Saved cluster summary → {save_path}/expected_loss_cluster_summary.csv")

    return df_scaled, heatmap_data

# ---------- STEP 4C: build display CSV for ALL customers (predict on X_full) ----------
def dual_expected_loss_display(df, train_out, df_raw=None, save_path=PATH):
    rf_full = train_out["rf_full"]
    X_full = train_out["X_full"]   # indexed = same as df_model rows

    # predict on full (index aligned with df_model which came from df)
    y_pred_all = rf_full.predict(X_full)

    # make a copy of original df (which has CustomerID, proba_churn_model, Order Value, ExpectedLoss_score)
    df_pred = df.copy()
    df_pred["ExpectedLoss_pred"] = np.nan
    # align by index: X_full.index equals df_model.index which equals df.index (we kept original df_model derived from df)
    df_pred.loc[X_full.index, "ExpectedLoss_pred"] = y_pred_all

    # if df_raw provided, restore original "Order Value" from df_raw by CustomerID
    if df_raw is not None and "CustomerID" in df_pred.columns and "CustomerID" in df_raw.columns and "Order Value" in df_raw.columns:
        # drop current Order Value and merge
        df_pred = df_pred.drop(columns=["Order Value"], errors="ignore")
        df_pred = df_pred.merge(df_raw[["CustomerID", "Order Value"]], on="CustomerID", how="left")

    # ExpectedLoss_real: real expected loss value = Order Value * churn probability (both numeric)
    df_pred["ExpectedLoss_real"] = df_pred["Order Value"] * df_pred["proba_churn_model"]

    # output full numeric csv for UI/backend
    cols_out = [
        "CustomerID", "cluster", "Order Value", "proba_churn_model",
        "OrderValue_scaled", "ExpectedLoss_score", "ExpectedLoss_pred", "ExpectedLoss_real"
    ]
    cols_out = [c for c in cols_out if c in df_pred.columns]
    os.makedirs(save_path, exist_ok=True)
    df_pred[cols_out].to_csv(os.path.join(save_path, "expected_loss_by_customer.csv"), index=False)
    print(f"[INFO] Saved full → {save_path}/expected_loss_by_customer.csv")

    # build display (readable) - percentages where appropriate
    df_display = pd.DataFrame({
        "Customer ID": df_pred["CustomerID"],
        "Cluster": df_pred["cluster"],
        "Order Value (1–3)": df_pred["Order Value"],
        "Churn Probability (%)": (df_pred["proba_churn_model"] * 100).round(1),
        "Expected Loss Score (%)": (df_pred["ExpectedLoss_pred"] * 100).round(1),
        "Expected Loss (%)": ((df_pred["ExpectedLoss_real"] / 3) * 100).round(1)
    })
    df_display.to_csv(os.path.join(save_path, "expected_loss_by_customer_display.csv"), index=False)
    print(f"[INFO] Saved display → {save_path}/expected_loss_by_customer_display.csv")
    return df_display

# ---------- STEP 4D: simple search function ----------
def dual_expected_loss_search(df_display):
    try:
        q = input("Nhập CustomerID (hoặc Cluster number) để tra cứu (Enter để bỏ qua): ").strip()
        if not q:
            print("Bỏ qua tra cứu.")
            return
        if q.isdigit():
            # treat as cluster filter
            q_int = int(q)
            match = df_display[df_display["Cluster"] == q_int]
        else:
            match = df_display[df_display["Customer ID"].str.upper() == q.upper()]
        if match.empty:
            print("Không tìm thấy kết quả cho:", q)
        else:
            print(match.to_string(index=False))
    except EOFError:
        print("Môi trường không hỗ trợ input(); bỏ qua tra cứu.")
def filter_by_cluster(df_display, cluster_id):
    """
    Lọc toàn bộ khách hàng thuộc cluster cụ thể.
    Dùng cho dropdown UI, KHÔNG yêu cầu input().
    """
    try:
        cluster_id = int(cluster_id)
        match = df_display[df_display["Cluster"] == cluster_id]
        if match.empty:
            print(f"Không có khách hàng nào trong Cluster {cluster_id}.")
        else:
            print(f"Danh sách khách hàng thuộc Cluster {cluster_id}:")
            print(match.to_string(index=False))
    except ValueError:
        print("Cluster ID phải là số nguyên.")

# ---------- MAIN ----------
if __name__ == "__main__":
    df, df_cluster = load_and_preprocess()
    df, X_train = train_churn_model(df, df_cluster)
    df = compute_expected_loss(df, X_train)

    train_out = dual_expected_loss_train(df, df_cluster)
    df_scaled, heatmap = dual_expected_loss_plot(train_out, df_cluster)
    df_display = dual_expected_loss_display(df, train_out, df_raw)
    dual_expected_loss_search(df_display)
    print("[DONE] Dual Expected Loss pipeline finished.")
    filter_by_cluster(df_display, 2)

"""
CUS000019 → Churn Probability = 97.3%, Expected Loss Score = 0.0%, Expected Loss = 32.4%

+) Churn Probability:Cao → khách hàng này rất có khả năng rời bỏ

+) Expected Loss Score = 0.0% → các đặc trưng hành vi hoặc giá trị tài chính của khách hàng này không dẫn tới tổn thất lớn — ví dụ đơn hàng nhỏ, cụm cluster “rủi ro thấp về giá trị”.

+) Expected Loss = 32.4% → Được tính thủ công từ Order Value * proba_churn_model — nếu order chỉ là 1 (nhỏ nhất), thì thiệt hại tính theo phần trăm cũng không cao.

==> Khách này rất dễ rời bỏ, nhưng tổn thất tài chính thấp (vì order value thấp)
"""