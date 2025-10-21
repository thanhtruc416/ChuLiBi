import os
import warnings

from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, precision_score, recall_score, brier_score_loss,
                             precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Optional imports for SHAP
try:
    import shap
except Exception:
    shap = None

RND = 42  # cố định seed cho reproducibility


# ============== DATA LOADING FUNCTIONS FOR UI ==============
def get_churn_data(input_path: str = None, output_dir: str = None):
    """
    Load và preprocess data for UI.
    Returns: dict with all necessary data for Frame08 UI
    """
    from pathlib import Path

    if input_path is None:
        ROOT = Path(__file__).resolve().parents[1]
        input_path = str(ROOT / "Dataset" / "Output" / "df_cluster_full.csv")

    if output_dir is None:
        ROOT = Path(__file__).resolve().parents[1]
        output_dir = str(ROOT / "Dataset" / "Output")

    try:
        # Load data
        df = load_data(input_path)
        df = create_proxy_churn(df)

        # Detect leakage
        leak_cols = detect_leakage(df)

        # Prepare core data
        df_core = prepare_core_df(df, leak_cols=leak_cols)

        # Preprocess
        X, y, scaler = preprocess(df_core)

        # Load model
        model_path = os.path.join(output_dir, 'best_churn_model.pkl')
        bundle = None
        if os.path.exists(model_path):
            bundle = joblib.load(model_path)
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"✗ Model not found at {model_path}")

        # Load feature importance
        fi_path = os.path.join(output_dir, 'feature_importance.csv')
        feature_importance = None
        if os.path.exists(fi_path):
            feature_importance = pd.read_csv(fi_path)
            print("✓ Loaded feature importance")

        # Calculate churn by segment
        churn_by_seg = churn_by_segment_data(df_core, df)

        # Get predictions if model available
        df_result = None
        if bundle:
            df_result = make_prediction_table(df, X, bundle)

        # Get REAL evaluation metrics from model comparison
        # Try to load saved comparison results, or run quick comparison
        eval_metrics = None
        eval_path = os.path.join(output_dir, 'model_comparison.csv')

        if os.path.exists(eval_path):
            # Load saved comparison results
            eval_metrics = pd.read_csv(eval_path)
            print("✓ Loaded saved model comparison results")
        else:
            # Run quick comparison to get real metrics
            print("Running quick model comparison to get real metrics...")
            eval_table, results_dict = compare_models(X, y)

            # Rename columns to match UI expectations and select needed columns
            eval_metrics = eval_table.copy()
            eval_metrics = eval_metrics[['Model', 'AUC', 'F1', 'Precision', 'Recall', 'Accuracy']].copy()

            # Save for future use
            eval_metrics.to_csv(eval_path, index=False)
            print(f"✓ Saved comparison results to {eval_path}")

        # Determine best model from real results
        best_model_name = eval_metrics.iloc[0]['Model'] if not eval_metrics.empty else 'XGBoost'

        return {
            'df': df,
            'df_core': df_core,
            'X': X,
            'y': y,
            'bundle': bundle,
            'feature_importance': feature_importance,
            'churn_by_seg': churn_by_seg,
            'df_result': df_result,
            'eval_metrics': eval_metrics,
            'avg_churn': df_core['churn'].mean(),
            'num_clusters': int(df['cluster'].nunique()) if 'cluster' in df.columns else 0,
            'best_model': best_model_name  # Real best model from comparison
        }

    except Exception as e:
        print(f"Error loading churn data: {e}")
        import traceback
        traceback.print_exc()
        return None


# Hàm chuẩn hóa min-max
def minmax(s: pd.Series) -> pd.Series:
    if s.nunique() <= 1:
        return pd.Series(0.5, index=s.index)
    r = (s - s.min()) / (s.max() - s.min())
    return r.fillna(0.5)


def load_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded data: {df.shape} from {input_path}")
    return df


# 1) TẠO CÁC PROXY CHURN (ĐẦY ĐỦ)
def create_proxy_churn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Nhóm biến cho từng loại churn
    exp_cols = [c for c in [
        "Delivery Rating", "Restaurant Rating",
        "Bad past experience_encoded", "Poor Hygiene_encoded",
        "Late Delivery_encoded", "Influence of rating_encoded",
        "pca_service_issue", "Maximum wait time_encoded"
    ] if c in df.columns]

    use_cols = [c for c in [
        "No. of orders placed", "Delivery Time",
        "Ease and convenient_encoded", "Self Cooking_encoded",
        "Health Concern_encoded", "pca_convenience"
    ] if c in df.columns]

    val_cols = [c for c in [
        "Order Value", "More Offers and Discount_encoded",
        "pca_deal_sensitive", "cluster"
    ] if c in df.columns]

    # --- churn_exp_raw ---
    if "churn_exp_raw" not in df.columns:
        exp_score = 0
        for c in exp_cols:
            if "Rating" in c:
                exp_score += -df[c]
            else:
                exp_score += df[c]
        df["churn_exp_raw"] = minmax(exp_score)

    # --- churn_use_raw ---
    if "churn_use_raw" not in df.columns:
        use_score = 0
        for c in use_cols:
            if c == "No. of orders placed":
                use_score += -df[c]
            elif c == "Ease and convenient_encoded":
                use_score += -df[c]
            else:
                use_score += df[c]
        df["churn_use_raw"] = minmax(use_score)

    # --- churn_val_raw ---
    if "churn_val_raw" not in df.columns:
        val_score = 0
        for c in val_cols:
            if c == "Order Value":
                val_score += -df[c]
            else:
                val_score += df[c]
        df["churn_val_raw"] = minmax(val_score)

    # --- churn_percent (hybrid) --
    w_exp, w_use, w_val = 0.4, 0.3, 0.3
    df["churn_percent"] = (w_exp * df["churn_exp_raw"] +
                           w_use * df["churn_use_raw"] +
                           w_val * df["churn_val_raw"])
    df["churn"] = (df["churn_percent"] >= 0.5).astype(int)

    print("Proxy churn created. Churn distribution:")
    print(df['churn'].value_counts(normalize=True).mul(100).round(2))
    return df


# 2) PHÁT HIỆN LEAKAGE TỰ ĐỘNG
# - cờ theo tên (rating, experience, issue, hygiene, past_experience)
# - cờ theo correlation tuyệt đối > 0.6 với churn_percent
# - cờ theo AUC_Solo (một biến đơn dự đoán nhãn quá tốt => khả năng leakage)
def detect_leakage(df: pd.DataFrame, threshold_auc=0.90, threshold_corr=0.6):
    print("Running automatic leakage detection...")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Tính AUC khi dùng mỗi feature numeric riêng lẻ (bỏ cột churn, cluster, id)
    auc_list = []
    for c in num_cols:
        if c in ["churn", "churn_percent"]:
            continue
        if df[c].nunique() < 3:
            continue
        try:
            auc = roc_auc_score(df["churn"], df[c])
            auc_list.append((c, auc))
        except Exception:
            continue
    auc_df = pd.DataFrame(auc_list, columns=["Feature", "AUC_Solo"]) if auc_list else pd.DataFrame(
        columns=["Feature", "AUC_Solo"])

    # corr với churn_percent
    corrs = df[num_cols].corr()["churn_percent"].to_dict() if "churn_percent" in df.columns else {k: 0 for k in
                                                                                                  num_cols}
    auc_df["Corr"] = auc_df["Feature"].map(corrs)

    # Cờ nghi ngờ
    suspicious_keywords = ["rating", "experience", "issue", "hygiene", "past", "complaint"]
    auc_df["ByNameFlag"] = auc_df["Feature"].str.lower().apply(lambda x: any(k in x for k in suspicious_keywords))
    auc_df["ByScoreFlag"] = auc_df["AUC_Solo"].abs() > threshold_auc
    auc_df["ByCorrFlag"] = auc_df["Corr"].abs() > threshold_corr

    leakage_df = auc_df[(auc_df["ByNameFlag"]) | (auc_df["ByScoreFlag"]) | (auc_df["ByCorrFlag"])]
    if leakage_df.empty:
        print("No suspicious leakage features found.")
        return []
    else:
        print("Suspicious leakage features (auto-detected):")
        print(leakage_df.sort_values(by='AUC_Solo', ascending=False).to_string(index=False))
        return leakage_df['Feature'].tolist()


# 3) Chọn biến cốt lõi (core features) — bạn có thể tinh chỉnh list này
# - giữ demographic, behavior, pca, cluster
def prepare_core_df(df: pd.DataFrame, leak_cols: list = None):
    core_vars = [
        'Age', 'Family size',
        'No. of orders placed', 'Order Value', 'Delivery Time',
        'pca_convenience', 'pca_service_issue', 'pca_deal_sensitive',
        'cluster'
    ]
    # chỉ giữ cột tồn tại
    core_vars = [c for c in core_vars if c in df.columns]
    # loại các biến bị coi là leakage
    leak_cols = leak_cols or []
    core_vars = [c for c in core_vars if c not in leak_cols]
    # thêm target
    core_vars += ['churn', 'churn_percent']

    df_core = df[core_vars].copy()
    print(f"Prepared core df with shape: {df_core.shape}")
    return df_core


# 4) Kiểm tra VIF (đa cộng tuyến) — áp dụng cho numeric features
def check_vif(df_core: pd.DataFrame):
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception as e:
        print("statsmodels not available, skipping VIF. Install statsmodels to run VIF.")
        return None

    X_vif = df_core.drop(columns=["churn", "churn_percent"], errors='ignore').select_dtypes(include=[np.number]).copy()
    X_vif = X_vif.fillna(X_vif.median())
    vif_data = pd.DataFrame({
        'Variable': X_vif.columns,
        'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    }).sort_values('VIF', ascending=False)
    print("VIF results:")
    print(vif_data.to_string(index=False))
    return vif_data


# 5) Tiền xử lý
def preprocess(df_core: pd.DataFrame):
    # Tách X, y
    X = df_core.drop(columns=['churn', 'churn_percent'], errors='ignore')
    y = df_core['churn']

    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X[num_features] = X[num_features].fillna(X[num_features].median())

    scaler = StandardScaler()
    X_scaled = X.copy()
    if len(num_features) > 0:
        X_scaled[num_features] = scaler.fit_transform(X_scaled[num_features])

    return X_scaled, y, scaler


# 6) TRAIN & ĐÁNH GIÁ MODEL (CHUẨN HÓA CÔNG BẰNG)
def compare_models(X: pd.DataFrame, y: pd.Series):
    models = {
        'Logistic': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RND),
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=RND, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, eval_metric='logloss', random_state=RND, n_jobs=-1)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
    results = {m: [] for m in models}

    for name, mdl in models.items():
        print(f"Evaluating model: {name}")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            Xtr, Xv = X.iloc[train_idx], X.iloc[val_idx]
            ytr, yv = y.iloc[train_idx], y.iloc[val_idx]

            pipe = ImbPipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RND, k_neighbors=5)),
                ("clf", mdl)
            ])

            pipe.fit(Xtr, ytr)
            proba = pipe.predict_proba(Xv)[:, 1]

            P, R, T = precision_recall_curve(yv, proba)
            F1s = 2 * P * R / (P + R + 1e-9)
            best_idx = np.nanargmax(F1s)
            best_thr = T[best_idx - 1] if best_idx > 0 else 0.5
            pred = (proba >= best_thr).astype(int)

            fold_metrics = {
                'Fold': fold,
                'AUC': roc_auc_score(yv, proba),
                'PR_AUC': average_precision_score(yv, proba),
                'F1': f1_score(yv, pred),
                'Precision': precision_score(yv, pred),
                'Recall': recall_score(yv, pred),
                'Accuracy': accuracy_score(yv, pred),
                'Brier': brier_score_loss(yv, proba),
                'BestThr': best_thr
            }
            results[name].append(fold_metrics)
            print(f" Fold {fold}: AUC={fold_metrics['AUC']:.3f} | F1={fold_metrics['F1']:.3f}")

    summary = []
    for name, metrics in results.items():
        dfm = pd.DataFrame(metrics)
        s = {
            'Model': name,
            'AUC': dfm['AUC'].mean(),
            'F1': dfm['F1'].mean(),
            'Precision': dfm['Precision'].mean(),
            'Recall': dfm['Recall'].mean(),
            'Accuracy': dfm['Accuracy'].mean(),
            'AUC_std': dfm['AUC'].std(),
            'F1_std': dfm['F1'].std(),
            'Brier_mean': dfm['Brier'].mean()
        }
        summary.append(s)

    eval_table = pd.DataFrame(summary).sort_values('AUC', ascending=False).reset_index(drop=True)
    print('\nModel comparison finished:')
    print(eval_table.to_string(index=False))
    return eval_table, results


# 7) TRAIN - VALIDATE - TEST (80/10/10) & LƯU MODEL CHUẨN
def train_final_and_save(X: pd.DataFrame, y: pd.Series, output_dir: str):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RND)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp,
                                                        random_state=RND)
    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    xgb_final = ImbPipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RND, k_neighbors=5)),
        ("clf", XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.07,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.2, reg_lambda=2.0, eval_metric='logloss', random_state=RND, n_jobs=-1
        ))
    ])

    xgb_final.fit(X_train, y_train)

    proba_val = xgb_final.predict_proba(X_valid)[:, 1]
    P, R, T = precision_recall_curve(y_valid, proba_val)
    F1s = 2 * P * R / (P + R + 1e-9)
    best_idx = np.nanargmax(F1s)
    best_thr = T[best_idx - 1] if best_idx > 0 else 0.5
    print(f"Best threshold (val): {best_thr:.3f}")

    proba_test = xgb_final.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= best_thr).astype(int)

    auc = roc_auc_score(y_test, proba_test)
    prauc = average_precision_score(y_test, proba_test)
    f1 = f1_score(y_test, pred_test)
    prec = precision_score(y_test, pred_test)
    rec = recall_score(y_test, pred_test)

    print(
        f"Final Test -> AUC: {auc:.3f} | PR_AUC: {prauc:.3f} | F1: {f1:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")

    bundle = {'model': xgb_final, 'threshold': best_thr}
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'best_churn_model.pkl')
    joblib.dump(bundle, model_path)
    print(f"Saved model bundle to: {model_path}")

    return bundle


# 8) Prediction table & display
def make_prediction_table(df: pd.DataFrame, X: pd.DataFrame, bundle: dict):
    model = bundle['model']
    thr = bundle['threshold']

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)

    # Giữ nguyên Customer_ID (ưu tiên chuỗi CUS001,...)
    if 'Customer_ID' in df.columns:
        cust_id = df['Customer_ID'].astype(str)
    else:
        cust_id = [f"CUS{i + 1:03d}" for i in range(len(X))]

    df_result = pd.DataFrame({
        'Customer_ID': cust_id,
        'cluster': df.get('cluster', pd.Series([None] * len(X))),
        'proba_churn': proba,
        'pred_churn': pred
    })

    cluster_labels = {
        0: "Khách hàng dịch vụ cao / nhạy cảm trải nghiệm",
        1: "Khách hàng nhạy ưu đãi / ổn định",
        2: "Khách hàng giá trị cao / trung thành vừa"
    }
    if 'cluster' in df_result.columns:
        try:
            df_result['cluster_label'] = df_result['cluster'].map(cluster_labels)
        except Exception:
            pass

    df_result['proba_churn_pct'] = (df_result['proba_churn'] * 100).map('{:.1f}%'.format)
    df_result['pred_churn_label'] = df_result['pred_churn'].map({0: 'Không rời bỏ', 1: 'Có nguy cơ'})

    return df_result


# 9)  Churn by segment - calculate data only
def churn_by_segment_data(df_core: pd.DataFrame, df_original: pd.DataFrame):
    """
    Tính churn rate theo phân khúc. Trả về DataFrame để dùng cho plotting.
    """
    import pandas as pd
    import numpy as np

    # --- Tổng quan churn ---
    avg_churn_flag = df_core['churn'].mean()
    avg_churn_pct = df_core['churn_percent'].mean()
    print('Tổng quan churn toàn bộ:')
    print(f"- Avg Churn (nhị phân): {avg_churn_flag:.2%}")
    print(f"- Avg Churn_percent (liên tục): {avg_churn_pct:.3f}")

    # --- Tìm cột phân khúc ---
    segment_col = None
    for cand in ['cluster', 'Gender', 'Segment', 'Group']:
        if cand in df_core.columns or cand in df_original.columns:
            segment_col = cand
            break
    if segment_col is None:
        print('Không tìm thấy cột phân khúc. Bỏ qua phần phân khúc.')
        return None

    # --- Gom nhóm ---
    seg_series = df_original[segment_col] if segment_col in df_original.columns else df_core[segment_col]
    churn_by_seg = (
        pd.DataFrame({
            segment_col: seg_series.values,
            'churn': df_core['churn'].values,
            'churn_percent': df_core['churn_percent'].values
        })
        .groupby(segment_col)
        .agg(n=('churn', 'size'), churn_rate=('churn', 'mean'), avg_churn_percent=('churn_percent', 'mean'))
        .sort_values('churn_rate', ascending=False)
    )

    print('Churn theo phân khúc:')
    print(churn_by_seg)

    return churn_by_seg


# 9a) Plot function for Tkinter embedding
def _plot_churn_rate_by_segment(ax, df_core: pd.DataFrame, df_original: pd.DataFrame):
    """
    Vẽ churn rate theo segment vào Axes (for Tkinter embedding).
    """
    import numpy as np

    churn_by_seg = churn_by_segment_data(df_core, df_original)
    if churn_by_seg is None or churn_by_seg.empty:
        ax.text(0.5, 0.5, 'No segment data available',
                ha='center', va='center', transform=ax.transAxes)
        return

    # Colors
    custom_palette = ['#644E94', '#9B86C2', '#E3D4E0']
    bg_color = "#FFFFFF"
    text_color = "#2E2E2E"

    ax.set_facecolor(bg_color)

    # Get data
    segment_col = churn_by_seg.index.name or 'cluster'
    data_reset = churn_by_seg.reset_index()

    # Bar plot
    x = np.arange(len(data_reset))
    bars = ax.bar(x, data_reset['churn_rate'],
                  color=custom_palette[:len(data_reset)], alpha=0.9)

    # Add percentage labels
    for bar, rate in zip(bars, data_reset['churn_rate']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom',
                fontsize=10, color=text_color, fontweight='bold')

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(data_reset[segment_col], rotation=15, ha='right')
    ax.set_ylabel('Churn Rate', fontsize=11, color=text_color)
    ax.set_xlabel(segment_col, fontsize=11, color=text_color)

    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


# 9b) Reasons pie chart for Tkinter
def _plot_reasons_pie(ax, feature_importance_df: pd.DataFrame):
    """
    Vẽ pie chart lý do churn (feature importance) vào Axes.
    """
    if feature_importance_df is None or feature_importance_df.empty:
        ax.text(0.5, 0.5, 'No feature importance data',
                ha='center', va='center', transform=ax.transAxes)
        return

    fi_df = feature_importance_df.head(8).copy()

    # Custom colors
    custom_colors = ['#644E94', '#7B5FA1', '#9B86C2', '#AF96B9',
                     '#BEA1C1', '#C7AAC6', '#E3D4E0', '#FAE4F2']

    # Pie chart with adjusted label distance
    wedges, texts, autotexts = ax.pie(
        fi_df['Importance'],
        labels=fi_df['Feature'],
        autopct='%1.1f%%',
        startangle=90,
        colors=custom_colors[:len(fi_df)],
        textprops={'fontsize': 6, 'color': '#2E2E2E'},
        labeldistance=1.1,  # Move labels closer to reduce cutoff
        pctdistance=0.75  # Position percentage text inside
    )

    # Format percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(6)

    # Format label text
    for text in texts:
        text.set_fontsize(6)

    # ax.set_title('Churn Reasons (Feature Importance)', fontsize=11,
    #             fontweight='bold', color='#2E2E2E', pad=10)


# 9c) Feature importance bar chart for Tkinter
def _plot_feature_importance(ax, feature_importance_df: pd.DataFrame, top_n=10):
    """
    Vẽ horizontal bar chart feature importance vào Axes.
    """
    if feature_importance_df is None or feature_importance_df.empty:
        ax.text(0.5, 0.5, 'No feature importance data',
                ha='center', va='center', transform=ax.transAxes)
        return

    fi_data = feature_importance_df.head(top_n)

    # Horizontal bars
    y_pos = range(len(fi_data))
    ax.barh(y_pos, fi_data['Importance'], color='#644E94', alpha=0.8)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fi_data['Feature'], fontsize=9)
    ax.set_xlabel('Importance', fontsize=10)
    # ax.set_title(f'Top {top_n} Features', fontsize=11, fontweight='bold', pad=10)
    ax.invert_yaxis()  # Highest at top

    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


# 9d) SHAP Summary Plot for Tkinter
def _plot_shap_summary(fig, bundle: dict, X: pd.DataFrame):
    """
    Vẽ SHAP summary plot (dot plot) vào Figure (for Tkinter embedding).
    Note: SHAP creates its own plot, so we pass Figure instead of Axes
    """
    if shap is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'SHAP not available\nInstall: pip install shap',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return

    try:
        model = bundle.get('model')
        clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else model

        # --- Gradient màu tím ---
        shap_gradient_high_low = ['#E3D4E0', '#644E94']
        custom_cmap = LinearSegmentedColormap.from_list("my_gradient", shap_gradient_high_low)

        # --- Tiền xử lý dữ liệu ---
        if hasattr(model, 'named_steps') and 'imputer' in model.named_steps and 'scaler' in model.named_steps:
            X_processed = model.named_steps['imputer'].transform(X)
            X_processed = model.named_steps['scaler'].transform(X_processed)
        else:
            X_processed = SimpleImputer(strategy='median').fit_transform(X)
            X_processed = StandardScaler().fit_transform(X)

        X_processed_df = pd.DataFrame(X_processed, columns=X.columns)

        # Sample data if too large for performance
        if len(X_processed_df) > 100:
            X_sample = X_processed_df.sample(n=100, random_state=42)
        else:
            X_sample = X_processed_df

        # --- Chọn explainer với error handling cho XGBoost base_score issue ---
        shap_values = None
        explainer = None

        if isinstance(clf, XGBClassifier):
            # Fix XGBoost base_score compatibility issue with SHAP
            try:
                import json
                booster = clf.get_booster()
                config = booster.save_config()
                config_dict = json.loads(config)

                # Fix base_score if it's in problematic format like '[5E-1]'
                if 'learner' in config_dict and 'learner_model_param' in config_dict['learner']:
                    base_score_str = config_dict['learner']['learner_model_param'].get('base_score', '0.5')
                    if '[' in base_score_str:
                        # Extract the number from [5E-1] format
                        base_score_str = base_score_str.strip('[]')
                        try:
                            base_score = float(base_score_str)
                            config_dict['learner']['learner_model_param']['base_score'] = str(base_score)
                            booster.load_config(json.dumps(config_dict))
                            print(f"✓ Fixed XGBoost base_score: {base_score}")
                        except ValueError:
                            # If still can't convert, use default
                            config_dict['learner']['learner_model_param']['base_score'] = '0.5'
                            booster.load_config(json.dumps(config_dict))
                            print("✓ Set XGBoost base_score to default 0.5")

                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_sample)
            except Exception as xgb_error:
                print(f"TreeExplainer failed for XGBoost: {xgb_error}")
                # Fallback to KernelExplainer
                explainer = None

        elif isinstance(clf, RandomForestClassifier):
            try:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_sample)
            except Exception as rf_error:
                print(f"TreeExplainer failed for RandomForest: {rf_error}")
                explainer = None

        elif isinstance(clf, LogisticRegression):
            try:
                explainer = shap.LinearExplainer(clf, X_sample)
                shap_values = explainer.shap_values(X_sample)
            except Exception as lr_error:
                print(f"LinearExplainer failed: {lr_error}")
                explainer = None

        # Fallback to KernelExplainer if other methods failed
        if explainer is None or shap_values is None:
            print("Using KernelExplainer as fallback (this may be slow)...")
            background = shap.sample(X_processed_df, min(50, len(X_processed_df)), random_state=42)

            def model_predict(x):
                try:
                    if hasattr(model, 'predict_proba'):
                        return model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1]
                    else:
                        return model.predict(pd.DataFrame(x, columns=X.columns))
                except:
                    # If full model doesn't work, try just clf
                    if hasattr(clf, 'predict_proba'):
                        return clf.predict_proba(x)[:, 1]
                    else:
                        return clf.predict(x)

            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X_sample, nsamples=50)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        # --- Create SHAP dot plot manually (avoiding pyplot conflicts) ---
        fig.clear()
        ax = fig.add_subplot(111)

        # Prepare data for dot plot
        feature_names = X.columns.tolist()
        feature_importance = np.abs(shap_values).mean(axis=0)

        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1][:10]  # Top 10
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_shap_values = shap_values[:, indices]

        # Get feature values for coloring
        X_sample_array = X_sample.values if hasattr(X_sample, 'values') else X_sample
        sorted_feature_values = X_sample_array[:, indices]

        # Normalize feature values for coloring (0 to 1)
        from matplotlib.colors import Normalize

        # Create the dot plot
        y_positions = np.arange(len(sorted_feature_names))

        for i, (feature_idx, feature_name) in enumerate(zip(indices, sorted_feature_names)):
            # Get SHAP values and feature values for this feature
            shap_vals = sorted_shap_values[:, i]
            feat_vals = sorted_feature_values[:, i]

            # Normalize feature values for coloring
            if feat_vals.max() > feat_vals.min():
                norm_vals = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min())
            else:
                norm_vals = np.ones_like(feat_vals) * 0.5

            # Create color map (light purple to dark purple)
            colors = custom_cmap(norm_vals)

            # Plot dots with jitter for visibility
            y_jitter = np.random.normal(len(sorted_feature_names) - 1 - i, 0.15, size=len(shap_vals))
            ax.scatter(shap_vals, y_jitter, c=colors, alpha=0.6, s=10, edgecolors='none')

        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels(sorted_feature_names[::-1], fontsize=9)
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=9)
        # ax.set_title('SHAP Summary Plot', fontsize=11, fontweight='bold', pad=10)
        ax.axvline(x=0, color='#999999', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.2)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Feature value', rotation=270, labelpad=15, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        fig.tight_layout()

        print("✓ SHAP summary plot created successfully")

    except Exception as e:
        print(f"✗ Error creating SHAP plot: {e}")
        import traceback
        traceback.print_exc()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'SHAP plot unavailable\n{str(e)[:50]}...',
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')


# Keep old function for backward compatibility (standalone mode)
def churn_by_segment_and_plot(df_core: pd.DataFrame, df_original: pd.DataFrame):
    """Legacy function - vẽ chart riêng (standalone mode)"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    churn_by_seg = churn_by_segment_data(df_core, df_original)
    if churn_by_seg is None:
        return None

    # --- Thiết lập style & 3 màu tím ---
    sns.set_style("whitegrid")
    custom_palette = ['#644E94', '#9B86C2', '#E3D4E0']
    bg_color = "#F9F7FB"
    text_color = "#2E2E2E"

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(bg_color)
    _plot_churn_rate_by_segment(ax, df_core, df_original)

    plt.tight_layout()
    plt.show()

    return churn_by_seg


# 10) SHAP analysis
def shap_analysis(bundle: dict, X: pd.DataFrame, output_dir: str):
    model = bundle.get('model')
    clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else model

    # --- Gradient và màu bar ---
    shap_gradient_high_low = ['#E3D4E0', '#644E94']
    custom_cmap = LinearSegmentedColormap.from_list("my_gradient", shap_gradient_high_low)
    shap_bar_color = shap_gradient_high_low[1]

    # --- Tiền xử lý dữ liệu ---
    if 'imputer' in model.named_steps and 'scaler' in model.named_steps:
        X_processed = model.named_steps['imputer'].transform(X)
        X_processed = model.named_steps['scaler'].transform(X_processed)
    else:
        X_processed = SimpleImputer(strategy='median').fit_transform(X)
        X_processed = StandardScaler().fit_transform(X)

    X_processed_df = pd.DataFrame(X_processed, columns=X.columns)

    # --- Chọn explainer ---
    try:
        if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(clf)
        elif isinstance(clf, LogisticRegression):
            explainer = shap.LinearExplainer(clf, X_processed_df)
        else:
            print('Model type không hỗ trợ trực tiếp — dùng KernelExplainer (chậm)')
            explainer = shap.KernelExplainer(clf.predict_proba, X_processed_df.sample(min(50, len(X_processed_df))))

        shap_values = explainer.shap_values(X_processed_df)
        if isinstance(shap_values, list):  # multi-class
            shap_values = shap_values[1]

        # --- Biểu đồ SHAP bar (mean |SHAP|) ---
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values, X_processed_df, feature_names=X.columns,
            plot_type='bar', show=False, color=shap_bar_color
        )
        plt.title("SHAP Feature Importance", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # --- Biểu đồ SHAP dot (chi tiết hướng ảnh hưởng) ---
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values, X_processed_df,
            feature_names=X.columns, show=False, cmap=custom_cmap
        )
        plt.title("SHAP Summary Plot", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # --- Xuất file CSV (nếu cần) ---
        mean_abs = np.abs(shap_values).mean(axis=0)
        fi_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': mean_abs
        }).sort_values('Importance', ascending=False)

        fi_path = os.path.join(output_dir, 'feature_importance.csv')
        fi_df.to_csv(fi_path, index=False)
        print(f'Feature importance saved to: {fi_path}')

        return fi_df

    except Exception as e:
        print('Error while running SHAP:', str(e))
        return None


# 11) Reason chart (pie) - Legacy standalone function
def reason_pie_chart(feature_importance_df: pd.DataFrame):
    """Legacy function - vẽ pie chart riêng (standalone mode)"""
    import matplotlib.pyplot as plt

    if feature_importance_df is None or feature_importance_df.empty:
        print('No feature importance data for pie chart.')
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_reasons_pie(ax, feature_importance_df)
    plt.tight_layout()
    plt.show()


# ------------------------ Main ------------------------
def main(args):
    input_path = args.input_path
    output_dir = args.output_dir
    test_mode = args.test_mode

    df = load_data(input_path)
    df = create_proxy_churn(df)

    leak_cols = detect_leakage(df)
    df_core = prepare_core_df(df, leak_cols=leak_cols)

    _ = check_vif(df_core)

    X, y, scaler = preprocess(df_core)

    if not test_mode:
        eval_table, _ = compare_models(X, y)
        # choose best by AUC
        best_model_name = eval_table.iloc[0]['Model']
        print(f"Best model (by CV AUC): {best_model_name}")

        bundle = train_final_and_save(X, y, output_dir)
    else:
        # test_mode: skip heavy training; try to load existing model
        model_path = os.path.join(output_dir, 'best_churn_model.pkl')
        if os.path.exists(model_path):
            bundle = joblib.load(model_path)
            print(f"Loaded existing model from {model_path}")
        else:
            raise FileNotFoundError('No saved model found in test_mode. Run without --test_mode True to train.')

    # Predictions and evaluation
    df_result = make_prediction_table(df, X, bundle)
    result_path = os.path.join(output_dir, 'churn_predictions_preview.csv')
    df_result.to_csv(result_path, index=False)
    print(f"Saved prediction preview to: {result_path}")

    # Churn by segment
    churn_by_seg = churn_by_segment_and_plot(df_core, df)

    # SHAP (if available)
    fi_df = None
    try:
        fi_df = shap_analysis(bundle, X, output_dir)
    except Exception as e:
        print('SHAP step failed:', str(e))

    # Reason chart
    if fi_df is not None:
        reason_pie_chart(fi_df)

    print('All done.')

    # Show Table Churn
    print("\n=== Xem bảng dự đoán churn theo cluster ===")
    cluster_input = input("Nhập số cluster muốn xem (hoặc Enter để xem tất cả): ")
    if cluster_input.strip():
        cluster_input = int(cluster_input)
        df_show = df_result[df_result['cluster'] == cluster_input]
    else:
        df_show = df_result
    # Chọn cột hiển thị rõ ràng
    cols_show = ['Customer_ID', 'cluster', 'proba_churn_pct', 'pred_churn_label', 'cluster_label']
    cols_show = [c for c in cols_show if c in df_show.columns]

    print("\nBảng dự đoán churn:")
    print(df_show[cols_show].head(15).to_string(index=False))