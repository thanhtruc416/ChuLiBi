#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RECOMMENDATION (CÁCH A: NGƯỠNG ĐỘNG + Fallback theo CHURN)
Bản chạy local/PyCharm — tự động dò file trong Dataset/Output theo cấu trúc ChuLiBi.

Tóm tắt:
- Đọc 2-3 file CSV: cluster (hành vi), expected loss (EL), và churn (tuỳ chọn).
- Chuẩn hoá ID, chọn cột EL hợp lệ, hợp nhất dữ liệu.
- Tính tín hiệu hành vi -> match_score -> priority cho từng gói.
- Tạo ngưỡng động (quantile) cho nhóm gói HEAVY theo keep_top, rồi chọn gói cho từng khách.
- In tóm tắt; có thể xuất df_rec ra CSV nếu truyền --out (mặc định ghi Dataset/Output/recommendations.csv).

Yêu cầu: Python 3.8+, pandas, numpy
"""

from pathlib import Path
import argparse
import sys
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np


# =========================
# [0] Defaults khớp cấu trúc ChuLiBi (file này ở ./Function)
# =========================
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent           # .../ChuLiBi
DATASET_DIR = PROJECT_ROOT / "Dataset"
OUTPUT_DIR  = DATASET_DIR / "Output"

# Chỉ dùng đúng file EL này
PREF_EL_NAMES = [
    "expected_loss_by_customer.csv",
]


# =========================
# [1] Tham số & hằng số
# =========================
ACTION_LIB: List[Dict] = [
    {"id": "SLA_UP", "name": "Ưu tiên giao nhanh/slot VIP + ETA rõ",
     "lift_hint": 0.42, "targets": ["Delivery Time↑", "Late Delivery↑", "pca_service_issue↑"]},
    {"id": "COUPON10", "name": "Coupon 10%/bundle deal",
     "lift_hint": 0.30, "targets": ["pca_deal_sensitive↑", "More Offers and Discount_encoded↑", "Influence of rating_encoded↑"]},
    {"id": "LOYALTY", "name": "Đăng ký loyalty/điểm thưởng quay lại",
     "lift_hint": 0.22, "targets": ["No. of orders placed↓", "pca_convenience↓"]},
    {"id": "QUALITY_SWITCH", "name": "Chuyển nhà hàng/đảm bảo chất lượng + xin lỗi",
     "lift_hint": 0.36, "targets": ["Restaurant Rating↓", "Bad past experience_encoded↑", "Poor Hygiene_encoded↑"]},
    {"id": "CARE_CALL", "name": "CS gọi chăm sóc/khảo sát ngắn",
     "lift_hint": 0.18, "targets": ["pca_service_issue↑", "Rating bất thường"]},
]

ACTION_LIB_EXT: List[Dict] = ACTION_LIB + [
    {"id": "REMIND_APP", "name": "Nhắc mở app + ưu đãi chung", "lift_hint": 0.10, "targets": []},
    {"id": "EDU_CONTENT", "name": "Gợi ý nội dung/giải thích lợi ích", "lift_hint": 0.06, "targets": []},
]

ACTION_CHANNEL: Dict[str, Tuple[str, str]] = {
    "SLA_UP": ("Push/SMS", "Ưu tiên giao nhanh + hiển thị ETA rõ cho đơn sắp tới"),
    "COUPON10": ("App/Email", "Tặng Coupon 10% và gợi ý combo phù hợp"),
    "LOYALTY": ("In-app", "Thăng hạng loyalty, x2 điểm tuần này"),
    "QUALITY_SWITCH": ("CS call", "Xin lỗi + đề xuất nhà hàng chất lượng, bảo chứng vệ sinh"),
    "CARE_CALL": ("CS call", "Gọi khảo sát ngắn để xử lý vấn đề bạn gặp phải"),
    "REMIND_APP": ("App/Email", "Nhắc mở app, xem ưu đãi chung"),
    "EDU_CONTENT": ("App/Email", "Gợi ý nội dung hữu ích, giải thích lợi ích dịch vụ"),
}

ID_CANDS = ["Customer_ID", "CustomerID", "customer_id", "cust_id", "User_ID", "user_id"]

EL_CANDS = [
    "ExpectedLoss_full_pred", "ExpectedLoss_money", "ExpectedLoss_score",
    "EL_money", "expected_loss_raw", "ExpectedLoss_pred", "EL", "expected_loss", "exp_loss",
    "EL_full_pred", "EL_noorder_pred", "EL_full_scaled", "EL_noorder_scaled"
]


# =========================
# [2] Helpers
# =========================
def autodetect_files() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Trả về (cluster, el, churn) nếu tìm thấy; cái nào không thấy -> None.
    - cluster: ưu tiên df_cluster_full.csv
    - el: cố định expected_loss_by_customer.csv
    - churn: ưu tiên churn_predictions_preview.csv
    """
    cl = OUTPUT_DIR / "df_cluster_full.csv"
    cl = cl if cl.exists() else None

    el = OUTPUT_DIR / "expected_loss_by_customer.csv"
    el = el if el.exists() else None

    ch = OUTPUT_DIR / "churn_predictions_preview.csv"
    ch = ch if ch.exists() else None

    return cl, el, ch


def pick_first(cands: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def detect_el_col(df_source: pd.DataFrame, id_col: Optional[str]) -> Optional[str]:
    # Ưu tiên tên phổ biến
    for c in EL_CANDS:
        if c in df_source.columns:
            s = pd.to_numeric(df_source[c], errors="coerce")
            if s.notna().any():
                return c
    # Nếu không có -> chọn numeric có phương sai lớn nhất
    num_cols = [c for c in df_source.columns if (id_col is None or c != id_col) and
                pd.api.types.is_numeric_dtype(df_source[c])]
    if not num_cols:
        return None
    variances = {c: pd.to_numeric(df_source[c], errors="coerce").var(skipna=True) for c in num_cols}
    return max(variances, key=lambda k: (variances[k] if pd.notna(variances[k]) else -1))


def safe01(v, default=0.5) -> float:
    try:
        x = float(v)
        return x if 0 <= x <= 1 else default
    except Exception:
        return default


def get_signals(row: pd.Series) -> Dict[str, float]:
    g = lambda c, d=0.5: row[c] if c in row.index else d
    return {
        "DeliveryTime": safe01(g("Delivery Time", 0.5)),  # chậm -> cao
        "LateDelivery": safe01(g("Late Delivery_encoded", 0.0), 0.0),  # trễ -> cao
        "PCA_Service": safe01(g("pca_service_issue", 0.0), 0.0),  # dịch vụ kém -> cao
        "PCA_Deal": safe01(g("pca_deal_sensitive", 0.0), 0.0),  # nhạy khuyến mãi -> cao
        "MoreOffers": safe01(g("More Offers and Discount_encoded", 0.0), 0.0),
        "InfluenceRating": safe01(g("Influence of rating_encoded", 0.0), 0.0),
        "OrdersLow": 1.0 - safe01(g("No. of orders placed", 0.5)),  # ít đơn -> cao
        "PCA_ConvenienceLow": 1.0 - safe01(g("pca_convenience", 0.5)),  # bất tiện -> cao
        "RestRatingLow": 1.0 - safe01(g("Restaurant Rating", 0.5)),  # rating thấp -> cao
        "BadExperience": safe01(g("Bad past experience_encoded", 0.0), 0.0),
        "PoorHygiene": safe01(g("Poor Hygiene_encoded", 0.0), 0.0),
        "RatingAbnormal": float(
            (safe01(g("Restaurant Rating", 0.5)) < 0.3) or
            (safe01(g("Delivery Rating", 0.5))   < 0.3)
        ),
    }


def match_score(sig: Dict[str, float], targets: List[str]) -> float:
    if not targets:
        return 0.0
    s = 0.0
    for t in targets:
        if t == "Delivery Time↑":
            s += sig["DeliveryTime"]
        elif t in ["Late Delivery↑", "Late Delivery_encoded↑"]:
            s += sig["LateDelivery"]
        elif t == "pca_service_issue↑":
            s += sig["PCA_Service"]
        elif t == "pca_deal_sensitive↑":
            s += sig["PCA_Deal"]
        elif t == "More Offers and Discount_encoded↑":
            s += sig["MoreOffers"]
        elif t == "Influence of rating_encoded↑":
            s += sig["InfluenceRating"]
        elif t == "No. of orders placed↓":
            s += sig["OrdersLow"]
        elif t == "pca_convenience↓":
            s += sig["PCA_ConvenienceLow"]
        elif t == "Restaurant Rating↓":
            s += sig["RestRatingLow"]
        elif t == "Bad past experience_encoded↑":
            s += sig["BadExperience"]
        elif t == "Poor Hygiene_encoded↑":
            s += sig["PoorHygiene"]
        elif t == "Rating bất thường":
            s += sig["RatingAbnormal"]
    return float(np.clip(s / max(1, len(targets)), 0, 1))


def eligible(action_id: str, sig: Dict[str, float]) -> bool:
    # Điều kiện "cửa vào" cho một số gói nặng
    if action_id == "SLA_UP":
        return (sig["DeliveryTime"] > 0.6) or (sig["LateDelivery"] > 0.6) or (sig["PCA_Service"] > 0.6)
    if action_id == "QUALITY_SWITCH":
        return (sig["RestRatingLow"] > 0.6) or (sig["BadExperience"] > 0.6) or (sig["PoorHygiene"] > 0.6)
    return True


def compute_best_action(row: pd.Series, elig_func, actions: List[Dict]) -> Optional[Dict]:
    # priority = EL_norm × churn × lift_est
    # lift_est = lift_hint × (0.5 + 0.5×match_score)
    sig = get_signals(row)
    ELn = float(row.get("EL_norm", 0.0))
    churn = float(row.get("proba_churn", 1.0))
    cands = []
    for a in actions:
        if not elig_func(a["id"], sig):
            continue
        m = match_score(sig, a["targets"])
        lift_est = a["lift_hint"] * (0.5 + 0.5 * m)
        p = ELn * churn * lift_est
        cands.append((a["id"], a["name"], m, lift_est, p))
    if not cands:
        return None
    best = max(cands, key=lambda z: z[4])
    return {
        "action_id": best[0],
        "action_name": best[1],
        "match": round(best[2], 3),
        "lift_est": round(best[3], 3),
        "priority_score": float(best[4]),
    }


def action_group(aid: str) -> str:
    heavy_actions = ["SLA_UP", "QUALITY_SWITCH", "COUPON10", "LOYALTY", "CARE_CALL"]
    light_actions = ["REMIND_APP", "EDU_CONTENT"]
    if aid in heavy_actions:
        return "HEAVY"
    if aid in light_actions:
        return "LIGHT"
    if aid == "NO_ACTION":
        return "NONE"
    return "OTHER"


# =========================
# [3] Core pipeline
# =========================
def run_pipeline(fn_cluster: Path,
                 fn_el: Path,
                 fn_churn: Optional[Path] = None,
                 keep_top: float = 0.55,
                 churn_fallback_thr: float = 0.05,
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Trả về:
      - df: bảng hợp nhất (chuẩn hoá) có EL_norm, proba_churn
      - df_rec: khuyến nghị 1 dòng/khách
      - thr: ngưỡng động cho HEAVY
    """
    # ---- Load
    if not fn_cluster or not fn_cluster.exists():
        raise FileNotFoundError(f"Thiếu file cluster: {fn_cluster}")
    if not fn_el or not fn_el.exists():
        raise FileNotFoundError(f"Thiếu file EL: {fn_el}")

    df_c = pd.read_csv(fn_cluster)
    df_el = pd.read_csv(fn_el)
    df_p = pd.read_csv(fn_churn) if (fn_churn and fn_churn.exists()) else None

    # ---- Detect & normalize IDs
    id_cl = pick_first(ID_CANDS, df_c)
    id_el = pick_first(ID_CANDS, df_el)
    if not id_cl or not id_el:
        raise ValueError("Không tìm thấy cột ID trong cluster hoặc EL.")

    for d, c in [(df_c, id_cl), (df_el, id_el)]:
        d[c] = d[c].astype(str).str.strip()

    # ---- Detect EL column
    el_col = detect_el_col(df_el, id_el)
    if not el_col:
        raise ValueError("Không tìm thấy cột Expected Loss trong file EL; kiểm tra header.")

    # ---- Dedupe
    df_c = df_c.dropna(subset=[id_cl]).drop_duplicates(subset=[id_cl], keep="first")
    tmp_el = df_el.copy()
    tmp_el[el_col] = pd.to_numeric(tmp_el[el_col], errors="coerce")
    df_el = (tmp_el.dropna(subset=[id_el])
             .sort_values(el_col, ascending=False)
             .drop_duplicates(subset=[id_el], keep="first"))

    id_ch = None
    if df_p is not None:
        id_ch = pick_first(ID_CANDS, df_p)
        if id_ch:
            df_p[id_ch] = df_p[id_ch].astype(str).str.strip()
            df_p = df_p.dropna(subset=[id_ch]).drop_duplicates(subset=[id_ch], keep="first")

    # ---- Merge core
    df = df_c.merge(df_el, left_on=id_cl, right_on=id_el, how="left", suffixes=("", "_el"))
    df = df.rename(columns={id_cl: "Customer_ID_STD"}).reset_index(drop=True)

    # ---- Merge churn (2 bước an toàn, tránh lỗi ngoặc)
    if df_p is not None and id_ch:
        churn_cols = [c for c in ["proba_churn", "proba_churn_model"] if c in df_p.columns]
        if churn_cols:
            ch_col = churn_cols[0]
            churn_tmp = df_p[[id_ch, ch_col]].copy()
            churn_tmp = churn_tmp.rename(columns={id_ch: "Customer_ID_STD", ch_col: "proba_churn_raw"})
            df = pd.merge(df, churn_tmp, on="Customer_ID_STD", how="left")

    # ---- EL_norm
    df["EL"] = pd.to_numeric(df[el_col], errors="coerce")
    if df["EL"].notna().any():
        df["EL_norm"] = df["EL"].rank(method="average", pct=True).fillna(0.0).clip(0, 1)
    else:
        df["EL"] = 0.0
        df["EL_norm"] = 0.0
    df["EL_norm"] = df["EL_norm"].where(df["EL"] > 0, 0.0)

    # ---- churn
    if "proba_churn_raw" in df.columns:
        df["proba_churn"] = pd.to_numeric(df["proba_churn_raw"], errors="coerce").clip(0, 1).fillna(1.0)
    else:
        df["proba_churn"] = 1.0

    # ---- Tạo ngưỡng động từ priority HEAVY dương
    tmp_scores: List[float] = []
    for _, r in df.iterrows():
        best = compute_best_action(r, eligible, ACTION_LIB)
        if best and best["priority_score"] > 0:
            tmp_scores.append(best["priority_score"])
    thr = float(np.quantile(tmp_scores, 1 - keep_top)) if tmp_scores else 0.0

    # ---- Chọn gói/khách
    rows = []
    for _, r in df.iterrows():
        base = compute_best_action(r, eligible, ACTION_LIB)  # HEAVY only
        if base and base["priority_score"] >= thr:
            chosen = base
        else:
            chosen = None
            if float(r["EL_norm"]) > 0:
                churn = float(r["proba_churn"])
                fallback_id = "REMIND_APP" if churn >= churn_fallback_thr else "EDU_CONTENT"
                light = compute_best_action(
                    r,
                    lambda aid, s: True,
                    [a for a in ACTION_LIB_EXT if a["id"] == fallback_id]
                )
                if light and light["priority_score"] > 0:
                    chosen = light

            if not chosen:
                chosen = {
                    "action_id": "NO_ACTION",
                    "action_name": "Không gửi – không đủ điều kiện / ưu tiên thấp",
                    "match": 0.0,
                    "lift_est": 0.0,
                    "priority_score": 0.0
                }

        ch, tmpl = ACTION_CHANNEL.get(chosen["action_id"], ("App", ""))
        rows.append({
            "Customer_ID": r["Customer_ID_STD"],
            "action_id": chosen["action_id"],
            "action_name": chosen["action_name"],
            "priority_score": round(float(chosen["priority_score"]), 6),
            "channel": ch,
            "template": tmpl
        })

    df_rec = pd.DataFrame(rows)
    df_rec["group"] = df_rec["action_id"].apply(action_group)
    return df, df_rec, thr


# =========================
# [4] CLI
# =========================
def parse_args():
    cl_default, el_default, ch_default = autodetect_files()

    p = argparse.ArgumentParser(
        description="Recommendation pipeline (dynamic threshold + churn fallback) — PyCharm/CLI version"
    )
    p.add_argument("--cluster", type=Path, default=cl_default,
                   help=f"CSV path for cluster/behavior features (default: {cl_default})")
    p.add_argument("--el", type=Path, default=el_default,
                   help=f"CSV path for expected loss by customer (default: {el_default})")
    p.add_argument("--churn", type=Path, default=ch_default,
                   help=f"CSV path for churn proba (optional, default: {ch_default})")
    p.add_argument("--keep-top", type=float, default=0.55,
                   help="Tỷ lệ giữ top HEAVY sau cắt ngưỡng động")
    p.add_argument("--churn-fallback-thr", type=float, default=0.05,
                   help="Ngưỡng churn tách REMIND_APP (>=) và EDU_CONTENT (<)")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "recommendations.csv",
                   help=f"CSV output (default: {OUTPUT_DIR / 'recommendations.csv'})")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        df, df_rec, thr = run_pipeline(
            fn_cluster=args.cluster,
            fn_el=args.el,
            fn_churn=args.churn,
            keep_top=args.keep_top,
            churn_fallback_thr=args.churn_fallback_thr
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Tóm tắt console
    base_customers = df["Customer_ID_STD"].nunique()
    out_rows = len(df_rec)
    uniq_customers = df_rec["Customer_ID"].nunique()

    print(f"Base customers: {base_customers:,}")
    print(f"Output rows (1 dòng/khách): {out_rows:,} | unique customers: {uniq_customers:,}")

    print("\nAction distribution:")
    print(df_rec["action_id"].value_counts())

    print(f"\nkeep_top = {args.keep_top:.2f}  =>  thr (ngưỡng động) = {thr:.6f}")

    print("\n--- Top 20 priority (sau chọn) ---")
    cols_show = ["Customer_ID", "action_id", "priority_score", "channel"]
    print(df_rec.sort_values("priority_score", ascending=False).head(20)[cols_show].to_string(index=False))

    print("\nPhân phối nhóm:")
    print(df_rec["group"].value_counts())

    if args.out:
        try:
            df_rec.to_csv(args.out, index=False)
            print(f"\nĐã lưu recommendations -> {args.out.resolve()}")
        except Exception as e:
            print(f"[WARN] Không thể ghi file output: {e}", file=sys.stderr)

# =========================================================
# [5] API helper cho UI / Notebook
# =========================================================
def get_recommendation_data(
    cluster_path: Optional[Path] = None,
    el_path: Optional[Path] = None,
    churn_path: Optional[Path] = None,
    keep_top: float = 0.55,
    churn_fallback_thr: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Hàm tiện ích cho UI:
    - Chạy pipeline và trả kết quả DataFrame (df_full, df_rec, thr).
    - Không ghi file, chỉ trả về dữ liệu.
    """
    # autodetect nếu không truyền
    if not cluster_path or not el_path:
        cl, el, ch = autodetect_files()
        cluster_path = cluster_path or cl
        el_path = el_path or el
        churn_path = churn_path or ch

    print(f"[INFO] Using cluster={cluster_path}, el={el_path}, churn={churn_path}")

    df, df_rec, thr = run_pipeline(
        fn_cluster=cluster_path,
        fn_el=el_path,
        fn_churn=churn_path,
        keep_top=keep_top,
        churn_fallback_thr=churn_fallback_thr,
    )

    # tóm tắt console
    print(f" Tổng KH: {len(df):,} | Có khuyến nghị: {len(df_rec):,}")
    print(df_rec["action_id"].value_counts())
    print(f"Ngưỡng động HEAVY (thr) = {thr:.6f}")

    return df, df_rec, thr