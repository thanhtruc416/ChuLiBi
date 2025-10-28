# Frame10_Recommend.py
# Recommendation Engine for Customer Retention

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class RecommendationEngine:
    """
    Recommendation engine that suggests the best action for each customer
    based on Expected Loss, churn probability, and behavioral signals.
    """
    
    def __init__(self, keep_top: float = 0.55, churn_fallback_thr: float = 0.05):
        """
        Args:
            keep_top: Proportion of heavy actions to keep after dynamic threshold (0-1)
            churn_fallback_thr: Churn threshold to split REMIND_APP vs EDU_CONTENT
        """
        self.keep_top = keep_top
        self.churn_fallback_thr = churn_fallback_thr
        
        # Action library - HEAVY actions (high impact/cost)
        self.ACTION_LIB = [
            {
                "id": "SLA_UP",
                "name": "Ưu tiên giao nhanh/slot VIP + ETA rõ",
                "lift_hint": 0.42,
                "targets": ["Delivery Time↑", "Late Delivery↑", "pca_service_issue↑"]
            },
            {
                "id": "COUPON10",
                "name": "Coupon 10%/bundle deal",
                "lift_hint": 0.30,
                "targets": ["pca_deal_sensitive↑", "More Offers and Discount_encoded↑", "Influence of rating_encoded↑"]
            },
            {
                "id": "LOYALTY",
                "name": "Đăng ký loyalty/điểm thưởng quay lại",
                "lift_hint": 0.22,
                "targets": ["No. of orders placed↓", "pca_convenience↓"]
            },
            {
                "id": "QUALITY_SWITCH",
                "name": "Chuyển nhà hàng/đảm bảo chất lượng + xin lỗi",
                "lift_hint": 0.36,
                "targets": ["Restaurant Rating↓", "Bad past experience_encoded↑", "Poor Hygiene_encoded↑"]
            },
            {
                "id": "CARE_CALL",
                "name": "CS gọi chăm sóc/khảo sát ngắn",
                "lift_hint": 0.18,
                "targets": ["pca_service_issue↑", "Rating bất thường"]
            },
        ]
        
        # LIGHT actions (fallback)
        self.ACTION_LIB_LIGHT = [
            {
                "id": "REMIND_APP",
                "name": "Nhắc mở app + ưu đãi chung",
                "lift_hint": 0.10,
                "targets": []
            },
            {
                "id": "EDU_CONTENT",
                "name": "Gợi ý nội dung/giải thích lợi ích",
                "lift_hint": 0.06,
                "targets": []
            },
        ]
        
        # Channel and template for each action
        self.ACTION_CHANNEL = {
            "SLA_UP":         ("Push/SMS", "Ưu tiên giao nhanh + hiển thị ETA rõ cho đơn sắp tới"),
            "COUPON10":       ("App/Email", "Tặng Coupon 10% và gợi ý combo phù hợp"),
            "LOYALTY":        ("In-app", "Thăng hạng loyalty, x2 điểm tuần này"),
            "QUALITY_SWITCH": ("CS call", "Xin lỗi + đề xuất nhà hàng chất lượng, bảo chứng vệ sinh"),
            "CARE_CALL":      ("CS call", "Gọi khảo sát ngắn để xử lý vấn đề bạn gặp phải"),
            "REMIND_APP":     ("App/Email", "Nhắc mở app, xem ưu đãi chung"),
            "EDU_CONTENT":    ("App/Email", "Gợi ý nội dung hữu ích, giải thích lợi ích dịch vụ"),
        }
        
        self.ID_CANDIDATES = ["Customer_ID", "CustomerID", "customer_id", "cust_id", "User_ID", "user_id"]
        self.EL_CANDIDATES = [
            "ExpectedLoss_full_pred", "ExpectedLoss_money", "ExpectedLoss_score",
            "EL_money", "expected_loss_raw", "ExpectedLoss_pred", "EL", "expected_loss", "exp_loss",
            "EL_full_pred", "EL_noorder_pred", "EL_full_scaled", "EL_noorder_scaled"
        ]
    
    def _pick_first_column(self, candidates: List[str], df: pd.DataFrame) -> Optional[str]:
        """Pick the first matching column from candidates"""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _detect_el_column(self, df: pd.DataFrame, id_col: str) -> Optional[str]:
        """Detect the Expected Loss column in the dataframe"""
        # Try common names first
        for col in self.EL_CANDIDATES:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                if s.notna().any():
                    return col
        
        # Fallback: find numeric column with highest variance
        num_cols = [c for c in df.columns 
                   if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            return None
        
        variances = {}
        for col in num_cols:
            var = pd.to_numeric(df[col], errors="coerce").var(skipna=True)
            variances[col] = var if pd.notna(var) else -1
        
        return max(variances, key=variances.get)
    
    def _safe_01(self, value, default: float = 0.5) -> float:
        """Convert value to [0,1] range safely"""
        try:
            x = float(value)
            return x if 0 <= x <= 1 else default
        except:
            return default
    
    def _get_signals(self, row: pd.Series) -> Dict[str, float]:
        """Extract behavioral signals from customer data"""
        def g(col, default=0.5):
            return row[col] if col in row.index else default
        
        return {
            "DeliveryTime":       self._safe_01(g("Delivery Time", 0.5)),
            "LateDelivery":       self._safe_01(g("Late Delivery_encoded", 0.0), 0.0),
            "PCA_Service":        self._safe_01(g("pca_service_issue", 0.0), 0.0),
            "PCA_Deal":           self._safe_01(g("pca_deal_sensitive", 0.0), 0.0),
            "MoreOffers":         self._safe_01(g("More Offers and Discount_encoded", 0.0), 0.0),
            "InfluenceRating":    self._safe_01(g("Influence of rating_encoded", 0.0), 0.0),
            "OrdersLow":          1.0 - self._safe_01(g("No. of orders placed", 0.5)),
            "PCA_ConvenienceLow": 1.0 - self._safe_01(g("pca_convenience", 0.5)),
            "RestRatingLow":      1.0 - self._safe_01(g("Restaurant Rating", 0.5)),
            "BadExperience":      self._safe_01(g("Bad past experience_encoded", 0.0), 0.0),
            "PoorHygiene":        self._safe_01(g("Poor Hygiene_encoded", 0.0), 0.0),
            "RatingAbnormal":     float(
                (self._safe_01(g("Restaurant Rating", 0.5)) < 0.3) or
                (self._safe_01(g("Delivery Rating", 0.5)) < 0.3)
            ),
        }
    
    def _match_score(self, signals: Dict[str, float], targets: List[str]) -> float:
        """Calculate match score based on behavioral signals and action targets"""
        if not targets:
            return 0.0
        
        score = 0.0
        for target in targets:
            if target == "Delivery Time↑":
                score += signals["DeliveryTime"]
            elif target in ["Late Delivery↑", "Late Delivery_encoded↑"]:
                score += signals["LateDelivery"]
            elif target == "pca_service_issue↑":
                score += signals["PCA_Service"]
            elif target == "pca_deal_sensitive↑":
                score += signals["PCA_Deal"]
            elif target == "More Offers and Discount_encoded↑":
                score += signals["MoreOffers"]
            elif target == "Influence of rating_encoded↑":
                score += signals["InfluenceRating"]
            elif target == "No. of orders placed↓":
                score += signals["OrdersLow"]
            elif target == "pca_convenience↓":
                score += signals["PCA_ConvenienceLow"]
            elif target == "Restaurant Rating↓":
                score += signals["RestRatingLow"]
            elif target == "Bad past experience_encoded↑":
                score += signals["BadExperience"]
            elif target == "Poor Hygiene_encoded↑":
                score += signals["PoorHygiene"]
            elif target == "Rating bất thường":
                score += signals["RatingAbnormal"]
        
        return float(np.clip(score / max(1, len(targets)), 0, 1))
    
    def _is_eligible(self, action_id: str, signals: Dict[str, float]) -> bool:
        """Check if customer is eligible for the action"""
        if action_id == "SLA_UP":
            return (signals["DeliveryTime"] > 0.6 or 
                   signals["LateDelivery"] > 0.6 or 
                   signals["PCA_Service"] > 0.6)
        
        if action_id == "QUALITY_SWITCH":
            return (signals["RestRatingLow"] > 0.6 or 
                   signals["BadExperience"] > 0.6 or 
                   signals["PoorHygiene"] > 0.6)
        
        return True
    
    def _compute_best_action(self, row: pd.Series, actions: List[Dict]) -> Optional[Dict]:
        """Compute best action for a customer from given action list"""
        signals = self._get_signals(row)
        el_norm = float(row["EL_norm"])
        churn = float(row["proba_churn"])
        
        candidates = []
        for action in actions:
            if not self._is_eligible(action["id"], signals):
                continue
            
            match = self._match_score(signals, action["targets"])
            lift_est = action["lift_hint"] * (0.5 + 0.5 * match)
            priority = el_norm * churn * lift_est
            
            candidates.append({
                "action_id": action["id"],
                "action_name": action["name"],
                "match": round(match, 3),
                "lift_est": round(lift_est, 3),
                "priority_score": priority
            })
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda x: x["priority_score"])
    
    def generate_recommendations(self, 
                                data_path: Path,
                                cluster_file: str = "df_cluster_full.csv",
                                churn_file: str = "churn_predictions_preview.csv") -> pd.DataFrame:
        """
        Generate recommendations for all customers
        
        Args:
            data_path: Path to data directory
            cluster_file: Cluster data filename
            churn_file: Churn predictions filename
            
        Returns:
            DataFrame with recommendations (one row per customer)
        """
        # Load data
        cluster_path = data_path / cluster_file
        churn_path = data_path / churn_file
        
        if not cluster_path.exists():
            raise FileNotFoundError(f"Cluster file not found: {cluster_path}")
        
        df_cluster = pd.read_csv(cluster_path)
        df_churn = pd.read_csv(churn_path) if churn_path.exists() else None
        
        # Detect ID columns
        id_cluster = self._pick_first_column(self.ID_CANDIDATES, df_cluster)
        if not id_cluster:
            raise ValueError("No ID column found in cluster data")
        
        # Normalize IDs
        df_cluster[id_cluster] = df_cluster[id_cluster].astype(str).str.strip()
        df_cluster = df_cluster.dropna(subset=[id_cluster]).drop_duplicates(subset=[id_cluster], keep="first")
        
        # Merge with churn data if available
        if df_churn is not None:
            id_churn = self._pick_first_column(self.ID_CANDIDATES, df_churn)
            if id_churn:
                df_churn[id_churn] = df_churn[id_churn].astype(str).str.strip()
                df_churn = df_churn.dropna(subset=[id_churn]).drop_duplicates(subset=[id_churn], keep="first")
                
                churn_cols = [c for c in ["proba_churn", "proba_churn_model"] if c in df_churn.columns]
                if churn_cols:
                    merge_data = df_churn[[id_churn, churn_cols[0]]].rename(
                        columns={id_churn: id_cluster, churn_cols[0]: "proba_churn"}
                    )
                    df_cluster = df_cluster.merge(merge_data, on=id_cluster, how="left")
        
        # Ensure churn column exists
        if "proba_churn" in df_cluster.columns:
            df_cluster["proba_churn"] = pd.to_numeric(
                df_cluster["proba_churn"], errors="coerce"
            ).clip(0, 1).fillna(1.0)
        else:
            df_cluster["proba_churn"] = 1.0
        
        # Calculate EL_norm (Expected Loss normalized)
        # For this version, we'll use a synthetic EL based on available features
        df_cluster["EL"] = self._calculate_synthetic_el(df_cluster)
        
        if df_cluster["EL"].notna().any():
            df_cluster["EL_norm"] = df_cluster["EL"].rank(method="average", pct=True).fillna(0.0).clip(0, 1)
        else:
            df_cluster["EL_norm"] = 0.0
        
        df_cluster["EL_norm"] = df_cluster["EL_norm"].where(df_cluster["EL"] > 0, 0.0)
        
        # Rename ID column to standard name
        df = df_cluster.rename(columns={id_cluster: "Customer_ID"})
        
        # Calculate dynamic threshold
        tmp_scores = []
        for _, row in df.iterrows():
            best = self._compute_best_action(row, self.ACTION_LIB)
            if best and best["priority_score"] > 0:
                tmp_scores.append(best["priority_score"])
        
        threshold = float(np.quantile(tmp_scores, 1 - self.keep_top)) if tmp_scores else 0.0
        
        # Generate recommendations
        results = []
        for _, row in df.iterrows():
            # Try heavy actions first
            best_heavy = self._compute_best_action(row, self.ACTION_LIB)
            
            chosen = None
            if best_heavy and best_heavy["priority_score"] >= threshold:
                chosen = best_heavy
            else:
                # Fallback to light actions
                if float(row["EL_norm"]) > 0:
                    churn = float(row["proba_churn"])
                    fallback_id = "REMIND_APP" if churn >= self.churn_fallback_thr else "EDU_CONTENT"
                    
                    light_action = [a for a in self.ACTION_LIB_LIGHT if a["id"] == fallback_id]
                    best_light = self._compute_best_action(row, light_action)
                    
                    if best_light and best_light["priority_score"] > 0:
                        chosen = best_light
            
            if not chosen:
                chosen = {
                    "action_id": "NO_ACTION",
                    "action_name": "Không gửi – không đủ điều kiện / ưu tiên thấp",
                    "match": 0.0,
                    "lift_est": 0.0,
                    "priority_score": 0.0
                }
            
            channel, template = self.ACTION_CHANNEL.get(
                chosen["action_id"], ("App", "")
            )
            
            # Get cluster info if available
            cluster_name = "Unknown"
            if "Cluster" in row.index:
                cluster_name = str(row["Cluster"])
            elif "cluster" in row.index:
                cluster_name = str(row["cluster"])
            
            # Format churn risk as percentage
            churn_risk = f"{int(row['proba_churn'] * 100)}%"
            
            results.append({
                "Customer_ID": row["Customer_ID"],
                "Cluster": cluster_name,
                "Churn_Risk": churn_risk,
                "action_id": chosen["action_id"],
                "action_name": chosen["action_name"],
                "priority_score": round(float(chosen["priority_score"]), 6),
                "channel": channel,
                "template": template
            })
        
        return pd.DataFrame(results)
    
    def _calculate_synthetic_el(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate synthetic Expected Loss based on available features
        This is a placeholder - replace with actual EL calculation
        """
        # Weighted combination of key risk factors
        el = pd.Series(0.0, index=df.index)
        
        if "Delivery Time" in df.columns:
            el += pd.to_numeric(df["Delivery Time"], errors="coerce").fillna(0.5) * 0.3
        
        if "Restaurant Rating" in df.columns:
            rating = pd.to_numeric(df["Restaurant Rating"], errors="coerce").fillna(0.5)
            el += (1 - rating) * 0.25
        
        if "No. of orders placed" in df.columns:
            orders = pd.to_numeric(df["No. of orders placed"], errors="coerce").fillna(0.5)
            el += (1 - orders) * 0.2
        
        if "pca_service_issue" in df.columns:
            el += pd.to_numeric(df["pca_service_issue"], errors="coerce").fillna(0.0) * 0.25
        
        return el.clip(0, 1)


# Convenience function for easy access
def get_recommendations(data_path: Path = None, 
                       keep_top: float = 0.55,
                       churn_fallback_thr: float = 0.05) -> pd.DataFrame:
    """
    Get recommendations for all customers
    
    Args:
        data_path: Path to data directory (defaults to Dataset/Output)
        keep_top: Proportion of heavy actions to keep (0-1)
        churn_fallback_thr: Churn threshold for fallback actions
    
    Returns:
        DataFrame with recommendations
    """
    if data_path is None:
        # Default to Dataset/Output relative to Function folder
        current_dir = Path(__file__).parent
        data_path = current_dir.parent / "Dataset" / "Output"
    
    engine = RecommendationEngine(keep_top=keep_top, churn_fallback_thr=churn_fallback_thr)
    return engine.generate_recommendations(data_path)

