# Function/predict_customer.py
"""
Hàm tích hợp để chạy xuyên suốt các tính năng:
- Phân cụm (Clustering)
- Dự đoán tỷ lệ rời bỏ (Churn Prediction)
- Expected Loss
- Recommend gói cho 1 khách hàng

Sử dụng các models và pipelines đã có sẵn từ các file chức năng.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ==================== PATH SETUP ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Dataset" / "Output"

# ==================== IMPORT CÁC MODULE ====================
# Import các hàm từ các file chức năng
# Sử dụng relative import để hoạt động khi chạy từ bất kỳ đâu
try:
    # Thử import relative (khi chạy như module)
    from .Frame07_Cluster import (
        attach_group_pca, select_X, kmeans_labels, CLUSTER_FEATURES, LIKERT_GROUPS
    )
    from .Frame08_churn import (
        create_proxy_churn, prepare_core_df, detect_leakage, preprocess
    )
    # Không cần import compute_expected_loss vì tự tính trong predict_expected_loss
    # from .Frame09_EL import compute_expected_loss
    from .Frame11_Recommend import (
        get_signals, match_score, eligible, compute_best_action, 
        ACTION_LIB, ACTION_LIB_EXT, ACTION_CHANNEL, ID_CANDS
    )
except ImportError:
    # Fallback: import trực tiếp (khi chạy từ Function folder hoặc như script)
    from Frame07_Cluster import (
        attach_group_pca, select_X, kmeans_labels, CLUSTER_FEATURES, LIKERT_GROUPS
    )
    from Frame08_churn import (
        create_proxy_churn, prepare_core_df, detect_leakage, preprocess
    )
    # Không cần import compute_expected_loss vì tự tính trong predict_expected_loss
    # from Frame09_EL import compute_expected_loss
    from Frame11_Recommend import (
        get_signals, match_score, eligible, compute_best_action, 
        ACTION_LIB, ACTION_LIB_EXT, ACTION_CHANNEL, ID_CANDS
    )

# ==================== CLASS PREDICT CUSTOMER ====================
class CustomerPredictor:
    """
    Class để dự đoán cluster, churn, expected loss và recommend gói cho 1 khách hàng
    """
    
    def __init__(self, data_dir: Path = None):
        """
        Khởi tạo predictor
        
        Args:
            data_dir: Đường dẫn đến thư mục Dataset/Output (mặc định: PROJECT_ROOT/Dataset/Output)
        """
        self.data_dir = data_dir or DATA_DIR
        
        # Các file cần thiết
        self.cluster_file = self.data_dir / "df_cluster_full.csv"
        self.scaled_file = self.data_dir / "df_scaled_model.csv"
        self.raw_file = self.data_dir / "df_raw_dashboard.csv"
        self.churn_model_file = self.data_dir / "best_churn_model.pkl"
        self.scaler_file = self.data_dir / "scaler.pkl"
        
        # Load dữ liệu và models nếu có
        self.df_cluster_full = None
        self.df_scaled = None
        self.df_raw = None
        self.churn_model = None
        self.scaler = None
        self.k_final = 3  # số cluster mặc định
        
        self._load_data_and_models()
    
    def _load_data_and_models(self):
        """Load dữ liệu và models đã train sẵn"""
        try:
            if self.cluster_file.exists():
                self.df_cluster_full = pd.read_csv(self.cluster_file)
                print(f"✓ Đã load cluster data: {self.df_cluster_full.shape}")
            
            if self.scaled_file.exists():
                self.df_scaled = pd.read_csv(self.scaled_file)
                print(f"✓ Đã load scaled data: {self.df_scaled.shape}")
            
            if self.raw_file.exists():
                self.df_raw = pd.read_csv(self.raw_file)
                print(f"✓ Đã load raw data: {self.df_raw.shape}")
            
            if self.churn_model_file.exists():
                bundle = joblib.load(self.churn_model_file)
                self.churn_model = bundle.get('model')
                self.churn_threshold = bundle.get('threshold', 0.5)
                print(f"✓ Đã load churn model")
            
            if self.scaler_file.exists():
                self.scaler = joblib.load(self.scaler_file)
                print(f"✓ Đã load scaler")
                
        except Exception as e:
            print(f"⚠ Cảnh báo khi load dữ liệu/models: {e}")
    
    def _preprocess_customer(self, customer: Dict) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu khách hàng để phù hợp với format đã train
        
        Args:
            customer: Dictionary chứa thông tin khách hàng
            
        Returns:
            DataFrame đã được preprocess giống dataset gốc
        """
        # Chuyển dict thành DataFrame
        df_cust = pd.DataFrame([customer])
        
        # Nếu không có CustomerID, tạo tạm
        if 'CustomerID' not in df_cust.columns:
            df_cust['CustomerID'] = 'CUSTOMER_NEW'
        
        # Đảm bảo có đủ các cột numeric cơ bản
        numeric_cols = [
            'Age', 'Family size', 'Restaurant Rating', 'Delivery Rating',
            'No. of orders placed', 'Delivery Time', 'Order Value'
        ]
        
        for col in numeric_cols:
            if col not in df_cust.columns:
                # Lấy giá trị trung bình từ dataset gốc nếu có
                if self.df_raw is not None and col in self.df_raw.columns:
                    df_cust[col] = self.df_raw[col].mean()
                else:
                    df_cust[col] = 0.5  # Giá trị mặc định
        
        # Đảm bảo có các cột Likert encoded
        likert_cols = [
            'Ease and convenient_encoded', 'Self Cooking_encoded', 
            'Health Concern_encoded', 'Late Delivery_encoded',
            'Poor Hygiene_encoded', 'Bad past experience_encoded',
            'More Offers and Discount_encoded'
        ]
        
        for col in likert_cols:
            if col not in df_cust.columns:
                # Tìm cột gốc (không có _encoded)
                base_col = col.replace('_encoded', '')
                if base_col in df_cust.columns:
                    # Nếu có giá trị text, encode
                    likert_map = {
                        'strongly disagree': 1, 'disagree': 2, 'neutral': 3,
                        'agree': 4, 'strongly agree': 5
                    }
                    val = df_cust[base_col].iloc[0]
                    if isinstance(val, str):
                        df_cust[col] = likert_map.get(val.lower().strip(), 3)
                    else:
                        df_cust[col] = val if 1 <= val <= 5 else 3
                else:
                    df_cust[col] = 3  # neutral
        
        # Các cột encoded khác
        if 'Maximum wait time_encoded' not in df_cust.columns:
            wait_map = {'30 minutes': 1, '45 minutes': 2, '60 minutes': 3, 'more than 60 minutes': 4}
            if 'Maximum wait time' in df_cust.columns:
                val = df_cust['Maximum wait time'].iloc[0]
                df_cust['Maximum wait time_encoded'] = wait_map.get(val.lower().strip(), 2)
            else:
                df_cust['Maximum wait time_encoded'] = 2
        
        if 'Influence of rating_encoded' not in df_cust.columns:
            inf_map = {'no': 1, 'maybe': 2, 'yes': 3}
            if 'Influence of rating' in df_cust.columns:
                val = df_cust['Influence of rating'].iloc[0]
                df_cust['Influence of rating_encoded'] = inf_map.get(val.lower().strip(), 2)
            else:
                df_cust['Influence of rating_encoded'] = 2
        
        return df_cust
    
    def _scale_customer(self, df_cust: pd.DataFrame) -> pd.DataFrame:
        """
        Scale dữ liệu khách hàng bằng scaler đã train
        
        Args:
            df_cust: DataFrame khách hàng đã preprocess
            
        Returns:
            DataFrame đã được scale
        """
        if self.scaler is None or self.df_scaled is None:
            print("⚠ Không có scaler, bỏ qua bước scale")
            return df_cust
        
        df_scaled = df_cust.copy()
        
        # Lấy danh sách cột numeric từ dataset gốc
        numeric_cols = [
            'Age', 'Family size', 'Restaurant Rating', 'Delivery Rating',
            'No. of orders placed', 'Delivery Time', 'Order Value'
        ]
        likert_cols = [
            'Ease and convenient_encoded', 'Self Cooking_encoded',
            'Health Concern_encoded', 'Late Delivery_encoded',
            'Poor Hygiene_encoded', 'Bad past experience_encoded',
            'More Offers and Discount_encoded'
        ]
        other_encoded = ['Maximum wait time_encoded', 'Influence of rating_encoded']
        
        all_numeric = numeric_cols + likert_cols + other_encoded
        all_numeric = [c for c in all_numeric if c in df_scaled.columns]
        
        # Scale các cột numeric
        if all_numeric:
            df_scaled[all_numeric] = self.scaler.transform(df_scaled[all_numeric])
        
        return df_scaled
    
    def predict_cluster(self, customer: Dict) -> int:
        """
        Dự đoán cluster cho khách hàng
        
        Args:
            customer: Dictionary chứa thông tin khách hàng
            
        Returns:
            Cluster ID (0, 1, 2, ...)
        """
        # Preprocess customer
        df_cust = self._preprocess_customer(customer)
        df_cust_scaled = self._scale_customer(df_cust)
        
        # Tạo PCA groups nếu cần
        df_cust_with_pca = attach_group_pca(df_cust_scaled, random_state=42)
        
        # Chọn features cho clustering
        features = [c for c in CLUSTER_FEATURES if c in df_cust_with_pca.columns]
        if len(features) < 2:
            print("⚠ Không đủ features để phân cụm, trả về cluster 0")
            return 0
        
        X_cust = select_X(df_cust_with_pca, features)
        
        # Nếu có df_cluster_full, dùng để train KMeans và predict
        if self.df_cluster_full is not None and len(self.df_cluster_full) > 0:
            # Load toàn bộ dữ liệu cluster để train KMeans
            # Không cần import run_pipeline vì đã có đủ hàm từ đầu file
            
            # Đếm số cluster từ dữ liệu có sẵn
            if 'cluster' in self.df_cluster_full.columns:
                self.k_final = int(self.df_cluster_full['cluster'].nunique())
            
            # Tạo X từ df_cluster_full
            df_cluster_features = attach_group_pca(self.df_cluster_full, random_state=42)
            X_full = select_X(df_cluster_features, features)
            
            # Train KMeans trên toàn bộ data
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.k_final, random_state=42, n_init=20)
            kmeans.fit(X_full)
            
            # Predict cho customer mới
            cluster = kmeans.predict(X_cust)[0]
            return int(cluster)
        else:
            # Fallback: trả về cluster 0 nếu không có dữ liệu
            print("⚠ Không có dữ liệu cluster, trả về cluster 0")
            return 0
    
    def predict_churn(self, customer: Dict, cluster: int = None) -> Tuple[float, int]:
        """
        Dự đoán churn probability và churn prediction cho khách hàng
        
        Args:
            customer: Dictionary chứa thông tin khách hàng
            cluster: Cluster ID (nếu đã predict trước đó)
            
        Returns:
            Tuple (proba_churn, pred_churn): Xác suất churn và dự đoán churn (0 hoặc 1)
        """
        if self.churn_model is None:
            print("⚠ Không có churn model, sử dụng giá trị mặc định")
            return 0.5, 0
        
        # Preprocess customer
        df_cust = self._preprocess_customer(customer)
        df_cust_scaled = self._scale_customer(df_cust)
        
        # Tạo PCA groups
        df_cust_with_pca = attach_group_pca(df_cust_scaled, random_state=42)
        
        # Thêm cluster nếu chưa có
        if 'cluster' not in df_cust_with_pca.columns:
            if cluster is not None:
                df_cust_with_pca['cluster'] = cluster
            else:
                df_cust_with_pca['cluster'] = self.predict_cluster(customer)
        
        # Tạo proxy churn (cần merge với df_raw để có đủ thông tin)
        df_with_churn = create_proxy_churn(df_cust_with_pca.copy())
        
        # Prepare core features như trong Frame08
        leak_cols = detect_leakage(df_with_churn)
        df_core = prepare_core_df(df_with_churn, leak_cols=leak_cols)
        
        # Preprocess để có X (features)
        X, y, _ = preprocess(df_core)
        
        # Predict với model
        try:
            proba_churn = self.churn_model.predict_proba(X)[0, 1]
            pred_churn = (proba_churn >= self.churn_threshold).astype(int)
            return float(proba_churn), int(pred_churn)
        except Exception as e:
            print(f"⚠ Lỗi khi predict churn: {e}")
            return 0.5, 0
    
    def predict_expected_loss(self, customer: Dict, proba_churn: float, 
                             cluster: int = None) -> Dict[str, float]:
        """
        Tính Expected Loss cho khách hàng
        
        Args:
            customer: Dictionary chứa thông tin khách hàng
            proba_churn: Xác suất churn đã dự đoán
            cluster: Cluster ID (optional)
            
        Returns:
            Dictionary chứa các metrics Expected Loss
        """
        # Lấy Order Value
        order_value = customer.get('Order Value', 0.5)
        if order_value <= 0:
            # Lấy trung bình từ dataset nếu có
            if self.df_raw is not None and 'Order Value' in self.df_raw.columns:
                order_value = self.df_raw['Order Value'].mean()
            else:
                order_value = 2.0  # Giá trị mặc định (1-3 scale)
        
        # Scale Order Value (MinMaxScaler 0-1)
        # Giả sử Order Value nằm trong khoảng 1-3
        order_value_scaled = (order_value - 1) / (3 - 1) if order_value >= 1 else order_value / 3
        
        # Expected Loss Score = proba_churn × OrderValue_scaled
        expected_loss_score = proba_churn * order_value_scaled
        
        # Expected Loss Real = proba_churn × Order Value
        expected_loss_real = proba_churn * order_value
        
        return {
            'Order Value': float(order_value),
            'OrderValue_scaled': float(order_value_scaled),
            'proba_churn': float(proba_churn),
            'ExpectedLoss_score': float(expected_loss_score),
            'ExpectedLoss_real': float(expected_loss_real),
        }
    
    def recommend_action(self, customer: Dict, proba_churn: float, 
                        expected_loss: Dict[str, float], cluster: int = None) -> Dict:
        """
        Recommend action (gói) cho khách hàng
        
        Args:
            customer: Dictionary chứa thông tin khách hàng
            proba_churn: Xác suất churn
            expected_loss: Dictionary Expected Loss metrics
            cluster: Cluster ID (optional)
            
        Returns:
            Dictionary chứa thông tin recommendation
        """
        # Preprocess customer để có đủ signals
        df_cust = self._preprocess_customer(customer)
        df_cust_scaled = self._scale_customer(df_cust)
        df_cust_with_pca = attach_group_pca(df_cust_scaled, random_state=42)
        
        # Thêm thông tin cần thiết vào row
        row = df_cust_with_pca.iloc[0].copy()
        row['proba_churn'] = proba_churn
        row['EL_norm'] = expected_loss.get('ExpectedLoss_score', 0.0)
        row['cluster'] = cluster if cluster is not None else self.predict_cluster(customer)
        
        # Tính EL_norm (normalized Expected Loss score)
        # Nếu có dataset gốc, tính percentile
        if self.df_cluster_full is not None and 'ExpectedLoss_score' in self.df_cluster_full.columns:
            el_scores = self.df_cluster_full['ExpectedLoss_score'].dropna()
            if len(el_scores) > 0:
                percentile = (el_scores <= row['EL_norm']).sum() / len(el_scores)
                row['EL_norm'] = percentile
        else:
            # Fallback: normalize từ 0-1
            row['EL_norm'] = min(max(row['EL_norm'], 0), 1)
        
        # Chọn action tốt nhất
        best_action = compute_best_action(row, eligible, ACTION_LIB)
        
        if not best_action or best_action['priority_score'] <= 0:
            # Fallback to light actions
            churn_fallback_thr = 0.05
            fallback_id = "REMIND_APP" if proba_churn >= churn_fallback_thr else "EDU_CONTENT"
            light_action = [a for a in ACTION_LIB_EXT if a["id"] == fallback_id]
            best_action = compute_best_action(row, lambda aid, s: True, light_action)
        
        if not best_action:
            best_action = {
                "action_id": "NO_ACTION",
                "action_name": "Không gửi – không đủ điều kiện / ưu tiên thấp",
                "match": 0.0,
                "lift_est": 0.0,
                "priority_score": 0.0
            }
        
        # Lấy channel và template
        channel, template = ACTION_CHANNEL.get(best_action["action_id"], ("App", ""))
        
        return {
            'action_id': best_action["action_id"],
            'action_name': best_action["action_name"],
            'priority_score': float(best_action["priority_score"]),
            'match_score': float(best_action["match"]),
            'lift_estimate': float(best_action["lift_est"]),
            'channel': channel,
            'template': template
        }
    
    def predict_all(self, customer: Dict) -> Dict:
        """
        Chạy xuyên suốt tất cả các bước: Cluster → Churn → Expected Loss → Recommend
        
        Args:
            customer: Dictionary chứa thông tin khách hàng
                Các trường bắt buộc/tùy chọn:
                - Age, Family size
                - Restaurant Rating, Delivery Rating
                - No. of orders placed, Delivery Time, Order Value
                - Các cột Likert (Ease and convenient, Self Cooking, Health Concern, etc.)
                - Maximum wait time, Influence of rating
                
        Returns:
            Dictionary chứa toàn bộ kết quả:
            {
                'cluster': int,
                'churn': {
                    'proba_churn': float,
                    'pred_churn': int,
                    'churn_risk_pct': str
                },
                'expected_loss': {
                    'Order Value': float,
                    'ExpectedLoss_score': float,
                    'ExpectedLoss_real': float,
                },
                'recommendation': {
                    'action_id': str,
                    'action_name': str,
                    'priority_score': float,
                    'channel': str,
                    'template': str
                }
            }
        """
        # Lưu input thô của khách hàng vào CSV (append)
        try:
            save_path = self.data_dir / "predict_new_customer.csv"
            df_input = pd.DataFrame([customer])
            # Append nếu file đã tồn tại, viết header nếu chưa có
            write_header = not save_path.exists()
            df_input.to_csv(save_path, index=False, mode='a', header=write_header, encoding='utf-8')
        except Exception as e:
            print(f"⚠ Không thể lưu input vào predict_new_customer.csv: {e}")

        print("\n" + "="*60)
        print("BẮT ĐẦU DỰ ĐOÁN CHO KHÁCH HÀNG")
        print("="*60)
        
        # 1. Predict Cluster
        print("\n[1/4] Đang phân cụm khách hàng...")
        cluster = self.predict_cluster(customer)
        print(f"✓ Cluster: {cluster}")
        
        # 2. Predict Churn
        print("\n[2/4] Đang dự đoán tỷ lệ rời bỏ...")
        proba_churn, pred_churn = self.predict_churn(customer, cluster=cluster)
        churn_risk_pct = f"{proba_churn * 100:.1f}%"
        print(f"✓ Xác suất churn: {churn_risk_pct}")
        print(f"✓ Dự đoán churn: {'Có nguy cơ' if pred_churn == 1 else 'Không rời bỏ'}")
        
        # 3. Expected Loss
        print("\n[3/4] Đang tính Expected Loss...")
        expected_loss = self.predict_expected_loss(customer, proba_churn, cluster=cluster)
        print(f"✓ Expected Loss Score: {expected_loss['ExpectedLoss_score']:.4f}")
        print(f"✓ Expected Loss Real: {expected_loss['ExpectedLoss_real']:.2f}")
        
        # 4. Recommendation
        print("\n[4/4] Đang gợi ý gói hành động...")
        recommendation = self.recommend_action(customer, proba_churn, expected_loss, cluster=cluster)
        print(f"✓ Gói đề xuất: {recommendation['action_name']}")
        print(f"✓ Kênh: {recommendation['channel']}")
        
        print("\n" + "="*60)
        print("HOÀN TẤT DỰ ĐOÁN")
        print("="*60 + "\n")
        
        return {
            'cluster': cluster,
            'churn': {
                'proba_churn': proba_churn,
                'pred_churn': pred_churn,
                'churn_risk_pct': churn_risk_pct
            },
            'expected_loss': expected_loss,
            'recommendation': recommendation
        }


# ==================== HÀM TIỆN ÍCH ====================
def predict_customer(customer: Dict, data_dir: Path = None) -> Dict:
    """
    Hàm tiện ích để dự đoán cho 1 khách hàng
    
    Args:
        customer: Dictionary chứa thông tin khách hàng
        data_dir: Đường dẫn đến Dataset/Output (optional)
        
    Returns:
        Dictionary chứa kết quả dự đoán đầy đủ
        ví dụ:
        >>> customer = {
        ...     'Age': 20,
        ...     'Gender': 'Female',
        ...     'Marital Status': 'Single',
        ...     'Occupation': 'Student',
        ...     'Educational Qualifications': 'Post Graduate',
        ...     'Family size': 4,
        ...     'Frequently used Medium': 'Food delivery apps',
        ...     'Frequently ordered Meal category': 'Breakfast',
        ...     'Perference': 'Non Veg foods (Lunch / Dinner)',
        ...     'Restaurant Rating': 1,
        ...     'Delivery Rating': 1,
        ...     'No. of orders placed': 150,
        ...     'Delivery Time': 45,
        ...     'Order Value': 1,
        ...     'Ease and convenient': 'Neutral',
        ...     'Self Cooking': 'Neutral',
        ...     'Health Concern': 'Neutral',
        ...     'Late Delivery': 'Neutral',
        ...     'Poor Hygiene': 'Neutral',
        ...     'Bad past experience': 'Neutral',
        ...     'More Offers and Discount': 'Neutral',
        ...     'Maximum wait time': '30 minutes',
        ...     'Influence of rating': 'Yes'
        ... }
        >>> result = predict_customer(customer)
        >>> print(result['cluster'])
        >>> print(result['churn']['churn_risk_pct'])
        >>> print(result['recommendation']['action_name'])
    """
    predictor = CustomerPredictor(data_dir=data_dir)
    return predictor.predict_all(customer)


# ==================== MAIN ====================
if __name__ == "__main__":
    # Nhập tay thông tin khách hàng từ console
    def _inp(prompt: str, default: str = "") -> str:
        val = input(f"{prompt} [{default}]: ").strip()
        return val if val != "" else default

    print("\nNhập thông tin khách hàng (Enter để dùng giá trị mặc định)")
    cust = {}
    # Các trường thô theo Frame09_predict.py (586-608)
    cust['Age'] = _inp("Age (số)")
    cust['Gender'] = _inp("Gender (Male/Female)")
    cust['Marital Status'] = _inp("Marital Status (Single/Married/...) ")
    cust['Occupation'] = _inp("Occupation")
    cust['Educational Qualifications'] = _inp("Educational Qualifications")
    cust['Family size'] = _inp("Family size (số)")
    cust['Frequently used Medium'] = _inp("Frequently used Medium")
    cust['Frequently ordered Meal category'] = _inp("Frequently ordered Meal category")
    cust['Perference'] = _inp("Perference")
    cust['Restaurant Rating'] = _inp("Restaurant Rating (1-5)")
    cust['Delivery Rating'] = _inp("Delivery Rating (1-5)")
    cust['No. of orders placed'] = _inp("No. of orders placed (số)")
    cust['Delivery Time'] = _inp("Delivery Time (phút)")
    cust['Order Value'] = _inp("Order Value (thang 1-3 hoặc số)")
    cust['Ease and convenient'] = _inp("Ease and convenient (Strongly disagree/Disagree/Neutral/Agree/Strongly agree)")
    cust['Self Cooking'] = _inp("Self Cooking (Likert)")
    cust['Health Concern'] = _inp("Health Concern (Likert)")
    cust['Late Delivery'] = _inp("Late Delivery (Likert)")
    cust['Poor Hygiene'] = _inp("Poor Hygiene (Likert)")
    cust['Bad past experience'] = _inp("Bad past experience (Likert)")
    cust['More Offers and Discount'] = _inp("More Offers and Discount (Likert)")
    cust['Maximum wait time'] = _inp("Maximum wait time (30 minutes/45 minutes/60 minutes/more than 60 minutes)")
    cust['Influence of rating'] = _inp("Influence of rating (No/Maybe/Yes)")

    # Chuyển kiểu cơ bản cho các trường số (nếu người dùng nhập số)
    def _to_int(x):
        try:
            return int(float(x))
        except Exception:
            return x
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return x

    cust['Age'] = _to_int(cust['Age'])
    cust['Family size'] = _to_int(cust['Family size'])
    cust['No. of orders placed'] = _to_int(cust['No. of orders placed'])
    cust['Delivery Time'] = _to_int(cust['Delivery Time'])
    cust['Restaurant Rating'] = _to_float(cust['Restaurant Rating'])
    cust['Delivery Rating'] = _to_float(cust['Delivery Rating'])
    cust['Order Value'] = _to_float(cust['Order Value'])

    result = predict_customer(cust)

    print("\n=== KẾT QUẢ DỰ ĐOÁN ===")
    print(f"Cluster: {result['cluster']}")
    print(f"Churn Risk: {result['churn']['churn_risk_pct']}")
    print(f"Expected Loss Real: {result['expected_loss']['ExpectedLoss_real']:.2f}")
    print(f"Đề xuất: {result['recommendation']['action_name']}")
    print(f"Kênh: {result['recommendation']['channel']}")

