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
from typing import Dict, Tuple
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==================== PATH SETUP ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Dataset" / "Output"

# ==================== IMPORT CÁC MODULE ====================
# Import các hàm từ các file chức năng
# Sử dụng relative import để hoạt động khi chạy từ bất kỳ đâu
try:
    # Thử import relative (khi chạy như module)
    from Function.Frame07_Cluster import (
        attach_group_pca, select_X, kmeans_labels, CLUSTER_FEATURES, LIKERT_GROUPS
    )
    from Function.Frame08_churn import (
        create_proxy_churn, prepare_core_df, detect_leakage, preprocess
    )
    # Không cần import compute_expected_loss vì tự tính trong predict_expected_loss
    # from .Frame09_EL import compute_expected_loss
    from Function.Frame11_Recommend import (
        get_signals, match_score, eligible, compute_best_action,
        ACTION_LIB, ACTION_LIB_EXT, ACTION_CHANNEL, ID_CANDS
    )
except ImportError:
    # Fallback: import trực tiếp (khi chạy từ Function folder hoặc như script)
    from Function.Frame07_Cluster import (
        attach_group_pca, select_X, kmeans_labels, CLUSTER_FEATURES, LIKERT_GROUPS
    )
    from Function.Frame08_churn import (
        create_proxy_churn, prepare_core_df, detect_leakage, preprocess
    )
    # Không cần import compute_expected_loss vì tự tính trong predict_expected_loss
    # from Frame09_EL import compute_expected_loss
    from Function.Frame10_Recommend import (
        get_signals, match_score, eligible, compute_best_action,
        ACTION_LIB, ACTION_LIB_EXT, ACTION_CHANNEL, ID_CANDS
    )
# ==================== VALIDATION FUNCTION ====================
def validate_customer_input(customer: Dict) -> Tuple[bool, list]:
    """
    Kiểm tra tính hợp lệ của dữ liệu khách hàng.
    Trả về (is_valid, error_messages)
    """
    errors = []
    # Delivery Time
    try:
        t = float(customer.get("Delivery Time", 0))
        if t <= 0 or t > 90:
            errors.append("Delivery Time phải trong khoảng 1–90 phút.")
    except ValueError:
        errors.append("Delivery Time phải là số.")
    # Tuổi
    try:
        age = float(customer.get("Age", 0))
        if age <= 0 or age > 100:
            errors.append(" Age phải nằm trong khoảng 1–120.")
    except ValueError:
        errors.append(" Age phải là số.")

    # Family size
    try:
        fam = int(customer.get("Family size", 0))
        if fam < 1 or fam > 10:
            errors.append(" Family size phải từ 1–10.")
    except ValueError:
        errors.append(" Family size phải là số nguyên.")

    # Restaurant Rating
    try:
        r = float(customer.get("Restaurant Rating", 0))
        if r < 1 or r > 5:
            errors.append(" Restaurant Rating phải trong khoảng 1–5.")
    except ValueError:
        errors.append(" Restaurant Rating phải là số.")

    # Delivery Rating
    try:
        d = float(customer.get("Delivery Rating", 0))
        if d < 1 or d > 5:
            errors.append(" Delivery Rating phải trong khoảng 1–5.")
    except ValueError:
        errors.append(" Delivery Rating phải là số.")

    # No. of orders placed
    try:
        n = int(customer.get("No. of orders placed", 0))
        if n <= 0:
            errors.append(" Số đơn hàng phải lớn hơn 0.")
    except ValueError:
        errors.append(" No. of orders placed phải là số.")

    return len(errors) == 0, errors
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
        # ==== CHUYỂN CÁC GIÁ TRỊ SỐ TỪ CHUỖI SANG FLOAT/INT NẾU CẦN ====
        numeric_fields = [
            'Age', 'Family size', 'Restaurant Rating', 'Delivery Rating',
            'No. of orders placed', 'Delivery Time', 'Order Value'
        ]
        for col in numeric_fields:
            if col in df_cust.columns:
                try:
                    # convert text như "25" -> 25.0, "" -> NaN
                    df_cust[col] = pd.to_numeric(df_cust[col], errors='coerce')
                except Exception:
                    df_cust[col] = np.nan

        # điền giá trị mặc định nếu bị NaN
        df_cust[numeric_fields] = df_cust[numeric_fields].fillna(0.0)

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
                if isinstance(val, str):
                    df_cust['Maximum wait time_encoded'] = wait_map.get(val.lower().strip(), 2)
                elif isinstance(val, (int, float)):
                    # Handle numeric input: convert to string format
                    if val <= 30:
                        df_cust['Maximum wait time_encoded'] = 1
                    elif val <= 45:
                        df_cust['Maximum wait time_encoded'] = 2
                    elif val <= 60:
                        df_cust['Maximum wait time_encoded'] = 3
                    else:
                        df_cust['Maximum wait time_encoded'] = 4
                else:
                    df_cust['Maximum wait time_encoded'] = 2
            else:
                df_cust['Maximum wait time_encoded'] = 2

        if 'Influence of rating_encoded' not in df_cust.columns:
            inf_map = {'no': 1, 'maybe': 2, 'yes': 3, 'low': 1, 'medium': 2, 'high': 3, 'very high': 3}
            if 'Influence of rating' in df_cust.columns:
                val = df_cust['Influence of rating'].iloc[0]
                if isinstance(val, str):
                    df_cust['Influence of rating_encoded'] = inf_map.get(val.lower().strip(), 2)
                else:
                    df_cust['Influence of rating_encoded'] = val if 1 <= val <= 3 else 2
            else:
                df_cust['Influence of rating_encoded'] = 2

        # ===== ONE-HOT ENCODING FOR CATEGORICAL VARIABLES =====
        # Danh sách các cột categorical cần encode (theo 02_encoding.py)
        cat_cols = [
            'Gender', 'Marital Status', 'Occupation', 'Educational Qualifications',
            'Frequently used Medium', 'Frequently ordered Meal category', 'Perference'
        ]

        # Lọc chỉ các cột tồn tại
        cat_cols_exist = [c for c in cat_cols if c in df_cust.columns]

        if cat_cols_exist:
            # QUAN TRỌNG: Không dùng drop_first=True cho single row vì sẽ không tạo cột nào!
            # Thay vào đó, tạo tất cả các cột one-hot rồi match với training data
            df_cust = pd.get_dummies(df_cust, columns=cat_cols_exist, drop_first=False)
            df_cust = df_cust.replace({True: 1, False: 0})

        # Đảm bảo có đầy đủ các cột one-hot như trong df_scaled_model
        # Load danh sách cột từ df_scaled để đối chiếu
        if self.df_scaled is not None:
            expected_cols = self.df_scaled.columns.tolist()

            # Lấy danh sách các cột one-hot có trong expected (để biết drop first của từng category)
            expected_onehot = [c for c in expected_cols if any(x in c for x in
                                                               ['Gender_', 'Marital Status_', 'Occupation_',
                                                                'Educational Qualifications_',
                                                                'Frequently used Medium_',
                                                                'Frequently ordered Meal category_', 'Perference_'])]

            # Xóa các cột one-hot KHÔNG có trong expected (đó là các cột bị drop_first)
            current_onehot = [c for c in df_cust.columns if any(x in c for x in
                                                                ['Gender_', 'Marital Status_', 'Occupation_',
                                                                 'Educational Qualifications_',
                                                                 'Frequently used Medium_',
                                                                 'Frequently ordered Meal category_', 'Perference_'])]

            for col in current_onehot:
                if col not in expected_onehot:
                    df_cust = df_cust.drop(columns=[col])

            # Thêm các cột one-hot thiếu (set = 0)
            for col in expected_cols:
                if col not in df_cust.columns and col != 'CustomerID':
                    df_cust[col] = 0

            # Sắp xếp lại thứ tự cột giống df_scaled (trừ CustomerID để sau)
            ordered_cols = ['CustomerID'] + [c for c in expected_cols if c != 'CustomerID' and c in df_cust.columns]
            df_cust = df_cust[ordered_cols]

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

        # Nếu có df_cluster_full, dùng để train PCA và KMeans
        if self.df_cluster_full is not None and len(self.df_cluster_full) > 0:
            # df_cluster_full đã chứa scaled data nhưng ta cần FIT PCA trên toàn bộ training data
            # rồi TRANSFORM customer mới để có PCA components chính xác

            # Lấy training data KHÔNG có PCA và cluster label
            df_train = self.df_scaled.drop(columns=['CustomerID'],
                                           errors='ignore') if self.df_scaled is not None else None

            if df_train is None or len(df_train) == 0:
                print("Không có dữ liệu training, trả về cluster 0")
                return 0

            # Fit PCA trên TOÀN BỘ training data
            from sklearn.decomposition import PCA
            pca_transformers = {}
            for new_col, cols in LIKERT_GROUPS.items():
                miss = [c for c in cols if c not in df_train.columns]
                if miss:
                    continue
                pca = PCA(n_components=1, random_state=42)
                pca.fit(df_train[cols])
                pca_transformers[new_col] = (pca, cols)

            # TRANSFORM customer với PCA đã fit
            df_cust_with_pca = df_cust_scaled.copy()
            for new_col, (pca, cols) in pca_transformers.items():
                miss = [c for c in cols if c not in df_cust_with_pca.columns]
                if miss:
                    df_cust_with_pca[new_col] = 0.0
                else:
                    comp = pca.transform(df_cust_with_pca[cols])
                    df_cust_with_pca[new_col] = comp

            # Chọn features cho clustering
            features = [c for c in CLUSTER_FEATURES if c in df_cust_with_pca.columns]
            if len(features) < 2:
                print("Không đủ features để phân cụm, trả về cluster 0")
                return 0

            X_cust = select_X(df_cust_with_pca, features)

            # df_cluster_full đã chứa PCA components được tính từ training data
            # Đếm số cluster từ dữ liệu có sẵn
            if 'cluster' in self.df_cluster_full.columns:
                self.k_final = int(self.df_cluster_full['cluster'].nunique())

            # Tạo X từ df_cluster_full (đã có PCA)
            X_full = select_X(self.df_cluster_full, features)

            # Train KMeans trên toàn bộ data
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.k_final, random_state=42, n_init=20)
            kmeans.fit(X_full)

            # Predict cho customer mới
            cluster = kmeans.predict(X_cust)[0]
            return int(cluster) + 1
        else:
            # Fallback: trả về cluster 0 nếu không có dữ liệu
            print("Không có dữ liệu cluster, trả về cluster 0")
            return 1

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
            print("Không có churn model, sử dụng giá trị mặc định")
            return 0.5, 0

        # Preprocess customer
        df_cust = self._preprocess_customer(customer)
        df_cust_scaled = self._scale_customer(df_cust)

        # Tạo PCA groups (PHẢI fit trên training data trước)
        if self.df_scaled is not None and len(self.df_scaled) > 0:
            df_train = self.df_scaled.drop(columns=['CustomerID'], errors='ignore')

            # Fit PCA trên training data
            from sklearn.decomposition import PCA
            pca_transformers = {}
            for new_col, cols in LIKERT_GROUPS.items():
                miss = [c for c in cols if c not in df_train.columns]
                if miss:
                    continue
                pca = PCA(n_components=1, random_state=42)
                pca.fit(df_train[cols])
                pca_transformers[new_col] = (pca, cols)

            # Transform customer với PCA đã fit
            df_cust_with_pca = df_cust_scaled.copy()
            for new_col, (pca, cols) in pca_transformers.items():
                miss = [c for c in cols if c not in df_cust_with_pca.columns]
                if miss:
                    df_cust_with_pca[new_col] = 0.0
                else:
                    comp = pca.transform(df_cust_with_pca[cols])
                    df_cust_with_pca[new_col] = comp
        else:
            # Fallback nếu không có training data
            df_cust_with_pca = attach_group_pca(df_cust_scaled, random_state=42)

        # Thêm cluster nếu chưa có
        if 'cluster' not in df_cust_with_pca.columns:
            if cluster is not None:
                df_cust_with_pca['cluster'] = cluster
            else:
                df_cust_with_pca['cluster'] = self.predict_cluster(customer)

        # CRITICAL FIX: Frame08's preprocess() function re-scales ALL numeric features
        # (including PCA and cluster) with StandardScaler.fit_transform on the BATCH data.
        # We need to replicate this: fit scaler on TRAINING data, then transform customer.

        # Lấy df_cluster_full để làm training data (đã có PCA và cluster)
        if self.df_cluster_full is not None and len(self.df_cluster_full) > 0:
            # Chuẩn bị training data giống Frame08's prepare_core_df
            df_train_full = self.df_cluster_full.copy()

            # Lấy các features cho churn model (bỏ churn và churn_percent nếu có)
            exclude_cols = ['CustomerID', 'churn', 'churn_percent']
            train_features = [c for c in df_train_full.columns if c not in exclude_cols]

            # Lấy ONLY numeric features (Frame08's preprocess scales ALL numeric)
            X_train = df_train_full[train_features].select_dtypes(include=[np.number])

            # Fit scaler trên training data
            batch_scaler = StandardScaler()
            batch_scaler.fit(X_train)

            # Lấy features tương ứng từ customer (đã có PCA và cluster)
            X_cust = df_cust_with_pca[[c for c in X_train.columns if c in df_cust_with_pca.columns]]

            # Transform customer với scaler đã fit trên training data
            X_cust_scaled = pd.DataFrame(
                batch_scaler.transform(X_cust),
                columns=X_cust.columns,
                index=X_cust.index
            )

            X = X_cust_scaled.copy()
        else:
            # Fallback: dùng data chưa scale lại
            X = df_cust_with_pca.copy()

        # Lấy danh sách features mà model thực sự mong đợi
        # Model được train với ImbPipeline có imputer và scaler steps
        try:
            expected_features = None

            # Thử lấy từ imputer (bước đầu tiên)
            if hasattr(self.churn_model, 'named_steps') and 'imputer' in self.churn_model.named_steps:
                imputer = self.churn_model.named_steps['imputer']
                if hasattr(imputer, 'feature_names_in_'):
                    expected_features = list(imputer.feature_names_in_)

            # Thử lấy từ scaler nếu imputer không có
            if expected_features is None and hasattr(self.churn_model,
                                                     'named_steps') and 'scaler' in self.churn_model.named_steps:
                scaler = self.churn_model.named_steps['scaler']
                if hasattr(scaler, 'feature_names_in_'):
                    expected_features = list(scaler.feature_names_in_)

            # Thử predict với một sample nhỏ để xem model mong đợi features gì
            if expected_features is None:
                try:
                    # Thử predict với X hiện tại để xem lỗi
                    _ = self.churn_model.predict_proba(X.iloc[[0]])[0, 1]
                    # Nếu không lỗi, có nghĩa là X đã đúng
                    expected_features = list(X.columns)
                except ValueError as ve:
                    # Parse error message để lấy expected features
                    error_msg = str(ve)
                    if "feature_names" in error_msg.lower() or "unseen" in error_msg.lower():
                        # Thử lấy từ error message
                        print(f"⚠ Không thể lấy feature names từ model, sử dụng features hiện có")
                        expected_features = list(X.columns)
                    else:
                        raise

            # Fallback cuối cùng: sử dụng features hiện có
            if expected_features is None:
                expected_features = list(X.columns)

            # Kiểm tra và loại bỏ features không có trong model
            missing_features = [f for f in expected_features if f not in X.columns]
            if missing_features:
                print(f"Cảnh báo: Thiếu features trong dữ liệu: {missing_features}")

            # Loại bỏ features không có trong model (như pca_service_issue nếu bị loại khi train)
            extra_features = [f for f in X.columns if f not in expected_features]
            if extra_features:
                print(f"Loại bỏ features không có trong model: {extra_features}")

            # Chỉ giữ lại features mà model mong đợi và có trong X
            available_expected = [f for f in expected_features if f in X.columns]
            if len(available_expected) == 0:
                raise ValueError("Không có feature nào khớp với model")

            # Tạo X với đúng features và đúng thứ tự như model mong đợi
            X_for_pred = X[available_expected].copy()

            # Nếu thiếu features, điền giá trị mặc định (median hoặc 0)
            if len(missing_features) > 0:
                print(f"Điền giá trị mặc định cho {len(missing_features)} features thiếu")
                for f in missing_features:
                    X_for_pred[f] = 0.0  # Giá trị mặc định

            # Đảm bảo X_for_pred có đúng thứ tự features như expected_features
            X_for_pred = X_for_pred[[f for f in expected_features if f in X_for_pred.columns]]

        except Exception as e:
            print(f"Lỗi khi lấy expected features: {e}")
            print(f"Thử predict trực tiếp với tất cả features...")
            # Fallback: thử loại bỏ pca_service_issue nếu có
            if 'pca_service_issue' in X.columns:
                print(f"Thử loại bỏ pca_service_issue...")
                X_for_pred = X.drop(columns=['pca_service_issue'], errors='ignore').copy()
            else:
                X_for_pred = X.copy()

        # Predict với model
        try:
            proba_churn = self.churn_model.predict_proba(X_for_pred)[0, 1]
            pred_churn = (proba_churn >= self.churn_threshold).astype(int)
            return float(proba_churn), int(pred_churn)
        except Exception as e:
            print(f"Lỗi khi predict churn: {e}")
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
        try:
            order_value = float(order_value)
        except Exception:
            order_value = 0.5

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
            print(f"Không thể lưu input vào predict_new_customer.csv: {e}")

        print("\n" + "=" * 60)
        print("BẮT ĐẦU DỰ ĐOÁN CHO KHÁCH HÀNG")
        print("=" * 60)

        # 1. Predict Cluster
        print("\n[1/4] Đang phân cụm khách hàng...")
        cluster = self.predict_cluster(customer)
        print(f"Cluster: {cluster}")

        # 2. Predict Churn
        print("\n[2/4] Đang dự đoán tỷ lệ rời bỏ...")
        proba_churn, pred_churn = self.predict_churn(customer, cluster=cluster)
        churn_risk_pct = f"{proba_churn * 100:.1f}%"
        print(f"Xác suất churn: {churn_risk_pct}")
        print(f"Dự đoán churn: {'Có nguy cơ' if pred_churn == 1 else 'Không rời bỏ'}")

        # 3. Expected Loss
        print("\n[3/4] Đang tính Expected Loss...")
        expected_loss = self.predict_expected_loss(customer, proba_churn, cluster=cluster)
        print(f"Expected Loss Score: {expected_loss['ExpectedLoss_score']:.4f}")
        print(f"Expected Loss Real: {expected_loss['ExpectedLoss_real']:.2f}")

        # 4. Recommendation
        print("\n[4/4] Đang gợi ý gói hành động...")
        recommendation = self.recommend_action(customer, proba_churn, expected_loss, cluster=cluster)
        print(f"Gói đề xuất: {recommendation['action_name']}")
        print(f"Kênh: {recommendation['channel']}")

        print("\n" + "=" * 60)
        print("HOÀN TẤT DỰ ĐOÁN")
        print("=" * 60 + "\n")

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
    is_valid, errors = validate_customer_input(customer)
    if not is_valid:
        print("\n DỮ LIỆU KHÔNG HỢP LỆ, DỪNG DỰ ĐOÁN:")
        for e in errors:
            print(e)
        raise ValueError("Dữ liệu khách hàng không hợp lệ. Hãy nhập lại đúng giá trị.")

    # Nếu hợp lệ thì mới chạy model
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
