🧠 ChuLiBi Machine Learning System
🏁 Giới thiệu

ChuLiBi là hệ thống máy học hỗ trợ doanh nghiệp dịch vụ giao đồ ăn trong việc:
1️⃣ Phân cụm khách hàng
2️⃣ Dự báo hành vi rời bỏ
3️⃣ Đo lường tổn thất
4️⃣ Đề xuất combo đồ ăn tối ưu

Ứng dụng được phát triển bằng ngôn ngữ Python, giao diện đồ họa người dùng với Tkinter, tích hợp các thư viện cốt lõi như Pandas, Numpy, Scikit-learn để xử lý và huấn luyện mô hình Machine Learning.

Các thuật toán sử dụng bao gồm:

K-Means Clustering để phân nhóm khách hàng

PCA (Principal Component Analysis) để giảm chiều dữ liệu

Logistic Regression, Random Forest, XGBoost cho mục tiêu dự báo churn

## 🗂️ Cấu trúc thư mục chính

Dưới đây là cấu trúc thư mục của dự án, trình bày theo định dạng Markdown (dễ đọc và có thể copy/paste):

- `ChuLiBi/`
	- `Dataset/`
		- `Input/`
			- `Customer online delivery dataset - Customer_data.csv`
		- `Output/`
			- `df_raw_dashboard.csv`
			- `df_encoded.csv`
			- `df_scaled_model.csv`
			- `df_cluster_full.csv`
			- `df_cluster_with_ID.csv`
			- `cluster_profile_scaled.csv`
			- `cluster_characteristics_descriptive.csv`
			- `Customer_data_with_ID.csv`
			- `churn_predictions_preview.csv`
			- `expected_loss_by_customer.csv`
			- `expected_loss_cluster_summary.csv`
			- `expected_loss_dual_map.csv`
			- `expected_loss_top50.csv`
			- `expected_loss_top50_display.csv`
			- `feature_importance.csv`
			- `model_comparison.csv`
			- `predict_new_customer.csv`
			- `recommendations.csv`
			- `best_churn_model.pkl`
			- `scaler.pkl`

	- `Font/`
		- `Crimson_Pro/`
			- `static/`
				- `CrimsonPro-Italic-VariableFont_wght.ttf`
				- `CrimsonPro-VariableFont_wght.ttf`
				- `OFL.txt`
				- `README.txt`
			- `CrimsonPro-Italic-VariableFont_wght.ttf`
			- `CrimsonPro-VariableFont_wght.ttf`
		- `Kodchasan/`
			- `Kodchasan-Bold.ttf`
			- `Kodchasan-Italic.ttf`
			- `Kodchasan-Regular.ttf`
			- `Kodchasan-SemiBold.ttf`
			- `OFL.txt`
		- `Rubik_Burned/`
			- `RubikBurned-Regular.ttf`
			- `OFL.txt`
		- `Young_Serif/`
			- `YoungSerif-Regular.ttf`
			- `OFL.txt`

	- `Frame/`
		- `Frame01/` — Đăng nhập
		- `Frame02/` — Đăng ký
		- `Frame03/` — Hoàn thiện hồ sơ
		- `Frame04–05/` — OTP & Reset password
		- `Frame06–10/` — Dashboard, Clustering, Churn, Expected Loss, Recommendation
		- `Frame11–13/` — Dự đoán mới, Hồ sơ cá nhân, Quản lý dữ liệu
		- `__init__.py`

	- `Function/`
		- `app_controller.py`
		- `db.py`
		- `dropdown_profile.py`
		- `user_repository.py`
		- `Frame01_auth.py`
		- `Frame02_Create.py`
		- `Frame03_Profile.py`
		- `Frame04_ForgetPassword.py`
		- `Frame05_ResetPassword.py`
		- `Frame06_chart_dashboard.py`
		- `Frame06_kpi_dashboard.py`
		- `Frame07_Cluster.py`
		- `Frame08_churn.py`
		- `Frame09_EL.py`
		- `Frame10_Recommend.py`
		- `Frame11_Predict.py`
		- `README.md`

	- `Preprocessing_data/`
		- `00_add_id.py`
		- `01_read_clean.py`
		- `02_encoding.py`
		- `03_scale_vif.py`
		- `README.md`

	- `QMess/`
		- `assets/`
		- `Qmess_calling.py`
		- `ui_popup_01.py` … `ui_popup_29.py`
		- `README.md`

	- `.env`
	- `.gitignore`
	- `ChuLiBi_Workflow`
	- `main.py`
	- `README.md`
	- `requirements.txt`

⚙️ Quick Start
1️⃣ Prerequisites
Python 3.12
MySQL Server

2️⃣ Installation
# Clone project
git clone https://github.com/<your-repo>/ChuLiBi.git
cd ChuLiBi

# Create environment
python -m venv venv
venv\Scripts\activate    # (Windows)
source venv/bin/activate # (macOS/Linux)

# Install dependencies
pip install -r requirements.txt

3️⃣ Environment Variables
Tạo file .env trong thư mục gốc (hoặc copy từ .env.copy)
DB_HOST=127.0.0.1
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=chulibi
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
APP_NAME=ChuLiBi Dashboard

4️⃣ Run Application
python main.py
