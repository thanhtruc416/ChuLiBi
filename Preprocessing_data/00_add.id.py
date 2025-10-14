# 00_add_customer_id.py
import pandas as pd
import os

# Đường dẫn
INPUT_PATH = "../Dataset/Input/Customer online delivery dataset - Customer_data.csv"
OUTPUT_DIR = "../Dataset/Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Đọc dữ liệu gốc
df = pd.read_csv(INPUT_PATH)
print(f"Đã đọc file gốc: {df.shape}")

# 2. Tạo cột CustomerID dạng CUS001, CUS002, ...
df.insert(0, "CustomerID", [f"CUS{i:03d}" for i in range(1, len(df) + 1)])

# 3. Lưu ra Output
output_path = os.path.join(OUTPUT_DIR, "Customer_data_with_ID.csv")
df.to_csv(output_path, index=False)
print(f"Đã lưu file có CustomerID: {output_path}")
print(df.head())