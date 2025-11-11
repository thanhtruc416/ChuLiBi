# Function/db.py
import os
import pymysql

class Connector:
    def __init__(self, server="localhost", port=3306,
                 database="chulibi", username="root", password="@Obama123"):
        self.server = server
        self.port = port
        self.database = database
        self.username = username
        self.password = password

    def connect(self):
        """Mở 1 connection mới; nhớ .close() (dùng with là an toàn nhất)."""
        return pymysql.connect(
            host=self.server,
            port=int(self.port),
            user=self.username,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def test(self):
        """Check nhanh thông số đang kết nối."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DATABASE() db, @@hostname host, @@port port;")
                return cur.fetchone()

# Lấy thông số từ ENV (không hard-code mật khẩu trong code)
connector = Connector(
    server   = os.getenv("DB_HOST", "127.0.0.1"),
    port     = int(os.getenv("DB_PORT", "3306")),
    database = os.getenv("DB_NAME", "chulibi"),
    username = os.getenv("DB_USER", "root"),
    password = os.getenv("DB_PASSWORD", "@Obama123"),
)

def get_conn():
    """Hàm tiện lợi để import ở nơi khác: from Function.db import get_conn"""
    return connector.connect()
