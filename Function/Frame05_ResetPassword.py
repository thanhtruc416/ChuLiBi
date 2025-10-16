# Function/reset_password.py
from datetime import datetime, timezone, timedelta
import bcrypt
from Function.db import get_conn

# ====== Optional: kiểm tra độ mạnh mật khẩu ======
def _password_ok(pw: str) -> tuple[bool, str]:
    if not pw or len(pw) < 8:
        return False, "Mật khẩu tối thiểu 8 ký tự"
    # muốn chặt hơn thì thêm: chữ hoa/thường/số/ký tự đặc biệt...
    return True, "OK"

# ====== Public API: lấy username từ email (để fill vào UI) ======
def get_username_by_email(email: str):
    """
    Trả (True, username) nếu tìm thấy; ngược lại (False, 'lý do').
    """
    email = (email or "").strip().lower()
    if not email:
        return False, "Thiếu email"

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT username FROM user_data WHERE email=%s", (email,))
                row = cur.fetchone()
        if not row:
            return False, "Không tìm thấy tài khoản với email này"
        return True, row["username"]
    except Exception as e:
        return False, f"Lỗi: {e}"

# ====== Public API: verify OTP + reset password trong 1 hàm ======
def reset_password_with_otp(email: str, otp: str, new_password: str):
    """
    Quy trình:
      1) Tìm user theo email
      2) Lấy OTP mới nhất (used=0)
      3) Kiểm tra hết hạn + so khớp bcrypt
      4) Đánh dấu used=1
      5) Cập nhật password_hash bằng bcrypt(new_password)
    Trả (True, 'Đổi mật khẩu thành công') hoặc (False, 'lý do').
    """
    email = (email or "").strip().lower()
    otp = (otp or "").strip()

    if not email:
        return False, "Thiếu email"
    if not otp:
        return False, "Vui lòng nhập OTP"

    ok, msg = _password_ok(new_password)
    if not ok:
        return False, msg

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # 1) user theo email
                cur.execute("SELECT id FROM user_data WHERE email=%s", (email,))
                user = cur.fetchone()
                if not user:
                    return False, "Không tìm thấy tài khoản với email này"

                user_id = user["id"]

                # 2) OTP mới nhất, chưa dùng
                cur.execute(
                    """SELECT id, otp_hash, expires_at, used
                       FROM password_resets
                       WHERE user_id=%s AND used=0
                       ORDER BY id DESC
                       LIMIT 1""",
                    (user_id,)
                )
                pr = cur.fetchone()
                if not pr:
                    return False, "Không có OTP khả dụng"

                # 3) kiểm tra hạn & so khớp
                expires_at = pr["expires_at"]
                now_utc = datetime.now(timezone.utc)
                if isinstance(expires_at, str):
                    # phòng trường hợp connector trả string
                    expires_at = datetime.fromisoformat(expires_at)
                if now_utc > expires_at.replace(tzinfo=timezone.utc):
                    return False, "OTP đã hết hạn"

                if not bcrypt.checkpw(otp.encode(), pr["otp_hash"].encode()):
                    return False, "OTP không đúng"

                # 4) đánh dấu used=1
                cur.execute("UPDATE password_resets SET used=1 WHERE id=%s", (pr["id"],))

                # 5) cập nhật password
                pw_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                cur.execute(
                    "UPDATE user_data SET password_hash=%s WHERE id=%s",
                    (pw_hash, user_id)
                )

        return True, "Đổi mật khẩu thành công"
    except Exception as e:
        return False, f"Lỗi: {e}"
