# -*- coding: utf-8 -*-
# File: Function/Frame02_Create.py
# Đăng ký tài khoản cho Frame02 — xử lý lệch múi giờ OTP 1 lần cho tất cả dữ liệu.

from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import re
import bcrypt
from Function.db import get_conn

# ===== Cấu hình & validate =====
MIN_PW_LEN = 6
UTC = timezone.utc
LOCAL_TZ = timezone(timedelta(hours=7))   # Asia/Ho_Chi_Minh

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_EMAIL_RE    = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# ---- Password policy ----
MIN_PW_LEN = 8
MAX_PW_LEN = 128
_WEAK_LIST = {
    "123456","12345678","123456789","1234567890","111111","000000",
    "password","pass123","qwerty","abc123","iloveyou","admin","letmein"
}
_SPECIALS = r"!@#$%^&*()\-_=+\[\]{};:'\",.<>/?\\|`~"

def _pw_check(pw: str):
    """
    Trả về (ok, msg). ok=False => msg mô tả lỗi cụ thể.
    Chính sách:
      - 8–128 ký tự
      - Có ít nhất 1 chữ thường, 1 chữ hoa, 1 số, 1 ký tự đặc biệt
      - Không chứa khoảng trắng
      - Không thuộc danh sách mật khẩu yếu phổ biến
    """
    if not isinstance(pw, str):
        return False, "Mật khẩu không hợp lệ"

    if len(pw) < MIN_PW_LEN or len(pw) > MAX_PW_LEN:
        return False, f"Mật khẩu phải từ {MIN_PW_LEN} đến {MAX_PW_LEN} ký tự"

    if any(c.isspace() for c in pw):
        return False, "Mật khẩu không được chứa khoảng trắng"

    low = any(c.islower() for c in pw)
    up  = any(c.isupper() for c in pw)
    dig = any(c.isdigit() for c in pw)
    sp  = any(c in _SPECIALS for c in pw)

    if not (low and up and dig and sp):
        return False, "Mật khẩu phải có chữ thường, chữ hoa, số và ký tự đặc biệt"

    if pw.lower() in _WEAK_LIST:
        return False, "Mật khẩu quá phổ biến, vui lòng chọn mật khẩu mạnh hơn"

    return True, ""


def _now_utc():
    return datetime.now(UTC)

def _is_valid_username(u: str) -> bool:
    return bool(u) and _USERNAME_RE.match(u)

def _is_valid_email(e: str) -> bool:
    return bool(e) and len(e) <= 254 and _EMAIL_RE.match(e)

def _pw_ok(pw: str) -> bool:
    return isinstance(pw, str) and len(pw) >= MIN_PW_LEN

# ===== DB helpers =====
def _fetch_latest_otp_row(email: str) -> Optional[dict]:
    """OTP mới nhất theo id giảm dần (tránh vớ bản cũ)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, email, otp_hash, expires_at, IFNULL(used,0) AS used, created_at
                FROM password_resets
                WHERE email=%s
                ORDER BY id DESC
                LIMIT 1
                """,
                (email.lower(),)
            )
            row = cur.fetchone()
    if not row:
        return None
    if isinstance(row, dict):
        return row
    keys = ["id", "email", "otp_hash", "expires_at", "used", "created_at"]
    return dict(zip(keys, row))

def _mark_otp_used(otp_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE password_resets SET used=1 WHERE id=%s", (otp_id,))
        conn.commit()

def _email_exists(email: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM user_data WHERE email=%s LIMIT 1", (email.lower(),))
            return cur.fetchone() is not None

def _username_exists(username: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM user_data WHERE username=%s LIMIT 1", (username,))
            return cur.fetchone() is not None

def _create_user(username: str, email: str, password: str) -> Dict[str, Any]:
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_data (username, password_hash, email, is_active, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (username, pw_hash, email.lower(), 1, _now_utc().strftime("%Y-%m-%d %H:%M:%S"))
            )
            user_id = cur.lastrowid
        conn.commit()
    return {
        "id": user_id,
        "username": username,
        "email": email.lower(),
        "is_active": 1,
        "full_name": None,
        "business_name": None,
        "role": None,
    }

# ===== Parse thời gian an toàn (vá lệch UTC/+07) =====
def _parse_as(dt_val, tz_assume):
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        d = dt_val
    else:
        try:
            d = datetime.fromisoformat(str(dt_val))
        except Exception:
            d = datetime.strptime(str(dt_val), "%Y-%m-%d %H:%M:%S")
    if d.tzinfo is None:
        d = d.replace(tzinfo=tz_assume)
    return d.astimezone(UTC)

def _safe_expiry_utc(raw_exp):
    """
    Cho 1 giá trị expires_at (naive), parse theo 2 giả định:
    - expires_at là UTC
    - expires_at là +07
    Lấy mốc MUỘN HƠN để tránh hết hạn oan khi dữ liệu lịch sử lẫn lộn.
    """
    exp_utc   = _parse_as(raw_exp, UTC)
    exp_local = _parse_as(raw_exp, LOCAL_TZ)
    candidates = [x for x in (exp_utc, exp_local) if x is not None]
    return max(candidates) if candidates else None

# ===== Public API =====
class AuthService:
    @staticmethod
    def register_user(username: str, email: str, password: str, confirm_password: str, otp: str) -> Dict[str, Any]:
        username = (username or "").strip()
        email    = (email or "").strip()
        otp      = (otp or "").strip()

        if not _is_valid_username(username):
            return {"success": False, "message": "Username không hợp lệ (3–32 ký tự: chữ/số/._-)"}
        if not _is_valid_email(email):
            return {"success": False, "message": "Email không hợp lệ (vui lòng quay lại bước trước)"}
        ok_pw, msg_pw = _pw_check(password)
        if not ok_pw:
            return {"success": False, "message": msg_pw}
        if password != confirm_password:
            return {"success": False, "message": "Xác nhận mật khẩu không khớp"}

        if not otp.isdigit() or not (4 <= len(otp) <= 8):
            return {"success": False, "message": "OTP không hợp lệ"}

        if _email_exists(email):
            return {"success": False, "message": "Email đã tồn tại trong hệ thống"}
        if _username_exists(username):
            return {"success": False, "message": "Username đã được sử dụng"}

        row = _fetch_latest_otp_row(email)
        if not row:
            return {"success": False, "message": "Không tìm thấy OTP cho email này. Vui lòng gửi lại OTP."}

        if int(row.get("used", 0) or 0) == 1:
            return {"success": False, "message": "OTP đã được sử dụng. Vui lòng yêu cầu mã mới."}

        # == FIX: chọn hạn OTP an toàn bất chấp lệch timezone ==
        exp_utc = _safe_expiry_utc(row.get("expires_at"))
        if exp_utc is None:
            return {"success": False, "message": "Không xác định được hạn OTP. Vui lòng gửi lại mã."}
        if _now_utc() > exp_utc:
            return {"success": False, "message": "OTP đã hết hạn. Vui lòng gửi lại OTP."}

        # Kiểm OTP
        otp_hash = (row.get("otp_hash") or "").encode()
        try:
            if not otp_hash or not bcrypt.checkpw(otp.encode(), otp_hash):
                return {"success": False, "message": "OTP không đúng"}
        except Exception:
            return {"success": False, "message": "OTP không đúng"}

        # Tạo user + mark used
        user_data = _create_user(username, email, password)
        try:
            _mark_otp_used(int(row["id"]))
        except Exception:
            pass

        return {"success": True, "message": "Tạo tài khoản thành công", "user_data": user_data}
