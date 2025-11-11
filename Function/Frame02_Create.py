from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import re
import bcrypt
from Function.db import get_conn

# ===== Cấu hình & validate =====
MIN_PW_LEN = 8
MAX_PW_LEN = 128
UTC = timezone.utc
LOCAL_TZ = timezone(timedelta(hours=7))   # Asia/Ho_Chi_Minh

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_EMAIL_RE    = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

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
        return False, "Password is invalid."

    if len(pw) < MIN_PW_LEN or len(pw) > MAX_PW_LEN:
        return False, "Password must have at least 8 characters. Include uppercase, lowercase, number and special character."

    if any(c.isspace() for c in pw):
        return False, "Password must not contain spaces."

    low = any(c.islower() for c in pw)
    up  = any(c.isupper() for c in pw)
    dig = any(c.isdigit() for c in pw)
    sp  = any(c in _SPECIALS for c in pw)

    if not (low and up and dig and sp):
        return False, "Password must include uppercase, lowercase, number and special character."

    if pw.lower() in _WEAK_LIST:
        return False, "Password is too common, please choose a stronger password."

    return True, ""


def _now_utc():
    return datetime.now(UTC)


def _is_valid_username(u: str) -> bool:
    return bool(u) and _USERNAME_RE.match(u)


def _is_valid_email(e: str) -> bool:
    return bool(e) and len(e) <= 254 and _EMAIL_RE.match(e)


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
    """Chọn hạn OTP an toàn nhất giữa UTC và +07."""
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

        # Validate dữ liệu đầu vào
        if not _is_valid_username(username):
            return {"success": False, "popup": "popup_09", "title": "Registration Failed",
                    "subtitle": "Username is not valid (3–32 characters: letters/numbers/._-)"}

        if not _is_valid_email(email):
            return {"success": False, "popup": "popup_09", "title": "Registration Failed",
                    "subtitle": "Email is not valid (please go back to the previous step)"}

        ok_pw, msg_pw = _pw_check(password)
        if not ok_pw:
            return {"success": False, "popup": "popup_10", "title": "Registration Failed",
                    "subtitle": msg_pw}

        if password != confirm_password:
            return {"success": False, "popup": "popup_11", "title": "Registration Failed",
                    "subtitle": "Confirm Password must be same as Password"}

        if not otp.isdigit() or not (4 <= len(otp) <= 8):
            return {"success": False, "popup": "popup_10", "title": "Registration Failed",
                    "subtitle": "OTP is not valid (4–8 digits)"}

        # Kiểm tra tồn tại trong DB
        if _email_exists(email):
            return {"success": False, "popup": "popup_09", "title": "Registration Failed",
                    "subtitle": "Email already exists"}
        if _username_exists(username):
            return {"success": False, "popup": "popup_09", "title": "Registration Failed",
                    "subtitle": "Username already exists"}

        # Lấy OTP mới nhất
        row = _fetch_latest_otp_row(email)
        if not row:
            return {"success": False, "popup": "popup_09", "title": "Registration Failed",
                    "subtitle": "No OTP found for this email. Please send OTP again."}

        if int(row.get("used", 0) or 0) == 1:
            return {"success": False, "popup": "popup_09", "title": "Registration Failed",
                    "subtitle": "OTP has been used. Please request a new OTP."}

        # Check hạn OTP
        exp_utc = _safe_expiry_utc(row.get("expires_at"))
        if exp_utc is None:
            return {"success": False, "popup": "popup_09", "title": "Registration Failed",
                    "subtitle": "Cannot determine OTP expiry. Please send OTP again."}

        if _now_utc() > exp_utc:
            return {"success": False, "popup": "popup_09", "title": "Registration Failed",
                    "subtitle": "OTP has expired. Please send OTP again."}

        # Kiểm tra OTP hash
        otp_hash = (row.get("otp_hash") or "").encode()
        try:
            if not otp_hash or not bcrypt.checkpw(otp.encode(), otp_hash):
                return {"success": False, "popup": "popup_09", "title": "Registration Failed",
                        "subtitle": "OTP is not correct"}
        except Exception:
            return {"success": False, "popup": "popup_09", "title": "Registration Failed",
                    "subtitle": "OTP is not correct"}

        # Tạo user + đánh dấu OTP đã dùng
        user_data = _create_user(username, email, password)
        try:
            _mark_otp_used(int(row["id"]))
        except Exception:
            pass

        return {"success": True, "popup": "popup_05", "title": "Registration Successful",
                "subtitle": "Account created! Please complete your profile.", "user_data": user_data}
