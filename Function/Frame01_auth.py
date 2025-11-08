
"""
Login bằng username + password.
Return:
  (True, {"id","email","username","full_name","business_name","role"})
  hoặc (False, "Lý do")
"""
import re
import bcrypt
from Function.db import get_conn

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
def _is_valid_username(username: str) -> bool:
    return bool(username) and _USERNAME_RE.match(username)

def _get_user_by_username(username: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, email, username, password_hash,
                    full_name, business_name, role
                FROM user_data
                WHERE username=%s
                """,
                (username,)
            )
            return cur.fetchone()

def login_with_password(username: str, password: str):
    username = (username or "").strip()
    password = (password or "").strip()

    if not _is_valid_username(username):
        return False, "Username không hợp lệ (3–32 ký tự: chữ/số/._-)"
    if not password:
        return False, "Thiếu mật khẩu"

    user = _get_user_by_username(username)
    if not user:
        return False, "Tài khoản không tồn tại"

    # Map field cho dict/tuple drivers
    if isinstance(user, dict):
        pwd_hash = user.get("password_hash") or ""
        payload = {
            "id": user.get("id"),
            "email": user.get("email"),
            "username": user.get("username"),
            "full_name": user.get("full_name"),
            "business_name": user.get("business_name"),
            "role": user.get("role"),
        }
    else:
        # tuple theo SELECT ở trên
        (uid, email, uname, pwd_hash, full_name, business_name, role) = user
        payload = {
            "id": uid,
            "email": email,
            "username": uname,
            "full_name": full_name,
            "business_name": business_name,
            "role": role,
        }

    try:
        if not bcrypt.checkpw(password.encode(), (pwd_hash or "").encode()):
            return False, "Mật khẩu không đúng"
    except Exception:
        return False, "Mật khẩu không đúng"

    return True, payload
