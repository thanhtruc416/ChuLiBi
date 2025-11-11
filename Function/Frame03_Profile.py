from datetime import datetime, timezone
from typing import Dict, Any, Optional
import re

from Function.db import get_conn

UTC = timezone.utc

# ====== Validate cơ bản ======
_NAME_RE = re.compile(r"^[^\t\n\r]{1,100}$")  # tránh ký tự control, giới hạn 100
_ROLE_RE = re.compile(r"^[A-Za-z0-9 _\-/]{1,50}$")

def _ok_name(s: str) -> bool:
    return isinstance(s, str) and bool(_NAME_RE.match(s.strip()))

def _ok_role(s: str) -> bool:
    return isinstance(s, str) and bool(_ROLE_RE.match(s.strip()))

def _user_exists(user_id: int) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM user_data WHERE id=%s LIMIT 1", (user_id,))
            return cur.fetchone() is not None

def _read_user(user_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, email, username, full_name, business_name, role, is_active
                FROM user_data
                WHERE id=%s
                """,
                (user_id,)
            )
            row = cur.fetchone()
    if not row:
        return None
    if isinstance(row, dict):
        return row
    # tuple → dict theo thứ tự SELECT
    keys = ["id", "email", "username", "full_name", "business_name", "role", "is_active"]
    return dict(zip(keys, row))

def _update_user_profile(user_id: int, full_name: str, business_name: str, role: str) -> None:
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        with conn.cursor() as cur:
            # nếu bạn có cột updated_at thì set; nếu chưa có, câu lệnh vẫn OK khi bỏ cột này
            try:
                cur.execute(
                    """
                    UPDATE user_data
                    SET full_name=%s, business_name=%s, role=%s, is_active=1, updated_at=%s
                    WHERE id=%s
                    """,
                    (full_name, business_name, role, now_utc, user_id)
                )
            except Exception:
                # fallback nếu bảng chưa có updated_at
                cur.execute(
                    """
                    UPDATE user_data
                    SET full_name=%s, business_name=%s, role=%s, is_active=1
                    WHERE id=%s
                    """,
                    (full_name, business_name, role, user_id)
                )
        conn.commit()

class AuthService:
    @staticmethod
    def update_user_profile(user_id: int, full_name: str, business_name: str, role: str) -> Dict[str, Any]:
        # --- Validate đầu vào ---
        if not isinstance(user_id, int) or user_id <= 0:
            return {"success": False, "message": "User ID không hợp lệ"}

        if not _ok_name(full_name):
            return {"success": False, "message": "Full name không hợp lệ (tối đa 100 ký tự)"}

        if not _ok_name(business_name):
            return {"success": False, "message": "Business name không hợp lệ (tối đa 100 ký tự)"}

        if not _ok_role(role):
            return {"success": False, "message": "Role không hợp lệ (chỉ chữ/số/khoảng trắng/_ - /)"}

        # --- Kiểm tra user tồn tại ---
        if not _user_exists(user_id):
            return {"success": False, "message": "Không tìm thấy người dùng"}

        # --- Cập nhật ---
        try:
            _update_user_profile(user_id, full_name.strip(), business_name.strip(), role.strip())
            # đọc lại để trả về UI
            payload = _read_user(user_id) or {}
            return {"success": True, "message": "Cập nhật hồ sơ thành công", "user_data": payload}
        except Exception as e:
            return {"success": False, "message": f"Lỗi hệ thống: {e}"}
