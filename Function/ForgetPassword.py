import os
import re
import smtplib
import random
import string
import bcrypt
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone

from Function.db import get_conn

OTP_TTL_MINUTES = 10  # OTP hết hạn sau 10 phút

# ====== VALIDATION: chỉ kiểm tra format email (không ép domain) ======
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _is_valid_email(email: str) -> bool:
    return bool(email) and len(email) <= 254 and _EMAIL_RE.match(email)

# ====== OTP & MAIL ======
def _gen_otp(n=6) -> str:
    return "".join(random.choices(string.digits, k=n))

def _send_mail(to_email: str, subject: str, body: str):
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    app  = os.getenv("APP_NAME", "Your App")

    if not (user and pwd):
        raise RuntimeError("Chưa cấu hình SMTP_USER/SMTP_PASS (App Password)")

    msg = EmailMessage()
    msg["From"] = f"{app} <{user}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)

# ====== PUBLIC API ======
def send_otp_if_email_exists(email: str):
    """
    Input: email người dùng nhập.
    - Validate format email.
    - Nếu tồn tại trong user_data: sinh OTP, lưu (hash, hạn 10p), gửi mail.
    - Nếu không: trả False cho UI hiển thị warning.

    Return:
      (True, "Đã gửi OTP")  hoặc  (False, "Lý do")
    """
    email = (email or "").strip()

    if not _is_valid_email(email):
        return False, "Email không hợp lệ"

    try:
        # Kiểm tra email
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username FROM user_data WHERE email=%s", (email.lower(),))
                user = cur.fetchone()

        if not user:
            return False, "Email không tồn tại"

        # Tạo & lưu OTP
        otp = _gen_otp(6)
        otp_hash = bcrypt.hashpw(otp.encode(), bcrypt.gensalt()).decode()
        expires = datetime.now(timezone.utc) + timedelta(minutes=OTP_TTL_MINUTES)

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO password_resets (user_id, otp_hash, expires_at) VALUES (%s, %s, %s)",
                    (user["id"], otp_hash, expires.strftime("%Y-%m-%d %H:%M:%S")),
                )

        # Gửi email
        body = (
            f"Xin chào {user['username']},\n\n"
            f"Mã OTP của bạn là: {otp}\n"
            f"Hiệu lực trong {OTP_TTL_MINUTES} phút.\n\n"
            f"Nếu không phải bạn yêu cầu, hãy bỏ qua email này."
        )
        _send_mail(email, "Mã OTP khôi phục mật khẩu", body)

        return True, "Đã gửi OTP"
    except Exception as e:
        return False, f"Lỗi: {e}"
