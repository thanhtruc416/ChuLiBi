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
    random.seed(datetime.now().timestamp())  # 👈 thêm dòng này
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
    email = (email or "").strip()

    if not _is_valid_email(email):
        return False, "Email không hợp lệ"

    try:
        # 1️⃣ Lấy thông tin user từ DB
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username FROM user_data WHERE email=%s", (email.lower(),))
                row = cur.fetchone()
                if not row:
                    return False, "Email không tồn tại"

                # Nếu fetchone trả về tuple (ví dụ (3, 'thanhtruc1'))
                # thì id là phần tử [0], username là phần tử [1]
                if isinstance(row, (list, tuple)):
                    user_id = int(row[0])
                    username = row[1]
                elif isinstance(row, dict):
                    user_id = int(row.get("id"))
                    username = row.get("username", "người dùng")
                else:
                    return False, "Lỗi dữ liệu trả về từ DB"

        # 2️⃣ Sinh OTP riêng
        otp = _gen_otp(6)
        otp_hash = bcrypt.hashpw(otp.encode(), bcrypt.gensalt()).decode()
        expires = datetime.now(timezone.utc) + timedelta(minutes=OTP_TTL_MINUTES)

        # 3️⃣ Lưu OTP riêng cho từng email
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO password_resets (user_id, email, otp_hash, expires_at, used, created_at)
                    VALUES (%s, %s, %s, %s, 0, NOW())
                    ON DUPLICATE KEY UPDATE
                        otp_hash = VALUES(otp_hash),
                        expires_at = VALUES(expires_at),
                        used = 0,
                        created_at = NOW()
                    """,
                    (user_id, email.lower(), otp_hash, expires.strftime("%Y-%m-%d %H:%M:%S")),
                )
            conn.commit()

        # 4️⃣ Gửi email OTP riêng
        body = (
            f"Xin chào {username},\n\n"
            f"Mã OTP của bạn là: {otp}\n"
            f"Hiệu lực trong {OTP_TTL_MINUTES} phút.\n\n"
            f"Nếu không phải bạn yêu cầu, hãy bỏ qua email này."
        )
        _send_mail(email, "Mã OTP khôi phục mật khẩu", body)

        return True, f"Đã gửi OTP cho {email}"

    except Exception as e:
        return False, f"Lỗi: {e}"
