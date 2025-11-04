# -*- coding: utf-8 -*-
# File: Function/signup_ex_fn.py
#
# Dùng cho màn ui_Frame02_ex (đăng ký, bước nhập email -> gửi OTP)
# - Validate email
# - Nếu email CHƯA tồn tại trong user_data:
#     + Kiểm tra rate-limit (mỗi 120s)
#     + Sinh OTP, lưu hash + hạn 10 phút vào password_resets
#     + Gửi email OTP
# - Trả (True, "msg") hoặc (False, "lý do")

import os
import re
import smtplib
import random
import string
import bcrypt
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone

from Function.db import get_conn

# ===================== CẤU HÌNH =====================
OTP_TTL_MINUTES = 10
RESEND_COOLDOWN_SECONDS = 120  # 2 phút

# Nếu DB đang lưu datetime "naive" theo giờ VN (+07)
LOCAL_TZ = timezone(timedelta(hours=7))   # Asia/Ho_Chi_Minh
UTC = timezone.utc

def _now_utc():
    return datetime.now(UTC)

# ===================== VALIDATION =====================
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _is_valid_email(email: str) -> bool:
    return bool(email) and len(email) <= 254 and _EMAIL_RE.match(email)

# ===================== OTP & MAIL =====================
def _gen_otp(n=6) -> str:
    return "".join(random.choices(string.digits, k=n))

def _send_mail(to_email: str, subject: str, body: str):
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    app  = os.getenv("APP_NAME", "ChuLiBi")

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

# ===================== TIME HELPERS =====================
def _parse_dt_maybe_naive(dt_val):
    """
    Chuẩn hoá giá trị thời gian từ DB về datetime aware (UTC).
    - Nếu tz-naive: coi như giờ VN (+07) rồi convert sang UTC.
    """
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        d = dt_val
    else:
        # cố gắng parse ISO, fallback "YYYY-mm-dd HH:MM:SS"
        try:
            d = datetime.fromisoformat(str(dt_val))
        except Exception:
            d = datetime.strptime(str(dt_val), "%Y-%m-%d %H:%M:%S")

    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL_TZ)
    return d.astimezone(UTC)

def _get_last_sent_time_utc(email: str):
    """
    Lấy thời điểm gửi OTP gần nhất (UTC). Ưu tiên created_at; nếu không có,
    suy từ expires_at - OTP_TTL_MINUTES.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Lấy bản ghi mới nhất cho email
            cur.execute(
                """
                SELECT created_at, expires_at
                FROM password_resets
                WHERE email=%s
                ORDER BY COALESCE(created_at, expires_at) DESC
                LIMIT 1
                """,
                (email.lower(),)
            )
            row = cur.fetchone()

    if not row:
        return None

    # map dict/tuple
    if isinstance(row, dict):
        created_at = row.get("created_at")
        expires_at = row.get("expires_at")
    else:
        created_at, expires_at = row[0], row[1]

    created_at = _parse_dt_maybe_naive(created_at) if created_at else None
    expires_at = _parse_dt_maybe_naive(expires_at) if expires_at else None

    if created_at:
        return created_at
    if expires_at:
        try:
            return expires_at - timedelta(minutes=OTP_TTL_MINUTES)
        except Exception:
            return None
    return None

def _cooldown_seconds_remaining(email: str) -> int:
    last = _get_last_sent_time_utc(email)
    if not last:
        return 0
    elapsed = (_now_utc() - last).total_seconds()
    remain = int(RESEND_COOLDOWN_SECONDS - elapsed)
    return remain if remain > 0 else 0

# ===================== PUBLIC API =====================
def send_otp_if_email_not_exists(email: str):
    """
    Input: email ở flow ĐĂNG KÝ.
    Return: (True, "Đã gửi OTP") hoặc (False, "Lý do")
    """
    email = (email or "").strip()

    if not _is_valid_email(email):
        return False, "Email không hợp lệ"

    try:
        # 0) Email chưa tồn tại trong user_data?
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM user_data WHERE email=%s LIMIT 1", (email.lower(),))
                exists = cur.fetchone() is not None
        if exists:
            return False, "Email đã tồn tại trong hệ thống"

        # 1) Rate-limit 120s
        remain = _cooldown_seconds_remaining(email)
        if remain > 0:
            if remain >= 60:
                mins = (remain + 59) // 60
                return False, f"Bạn vừa yêu cầu OTP. Vui lòng thử lại sau {mins} phút."
            else:
                return False, f"Bạn vừa yêu cầu OTP. Vui lòng thử lại sau {remain} giây."
        # 1.5) Kiểm tra nếu OTP cũ vẫn còn hiệu lực (chưa hết hạn)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT expires_at
                    FROM password_resets
                    WHERE email = %s AND used = 0
                    ORDER BY expires_at DESC
                    LIMIT 1
                """, (email.lower(),))
                row = cur.fetchone()

        if row:
            exp = _parse_dt_maybe_naive(row[0])
            if exp and exp > _now_utc():
                remain_min = int((exp - _now_utc()).total_seconds() // 60)
                return False, f"OTP đã được gửi. Vui lòng kiểm tra email (còn hiệu lực {remain_min} phút)."

        # 2) Tạo & lưu OTP (hash + expiry)
        otp = _gen_otp(6)
        otp_hash = bcrypt.hashpw(otp.encode(), bcrypt.gensalt()).decode()
        expires_utc = _now_utc() + timedelta(minutes=OTP_TTL_MINUTES)
        created_utc = _now_utc()

        with get_conn() as conn:
            with conn.cursor() as cur:
                # Cố gắng chèn đủ cột; nếu DB bạn chưa có created_at/used, fallback bản rút gọn
                try:
                    cur.execute(
                        """
                        INSERT INTO password_resets (email, otp_hash, expires_at, used, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            email.lower(),
                            otp_hash,
                            expires_utc.strftime("%Y-%m-%d %H:%M:%S"),
                            0,
                            created_utc.strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )
                except Exception:
                    # bảng thiếu 1 hoặc 2 cột -> chèn tối thiểu
                    cur.execute(
                        """
                        INSERT INTO password_resets (email, otp_hash, expires_at)
                        VALUES (%s, %s, %s)
                        """,
                        (
                            email.lower(),
                            otp_hash,
                            expires_utc.strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )
            conn.commit()

        # 3) Gửi email
        body = (
            f"Xin chào,\n\n"
            f"Mã OTP đăng ký của bạn là: {otp}\n"
            f"Hiệu lực trong {OTP_TTL_MINUTES} phút.\n\n"
            f"Nếu không phải bạn thực hiện, hãy bỏ qua email này."
        )
        _send_mail(email, "Mã OTP đăng ký tài khoản", body)

        return True, "Đã gửi OTP"

    except Exception as e:
        # trả lỗi gọn cho UI
        return False, f"Lỗi: {e}"
