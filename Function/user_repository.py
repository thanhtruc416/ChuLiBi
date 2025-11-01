# Function/user_repository.py
from Function.db import get_conn
import bcrypt


def update_user_info(username, full_name, business_name, role, email, password=None):
    """
    Cập nhật thông tin người dùng trong bảng user_data.
    Nếu password=None thì không cập nhật mật khẩu.
    """

    # Câu SQL cơ bản
    sql = """
        UPDATE user_data
        SET full_name = %s,
            business_name = %s,
            role = %s,
            email = %s
            {extra}
        WHERE username = %s
    """

    params = [full_name, business_name, role, email]
    extra = ""

    # Nếu có mật khẩu mới, hash lại và thêm vào SQL
    if password:
        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        extra = ", password_hash = %s"
        params.append(hashed_pw)

    params.append(username)
    sql = sql.format(extra=extra)

    # Thực thi truy vấn
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
        conn.commit()
