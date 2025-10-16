# Function/Frame01_auth.py
"""
Authentication module for handling user login, password verification
"""
import bcrypt
from Function.db import get_conn

class AuthService:
    """Handle user authentication operations"""
    
    @staticmethod
    def verify_login(username: str, password: str) -> dict:
        """
        Verify user credentials and return user data if valid.
        
        Args:
            username: User's username
            password: Plain text password
            
        Returns:
            dict with keys:
                - success: bool
                - message: str
                - user_data: dict (if success=True) with id, username, email
        """
        if not username or not password:
            return {
                "success": False,
                "message": "Username and password are required",
                "user_data": None
            }
        
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    # Fetch user by username
                    sql = """
                        SELECT id, username, password_hash, email, is_active, 
                               full_name, business_name, role
                        FROM user_data
                        WHERE username = %s
                    """
                    cursor.execute(sql, (username,))
                    user = cursor.fetchone()
                    
                    if not user:
                        return {
                            "success": False,
                            "message": "Invalid username or password",
                            "user_data": None
                        }
                    
                    # Check if account is active
                    if not user['is_active']:
                        return {
                            "success": False,
                            "message": "Account is disabled. Please contact support.",
                            "user_data": None
                        }
                    
                    # Verify password
                    password_hash = user['password_hash']
                    
                    # Check if password matches
                    if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                        return {
                            "success": True,
                            "message": "Login successful",
                            "user_data": {
                                "id": user['id'],
                                "username": user['username'],
                                "email": user['email'],
                                "full_name": user.get('full_name'),
                                "business_name": user.get('business_name'),
                                "role": user.get('role')
                            }
                        }
                    else:
                        return {
                            "success": False,
                            "message": "Invalid username or password",
                            "user_data": None
                        }
                        
        except Exception as e:
            print(f"Login error: {e}")
            return {
                "success": False,
                "message": f"Database error: {str(e)}",
                "user_data": None
            }
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password for storing in database.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def get_user_by_email(email: str) -> dict:
        """
        Get user data by email.
        
        Args:
            email: User's email address
            
        Returns:
            User dict or None if not found
        """
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    sql = """
                        SELECT id, username, email, is_active
                        FROM user_data
                        WHERE email = %s
                    """
                    cursor.execute(sql, (email,))
                    return cursor.fetchone()
        except Exception as e:
            print(f"Error fetching user by email: {e}")
            return None
    
    @staticmethod
    def get_user_by_username(username: str) -> dict:
        """
        Get user data by username.
        
        Args:
            username: User's username
            
        Returns:
            User dict or None if not found
        """
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    sql = """
                        SELECT id, username, email, is_active
                        FROM user_data
                        WHERE username = %s
                    """
                    cursor.execute(sql, (username,))
                    return cursor.fetchone()
        except Exception as e:
            print(f"Error fetching user by username: {e}")
            return None
    
    @staticmethod
    def register_user(username: str, email: str, password: str, confirm_password: str) -> dict:
        """
        Register a new user.
        
        Args:
            username: Desired username
            email: User's email address
            password: Plain text password
            confirm_password: Password confirmation
            
        Returns:
            dict with keys:
                - success: bool
                - message: str
                - user_data: dict (if success=True) with id, username, email
        """
        # Validation
        if not username or not email or not password or not confirm_password:
            return {
                "success": False,
                "message": "All fields are required",
                "user_data": None
            }
        
        # Check username length
        if len(username) < 3:
            return {
                "success": False,
                "message": "Username must be at least 3 characters long",
                "user_data": None
            }
        
        if len(username) > 50:
            return {
                "success": False,
                "message": "Username must be less than 50 characters",
                "user_data": None
            }
        
        # Check password length
        if len(password) < 6:
            return {
                "success": False,
                "message": "Password must be at least 6 characters long",
                "user_data": None
            }
        
        # Check password confirmation
        if password != confirm_password:
            return {
                "success": False,
                "message": "Passwords do not match",
                "user_data": None
            }
        
        # Basic email validation
        if "@" not in email or "." not in email:
            return {
                "success": False,
                "message": "Please enter a valid email address",
                "user_data": None
            }
        
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    # Check if username already exists
                    cursor.execute("SELECT id FROM user_data WHERE username = %s", (username,))
                    if cursor.fetchone():
                        return {
                            "success": False,
                            "message": "Username already exists. Please choose another one.",
                            "user_data": None
                        }
                    
                    # Check if email already exists
                    cursor.execute("SELECT id FROM user_data WHERE email = %s", (email,))
                    if cursor.fetchone():
                        return {
                            "success": False,
                            "message": "Email already registered. Please use another email or login.",
                            "user_data": None
                        }
                    
                    # Hash password
                    password_hash = AuthService.hash_password(password)
                    
                    # Insert new user
                    sql = """
                        INSERT INTO user_data (username, password_hash, email, is_active)
                        VALUES (%s, %s, %s, 1)
                    """
                    cursor.execute(sql, (username, password_hash, email))
                    conn.commit()
                    
                    # Get the newly created user ID
                    user_id = cursor.lastrowid
                    
                    return {
                        "success": True,
                        "message": "Account created successfully! Please login.",
                        "user_data": {
                            "id": user_id,
                            "username": username,
                            "email": email
                        }
                    }
                    
        except Exception as e:
            print(f"Registration error: {e}")
            return {
                "success": False,
                "message": f"Registration failed: {str(e)}",
                "user_data": None
            }
    
    @staticmethod
    def update_user_profile(user_id: int, full_name: str, business_name: str, user_role: str) -> dict:
        """
        Update user profile information.
        
        Args:
            user_id: User's ID
            full_name: User's full name
            business_name: User's business name
            user_role: User's role/position
            
        Returns:
            dict with keys:
                - success: bool
                - message: str
                - user_data: dict (if success=True)
        """
        # Validation
        if not user_id:
            return {
                "success": False,
                "message": "User ID is required",
                "user_data": None
            }
        
        if not full_name or not business_name or not user_role:
            return {
                "success": False,
                "message": "All profile fields are required",
                "user_data": None
            }
        
        # Trim whitespace
        full_name = full_name.strip()
        business_name = business_name.strip()
        user_role = user_role.strip()
        
        # Validate lengths
        if len(full_name) < 2:
            return {
                "success": False,
                "message": "Full name must be at least 2 characters",
                "user_data": None
            }
        
        if len(business_name) < 2:
            return {
                "success": False,
                "message": "Business name must be at least 2 characters",
                "user_data": None
            }
        
        if len(user_role) < 2:
            return {
                "success": False,
                "message": "Role must be at least 2 characters",
                "user_data": None
            }
        
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    # Update user profile
                    sql = """
                        UPDATE user_data
                        SET full_name = %s,
                            business_name = %s,
                            user_role = %s,
                            profile_completed = 1
                        WHERE id = %s
                    """
                    cursor.execute(sql, (full_name, business_name, user_role, user_id))
                    conn.commit()
                    
                    if cursor.rowcount == 0:
                        return {
                            "success": False,
                            "message": "User not found",
                            "user_data": None
                        }
                    
                    # Fetch updated user data
                    cursor.execute("""
                        SELECT id, username, email, full_name, business_name, 
                               user_role, profile_completed
                        FROM user_data
                        WHERE id = %s
                    """, (user_id,))
                    user = cursor.fetchone()
                    
                    return {
                        "success": True,
                        "message": "Profile updated successfully",
                        "user_data": {
                            "id": user['id'],
                            "username": user['username'],
                            "email": user['email'],
                            "full_name": user['full_name'],
                            "business_name": user['business_name'],
                            "user_role": user['user_role'],
                            "profile_completed": user['profile_completed']
                        }
                    }
                    
        except Exception as e:
            print(f"Profile update error: {e}")
            return {
                "success": False,
                "message": f"Failed to update profile: {str(e)}",
                "user_data": None
            }
    
    @staticmethod
    def get_user_profile(user_id: int) -> dict:
        """
        Get user profile by ID.
        
        Args:
            user_id: User's ID
            
        Returns:
            User profile dict or None
        """
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    sql = """
                        SELECT id, username, email, full_name, business_name,
                               role, is_active
                        FROM user_data
                        WHERE id = %s
                    """
                    cursor.execute(sql, (user_id,))
                    return cursor.fetchone()
        except Exception as e:
            print(f"Error fetching user profile: {e}")
            return None
    
    @staticmethod
    def update_user_profile(user_id: int, full_name: str, business_name: str, role: str) -> dict:
        """
        Update user profile information.
        
        Args:
            user_id: User's ID
            full_name: User's full name
            business_name: Business name
            role: User's role
            
        Returns:
            dict with keys:
                - success: bool
                - message: str
        """
        if not user_id:
            return {
                "success": False,
                "message": "User ID is required"
            }
        
        # Validate required fields
        if not full_name or not business_name or not role:
            return {
                "success": False,
                "message": "All fields (Full Name, Business Name, and Role) are required"
            }
        
        # Validate field lengths
        if len(full_name) > 255:
            return {
                "success": False,
                "message": "Full name must be less than 255 characters"
            }
        
        if len(business_name) > 255:
            return {
                "success": False,
                "message": "Business name must be less than 255 characters"
            }
        
        if len(role) > 100:
            return {
                "success": False,
                "message": "Role must be less than 100 characters"
            }
        
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    # Update user profile
                    sql = """
                        UPDATE user_data
                        SET full_name = %s,
                            business_name = %s,
                            role = %s
                        WHERE id = %s
                    """
                    cursor.execute(sql, (full_name, business_name, role, user_id))
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        return {
                            "success": True,
                            "message": "Profile updated successfully"
                        }
                    else:
                        return {
                            "success": False,
                            "message": "User not found"
                        }
                        
        except Exception as e:
            print(f"Profile update error: {e}")
            return {
                "success": False,
                "message": f"Failed to update profile: {str(e)}"
            }


if __name__ == "__main__":
    # Test authentication
    print("Testing authentication...")
    
    # Test with sample credentials (username: thanhtruc, any password for testing)
    result = AuthService.verify_login("thanhtruc", "test123")
    print(f"Login test result: {result}")
    
    # Generate a proper hash for testing
    test_password = "test123"
    hashed = AuthService.hash_password(test_password)
    print(f"\nHashed password for 'test123': {hashed}")
    print("\nUpdate your database with:")
    print(f"UPDATE user_data SET password_hash='{hashed}' WHERE username='thanhtruc';")
