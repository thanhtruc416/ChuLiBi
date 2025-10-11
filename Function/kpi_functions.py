"""
KPI Functions for Retail Dashboard
This module contains 8 functions to retrieve key performance indicators from the database.
"""

import mysql.connector
from mysql.connector import Error
from typing import Optional, Dict, Any


def get_db_connection():
    """
    Establish a connection to the MySQL database.
    
    Returns:
        connection: MySQL database connection object
    """
    DB_CONFIG = {
        'host': 'localhost',
        'port': 3306,
        'user': '', # Update with your MySQL username
        'password': '', # Update with your MySQL password
        'database': '' # Update with your database name
    }

    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None


def execute_query(query: str) -> Optional[Any]:
    """
    Execute a SQL query and return the result.
    
    Args:
        query: SQL query string
        
    Returns:
        Query result or None if error occurs
    """
    connection = get_db_connection()
    if not connection:
        return None
    
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result else None
    except Error as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def get_total_customers() -> Optional[int]:
    """
    KPI 1: Get the total number of unique customers.
    
    Returns:
        Total number of customers or None if error occurs
    """
    query = """
        SELECT COUNT(DISTINCT CustomerID) 
        FROM df_raw_dashboard
    """
    result = execute_query(query)
    return result


def get_avg_age() -> Optional[float]:
    """
    KPI 2: Get the average age of customers.
    
    Returns:
        Average age or None if error occurs
    """
    query = """
        SELECT AVG(Age) 
        FROM df_raw_dashboard
        WHERE Age IS NOT NULL
    """
    result = execute_query(query)
    return round(result, 2) if result else None


def get_total_orders() -> Optional[int]:
    """
    KPI 3: Get the total number of orders placed.
    
    Returns:
        Total number of orders or None if error occurs
    """
    query = """
        SELECT SUM(`No. of orders placed`) 
        FROM df_raw_dashboard
        WHERE `No. of orders placed` IS NOT NULL
    """
    result = execute_query(query)
    return result


def get_high_frequency_customer_rate() -> Optional[float]:
    """
    KPI 4: Get the percentage of high-frequency customers.
    High-frequency customers are defined as those with more than average orders.
    
    Returns:
        Percentage of high-frequency customers or None if error occurs
    """
    query = """
        SELECT 
            (COUNT(CASE WHEN `No. of orders placed` > avg_orders THEN 1 END) * 100.0 / COUNT(*)) AS high_frequency_rate
        FROM df_raw_dashboard
        CROSS JOIN (
            SELECT AVG(`No. of orders placed`) AS avg_orders 
            FROM df_raw_dashboard 
            WHERE `No. of orders placed` IS NOT NULL
        ) AS avg_table
        WHERE `No. of orders placed` IS NOT NULL
    """
    result = execute_query(query)
    return round(result, 2) if result else None


def get_avg_order_value() -> Optional[float]:
    """
    KPI 5: Get the average order value.
    
    Returns:
        Average order value or None if error occurs
    """
    query = """
        SELECT AVG(`Order Value`) 
        FROM df_raw_dashboard
        WHERE `Order Value` IS NOT NULL
    """
    result = execute_query(query)
    return round(result, 2) if result else None


def get_avg_delivery_time() -> Optional[float]:
    """
    KPI 6: Get the average delivery time.
    
    Returns:
        Average delivery time or None if error occurs
    """
    query = """
        SELECT AVG(`Delivery Time`) 
        FROM df_raw_dashboard
        WHERE `Delivery Time` IS NOT NULL
    """
    result = execute_query(query)
    return round(result, 2) if result else None


def get_avg_restaurant_rating() -> Optional[float]:
    """
    KPI 7: Get the average restaurant rating.
    
    Returns:
        Average restaurant rating or None if error occurs
    """
    query = """
        SELECT AVG(`Restaurant Rating`) 
        FROM df_raw_dashboard
        WHERE `Restaurant Rating` IS NOT NULL
    """
    result = execute_query(query)
    return round(result, 2) if result else None


def get_avg_delivery_rating() -> Optional[float]:
    """
    KPI 8: Get the average delivery rating.
    
    Returns:
        Average delivery rating or None if error occurs
    """
    query = """
        SELECT AVG(`Delivery Rating`) 
        FROM df_raw_dashboard
        WHERE `Delivery Rating` IS NOT NULL
    """
    result = execute_query(query)
    return round(result, 2) if result else None


def get_all_kpis() -> Dict[str, Any]:
    """
    Get all KPIs at once and return as a dictionary.
    
    Returns:
        Dictionary containing all KPI values
    """
    # Try to get real data from database
    connection = get_db_connection()
    
    # If no database connection, return placeholder data
    if not connection:
        print("Warning: Database not connected. Using placeholder data.")
        return {
            'total_customers': 1250,
            'avg_age': 32.5,
            'total_orders': 8450,
            'high_frequency_customer_rate': 35.2,
            'avg_order_value': 245.80,
            'avg_delivery_time': 28.5,
            'avg_restaurant_rating': 4.2,
            'avg_delivery_rating': 4.5
        }
    
    # Get real data
    kpis = {
        'total_customers': get_total_customers(),
        'avg_age': get_avg_age(),
        'total_orders': get_total_orders(),
        'high_frequency_customer_rate': get_high_frequency_customer_rate(),
        'avg_order_value': get_avg_order_value(),
        'avg_delivery_time': get_avg_delivery_time(),
        'avg_restaurant_rating': get_avg_restaurant_rating(),
        'avg_delivery_rating': get_avg_delivery_rating()
    }
    return kpis


if __name__ == "__main__":
    # Test all KPI functions
    print("=" * 60)
    print("RETAIL DASHBOARD KPIs")
    print("=" * 60)
    
    print(f"\n1. Total Customers: {get_total_customers()}")
    print(f"2. Average Age: {get_avg_age()}")
    print(f"3. Total Orders: {get_total_orders()}")
    print(f"4. High Frequency Customer Rate: {get_high_frequency_customer_rate()}%")
    print(f"5. Average Order Value: ${get_avg_order_value()}")
    print(f"6. Average Delivery Time: {get_avg_delivery_time()} mins")
    print(f"7. Average Restaurant Rating: {get_avg_restaurant_rating()}/5")
    print(f"8. Average Delivery Rating: {get_avg_delivery_rating()}/5")
    
    print("\n" + "=" * 60)
    print("All KPIs as Dictionary:")
    print("=" * 60)
    print(get_all_kpis())
