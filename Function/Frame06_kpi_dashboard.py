"""
KPI Functions for Retail Dashboard
This module contains 8 functions to retrieve key performance indicators from CSV data.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd

# Path to CSV file
ROOT = Path(__file__).resolve().parents[1]  # Go up to ChuLiBi directory
CSV_PATH = ROOT / "Dataset"/"Output" / "df_raw_dashboard.csv"

# Global variable to cache the dataframe
_df_cache = None


def load_data() -> Optional[pd.DataFrame]:
    """
    Load data from CSV file and cache it.

    Returns:
        DataFrame or None if error occurs
    """
    global _df_cache

    if _df_cache is not None:
        return _df_cache

    try:
        _df_cache = pd.read_csv(CSV_PATH)
        print(f"Data loaded successfully from {CSV_PATH}")
        return _df_cache
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def get_total_customers() -> Optional[int]:
    """
    KPI 1: Get the total number of unique customers.

    Returns:
        Total number of customers or None if error occurs
    """
    df = load_data()
    if df is None:
        return None

    try:
        return int(df['CustomerID'].nunique())
    except Exception as e:
        print(f"Error calculating total customers: {e}")
        return None


def get_avg_age() -> Optional[float]:
    """
    KPI 2: Get the average age of customers.

    Returns:
        Average age or None if error occurs
    """
    df = load_data()
    if df is None:
        return None

    try:
        avg_age = df['Age'].mean()
        return round(avg_age, 2) if pd.notna(avg_age) else None
    except Exception as e:
        print(f"Error calculating average age: {e}")
        return None


def get_total_orders() -> Optional[int]:
    """
    KPI 3: Get the total number of orders placed.

    Returns:
        Total number of orders or None if error occurs
    """
    df = load_data()
    if df is None:
        return None

    try:
        total_orders = df['No. of orders placed'].sum()
        return int(total_orders) if pd.notna(total_orders) else None
    except Exception as e:
        print(f"Error calculating total orders: {e}")
        return None


def get_high_frequency_customer_rate() -> Optional[float]:
    """
    KPI 4: Get the percentage of high-frequency customers.
    High-frequency customers are defined as those with more than average orders.

    Returns:
        Percentage of high-frequency customers or None if error occurs
    """
    df = load_data()
    if df is None:
        return None

    try:
        # Filter out null values
        df_filtered = df[df['No. of orders placed'].notna()]

        # Calculate average orders
        avg_orders = df_filtered['No. of orders placed'].mean()

        # Count high frequency customers (more than average)
        high_freq_count = (df_filtered['No. of orders placed'] > avg_orders).sum()
        total_count = len(df_filtered)

        # Calculate percentage
        high_freq_rate = (high_freq_count * 100.0 / total_count) if total_count > 0 else 0
        return round(high_freq_rate, 2)
    except Exception as e:
        print(f"Error calculating high frequency customer rate: {e}")
        return None


def get_avg_order_value() -> Optional[float]:
    """
    KPI 5: Get the average order value.

    Returns:
        Average order value or None if error occurs
    """
    df = load_data()
    if df is None:
        return None

    try:
        avg_value = df['Order Value'].mean()
        return round(avg_value, 2) if pd.notna(avg_value) else None
    except Exception as e:
        print(f"Error calculating average order value: {e}")
        return None


def get_avg_delivery_time() -> Optional[float]:
    """
    KPI 6: Get the average delivery time.

    Returns:
        Average delivery time or None if error occurs
    """
    df = load_data()
    if df is None:
        return None

    try:
        avg_time = df['Delivery Time'].mean()
        return round(avg_time, 2) if pd.notna(avg_time) else None
    except Exception as e:
        print(f"Error calculating average delivery time: {e}")
        return None


def get_avg_restaurant_rating() -> Optional[float]:
    """
    KPI 7: Get the average restaurant rating.

    Returns:
        Average restaurant rating or None if error occurs
    """
    df = load_data()
    if df is None:
        return None

    try:
        avg_rating = df['Restaurant Rating'].mean()
        return round(avg_rating, 2) if pd.notna(avg_rating) else None
    except Exception as e:
        print(f"Error calculating average restaurant rating: {e}")
        return None


def get_avg_delivery_rating() -> Optional[float]:
    """
    KPI 8: Get the average delivery rating.

    Returns:
        Average delivery rating or None if error occurs
    """
    df = load_data()
    if df is None:
        return None

    try:
        avg_rating = df['Delivery Rating'].mean()
        return round(avg_rating, 2) if pd.notna(avg_rating) else None
    except Exception as e:
        print(f"Error calculating average delivery rating: {e}")
        return None


def get_all_kpis() -> Dict[str, Any]:
    """
    Get all KPIs at once and return as a dictionary.

    Returns:
        Dictionary containing all KPI values
    """
    # Try to load data from CSV
    df = load_data()

    # If no data, return placeholder data
    if df is None:
        print("Warning: CSV data not loaded. Using placeholder data.")
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

    # Get real data from CSV
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
