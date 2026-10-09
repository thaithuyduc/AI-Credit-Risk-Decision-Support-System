"""
This module contains utility functions for interacting with a PostgreSQL database using SQLAlchemy and pandas. 
It includes functions to map pandas dtypes to PostgreSQL types, sanitize column names, generate CREATE TABLE SQL statements, and retrieve data based on a specific customer ID from multiple related tables.
"""

import pandas as pd
import numpy as np
from sqlalchemy import text, bindparam
from sqlalchemy.orm import Session

def map_dtype_to_postgres(dtype, series: pd.Series) -> str:
    """Map a pandas dtype to a PostgreSQL column type."""

    # Integer types
    if pd.api.types.is_integer_dtype(dtype):
        # Use BIGINT to be safe with large IDs; use INTEGER if you know values are small
        return "INTEGER"

    # Float types
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"

    # Boolean
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"

    # Categorical -> treat as text
    if isinstance(dtype, pd.CategoricalDtype):
        return "TEXT"

    # Object columns: could be strings, mixed types, or numbers-as-strings.
    # Default to TEXT (safest for first import; convert types after loading).
    if pd.api.types.is_object_dtype(dtype):
        return "TEXT"

    # Fallback
    return "TEXT"

def sanitize_column_name(col: str) -> str:
    """Lowercase and replace spaces/special chars so SQL doesn't need quoting."""
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )

def generate_create_table_sql(df: pd.DataFrame, table_name: str) -> str:
    """
    Loop through df columns, map dtypes, and build a CREATE TABLE statement.
    Generate a PostgreSQL CREATE TABLE statement from a pandas DataFrame's dtypes.
    """
    lines = []
    for col in df.columns:
        # col_name = sanitize_column_name(col)
        pg_type = map_dtype_to_postgres(df[col].dtype, df[col])
        lines.append(f"    {col} {pg_type}")

    columns_sql = ",\n".join(lines)
    sql = f"CREATE TABLE {table_name} (\n{columns_sql}\n);"
    return sql

def ensure_not_empty(df: pd.DataFrame, key_col: str = None, dummy_value=None) -> pd.DataFrame:
    """
    Nếu DataFrame rỗng thì tạo 1 dòng dummy có đúng cột.
    """
    if df is None or df.empty:
        # Tạo 1 dòng với NaN cho tất cả cột
        dummy = pd.DataFrame([{col: np.nan for col in df.columns}])
        
        # Nếu có cột key thì gán giá trị dummy
        if key_col and key_col in dummy.columns:
            dummy[key_col] = dummy_value if dummy_value is not None else -999999
        
        return dummy
    return df

def get_data_based_on_id(db: Session, id: int):
    """
    Hàm này nhận vào một mã khách hàng sau đó sẽ lấy ra dữ liệu của khách hàng này
    từ tất cả các bảng có trong Database.
    """

    def to_df(result):
        """Build a DataFrame that keeps column names even when zero rows are returned."""
        columns = list(result.keys())
        rows = result.mappings().all()
        return pd.DataFrame(rows, columns=columns)

    result_application_train = db.execute(
        text("SELECT * FROM application_train WHERE sk_id_curr = :sk_id_curr"),
        {"sk_id_curr": id},
    )
    df_application_train = to_df(result_application_train)

    result_previous_application = db.execute(
        text("SELECT * FROM previous_application WHERE sk_id_curr = :sk_id_curr"),
        {"sk_id_curr": id},
    )
    df_previous_application = to_df(result_previous_application)
    sk_id_prev_previous_application = df_previous_application["sk_id_prev"].tolist()

    result_bureau = db.execute(
        text("SELECT * FROM bureau WHERE sk_id_curr = :sk_id_curr"),
        {"sk_id_curr": id},
    )
    df_bureau = to_df(result_bureau)
    sk_id_bureau_from_bureau = df_bureau["sk_id_bureau"].tolist()

    # --- IN clauses: must use bindparam(..., expanding=True) ---

    bureau_balance_stmt = text(
        "SELECT * FROM bureau_balance WHERE sk_id_bureau IN :sk_id_bureau"
    ).bindparams(bindparam("sk_id_bureau", expanding=True))
    result_bureau_balance = db.execute(
        bureau_balance_stmt,
        {"sk_id_bureau": sk_id_bureau_from_bureau or [None]},  # avoid empty-list error
    )
    df_bureau_balance = to_df(result_bureau_balance)

    credit_card_balance_stmt = text(
        "SELECT * FROM credit_card_balance WHERE sk_id_prev IN :sk_id_prev"
    ).bindparams(bindparam("sk_id_prev", expanding=True))
    result_credit_card_balance = db.execute(
        credit_card_balance_stmt,
        {"sk_id_prev": sk_id_prev_previous_application or [None]},
    )
    df_credit_card_balance = to_df(result_credit_card_balance)

    installments_payments_stmt = text(
        "SELECT * FROM installments_payments WHERE sk_id_prev IN :sk_id_prev"
    ).bindparams(bindparam("sk_id_prev", expanding=True))
    result_installments_payments = db.execute(
        installments_payments_stmt,
        {"sk_id_prev": sk_id_prev_previous_application or [None]},
    )
    df_installments_payments = to_df(result_installments_payments)

    pos_cash_balance_stmt = text(
        "SELECT * FROM pos_cash_balance WHERE sk_id_prev IN :sk_id_prev"
    ).bindparams(bindparam("sk_id_prev", expanding=True))
    result_POS_CASH_balance = db.execute(
        pos_cash_balance_stmt,
        {"sk_id_prev": sk_id_prev_previous_application or [None]},
    )
    df_POS_CASH_balance = to_df(result_POS_CASH_balance)

    # upper case column names for all DataFrames
    df_application_train.columns = [col.upper() for col in df_application_train.columns]
    df_previous_application.columns = [col.upper() for col in df_previous_application.columns]
    df_bureau.columns = [col.upper() for col in df_bureau.columns]
    df_bureau_balance.columns = [col.upper() for col in df_bureau_balance.columns]
    df_credit_card_balance.columns = [col.upper() for col in df_credit_card_balance.columns]
    df_installments_payments.columns = [col.upper() for col in df_installments_payments.columns]
    df_POS_CASH_balance.columns = [col.upper() for col in df_POS_CASH_balance.columns]
    
    # Ensure all DataFrames are not empty
    df_bureau_balance      = ensure_not_empty(df_bureau_balance,      key_col="SK_ID_BUREAU")
    df_POS_CASH_balance    = ensure_not_empty(df_POS_CASH_balance,    key_col="SK_ID_PREV")
    df_application_train   = ensure_not_empty(df_application_train,   key_col="SK_ID_CURR")
    df_credit_card_balance = ensure_not_empty(df_credit_card_balance, key_col="SK_ID_PREV")
    df_previous_application= ensure_not_empty(df_previous_application,key_col="SK_ID_PREV")
    df_bureau              = ensure_not_empty(df_bureau,              key_col="SK_ID_CURR")
    df_installments_payments = ensure_not_empty(df_installments_payments, key_col="SK_ID_PREV")

    dict_result = {
        "application_train": df_application_train,
        "previous_application": df_previous_application,
        "bureau": df_bureau,
        "bureau_balance": df_bureau_balance,
        "credit_card_balance": df_credit_card_balance,
        "installments_payments": df_installments_payments,
        "POS_CASH_balance": df_POS_CASH_balance,
    }
    
    return dict_result