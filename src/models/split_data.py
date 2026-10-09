
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# Đọc file config
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

RAW_DIR = PROJECT_ROOT / config['paths']['raw_data']
PROCESSED_DIR = PROJECT_ROOT / config['paths']['processed_data']

# Tạo thư mục processed nếu chưa có
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_data():
    application_train = pd.read_csv(RAW_DIR / "application_train_sample.csv")
    bureau = pd.read_csv(RAW_DIR / "bureau_sample.csv")
    bureau_balance = pd.read_csv(RAW_DIR / "bureau_balance_sample.csv")
    previous_application = pd.read_csv(RAW_DIR / "previous_application_sample.csv")
    POS_CASH_balance = pd.read_csv(RAW_DIR / "POS_CASH_balance_sample.csv")
    installments_payments = pd.read_csv(RAW_DIR / "instalments_payments_sample.csv")
    credit_card_balance = pd.read_csv(RAW_DIR / "credit_card_balance_sample.csv")

    return application_train, bureau, bureau_balance, previous_application, POS_CASH_balance, installments_payments, credit_card_balance

def split_data():
    application_train, bureau, bureau_balance, previous_application, POS_CASH_balance, installments_payments, credit_card_balance = load_raw_data()

    X = application_train.drop(columns=['TARGET'])
    y = application_train['TARGET']

    # chia tập dữ liệu cho application_train
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)

    application_train_train = pd.concat([X_train, y_train], axis=1)
    application_train_val = pd.concat([X_val, y_val], axis=1)
    application_train_test = pd.concat([X_test, y_test], axis=1)

    SK_ID_CURR_train = X_train['SK_ID_CURR'].values
    SK_ID_CURR_val = X_val['SK_ID_CURR'].values
    SK_ID_CURR_test = X_test['SK_ID_CURR'].values

    # chia tập dữ liệu cho bureau
    bureau_train = bureau[bureau['SK_ID_CURR'].isin(SK_ID_CURR_train)]
    bureau_val = bureau[bureau['SK_ID_CURR'].isin(SK_ID_CURR_val)]
    bureau_test = bureau[bureau['SK_ID_CURR'].isin(SK_ID_CURR_test)]

    SK_ID_BUREAU_train = bureau_train['SK_ID_BUREAU'].values
    SK_ID_BUREAU_val = bureau_val['SK_ID_BUREAU'].values
    SK_ID_BUREAU_test = bureau_test['SK_ID_BUREAU'].values

    # chia tập dữ liệu cho previous_application
    previous_application_train = previous_application[previous_application['SK_ID_CURR'].isin(SK_ID_CURR_train)]
    previous_application_val = previous_application[previous_application['SK_ID_CURR'].isin(SK_ID_CURR_val)]
    previous_application_test = previous_application[previous_application['SK_ID_CURR'].isin(SK_ID_CURR_test)]

    SK_ID_PREV_train = previous_application_train['SK_ID_PREV'].values
    SK_ID_PREV_val = previous_application_val['SK_ID_PREV'].values
    SK_ID_PREV_test = previous_application_test['SK_ID_PREV'].values

    # chia tập dữ liệu cho bureau_balance
    bureau_balance_train = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(SK_ID_BUREAU_train)]
    bureau_balance_val = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(SK_ID_BUREAU_val)]
    bureau_balance_test = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(SK_ID_BUREAU_test)]

    # chia tập dữ liệu cho POS_CASH_balance
    filter_pos_cash_train = (POS_CASH_balance['SK_ID_CURR'].isin(SK_ID_CURR_train)) & (POS_CASH_balance['SK_ID_PREV'].isin(SK_ID_PREV_train))
    filter_pos_cash_val = (POS_CASH_balance['SK_ID_CURR'].isin(SK_ID_CURR_val)) & (POS_CASH_balance['SK_ID_PREV'].isin(SK_ID_PREV_val))
    filter_pos_cash_test = (POS_CASH_balance['SK_ID_CURR'].isin(SK_ID_CURR_test)) & (POS_CASH_balance['SK_ID_PREV'].isin(SK_ID_PREV_test))

    POS_CASH_balance_train = POS_CASH_balance[filter_pos_cash_train]
    POS_CASH_balance_val = POS_CASH_balance[filter_pos_cash_val]
    POS_CASH_balance_test = POS_CASH_balance[filter_pos_cash_test]

    # chia tập dữ liệu cho instalments_payments
    filter_instalments_train = (installments_payments['SK_ID_CURR'].isin(SK_ID_CURR_train)) & (installments_payments['SK_ID_PREV'].isin(SK_ID_PREV_train))
    filter_instalments_val = (installments_payments['SK_ID_CURR'].isin(SK_ID_CURR_val)) & (installments_payments['SK_ID_PREV'].isin(SK_ID_PREV_val))
    filter_instalments_test = (installments_payments['SK_ID_CURR'].isin(SK_ID_CURR_test)) & (installments_payments['SK_ID_PREV'].isin(SK_ID_PREV_test))

    installments_payments_train = installments_payments[filter_instalments_train]
    installments_payments_val = installments_payments[filter_instalments_val]
    installments_payments_test = installments_payments[filter_instalments_test]

    # chia tập dữ liệu cho credit_card_balance
    filter_credit_train = (credit_card_balance['SK_ID_CURR'].isin(SK_ID_CURR_train)) & (credit_card_balance['SK_ID_PREV'].isin(SK_ID_PREV_train))
    filter_credit_val = (credit_card_balance['SK_ID_CURR'].isin(SK_ID_CURR_val)) & (credit_card_balance['SK_ID_PREV'].isin(SK_ID_PREV_val))
    filter_credit_test = (credit_card_balance['SK_ID_CURR'].isin(SK_ID_CURR_test)) & (credit_card_balance['SK_ID_PREV'].isin(SK_ID_PREV_test))

    credit_card_balance_train = credit_card_balance[filter_credit_train]
    credit_card_balance_val = credit_card_balance[filter_credit_val]
    credit_card_balance_test = credit_card_balance[filter_credit_test]

    # Lưu các tập dữ liệu đã chia vào thư mục processed
    application_train_train.to_csv(PROCESSED_DIR / "application_train_train.csv", index=False)
    application_train_val.to_csv(PROCESSED_DIR / "application_train_val.csv", index=False)
    application_train_test.to_csv(PROCESSED_DIR / "application_train_test.csv", index=False)

    bureau_train.to_csv(PROCESSED_DIR / "bureau_train.csv", index=False)
    bureau_val.to_csv(PROCESSED_DIR / "bureau_val.csv", index=False)
    bureau_test.to_csv(PROCESSED_DIR / "bureau_test.csv", index=False)

    bureau_balance_train.to_csv(PROCESSED_DIR / "bureau_balance_train.csv", index=False)
    bureau_balance_val.to_csv(PROCESSED_DIR / "bureau_balance_val.csv", index=False)
    bureau_balance_test.to_csv(PROCESSED_DIR / "bureau_balance_test.csv", index=False)

    previous_application_train.to_csv(PROCESSED_DIR / "previous_application_train.csv", index=False)
    previous_application_val.to_csv(PROCESSED_DIR / "previous_application_val.csv", index=False)
    previous_application_test.to_csv(PROCESSED_DIR / "previous_application_test.csv", index=False)

    POS_CASH_balance_train.to_csv(PROCESSED_DIR / "POS_CASH_balance_train.csv", index=False)
    POS_CASH_balance_val.to_csv(PROCESSED_DIR / "POS_CASH_balance_val.csv", index=False)
    POS_CASH_balance_test.to_csv(PROCESSED_DIR / "POS_CASH_balance_test.csv", index=False)

    installments_payments_train.to_csv(PROCESSED_DIR / "installments_payments_train.csv", index=False)
    installments_payments_val.to_csv(PROCESSED_DIR / "installments_payments_val.csv", index=False)
    installments_payments_test.to_csv(PROCESSED_DIR / "installments_payments_test.csv", index=False)

    credit_card_balance_train.to_csv(PROCESSED_DIR / "credit_card_balance_train.csv", index=False)
    credit_card_balance_val.to_csv(PROCESSED_DIR / "credit_card_balance_val.csv", index=False)
    credit_card_balance_test.to_csv(PROCESSED_DIR / "credit_card_balance_test.csv", index=False)

if __name__ == "__main__":
    split_data()