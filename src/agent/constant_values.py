table_definitions = {
    "application_train": [
        "Contains info about current loan and loan applicant at application time."
    ],
    "bureau": [
        "All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have current loan).",
        "For every current loan, there are as many rows as number of credits the client had in Credit Bureau before the application date."
    ],
    "bureau_balance": [
        "Contains monthly balances of previous credits in Credit Bureau.",
        "Has one row for each month of history of every previous credit reported to Credit Bureau."
    ],
    "POS_CASH_balance": [
        "Contains monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.",
        "Has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to current loan."
    ],
    "credit_card_balance": [
        "Contains monthly balance snapshots of previous credit cards that the applicant has with Home Credit.",
        "Has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to current loan."
    ],
    "previous_application": [
        "All previous applications for Home Credit loans of clients who have current loan.",
        "There is one row for each previous application related to current loan."
    ],
    "installments_payments": [
        "Repayment history for the previously disbursed credits in Home Credit related to current loan.",
        "There is a) one row for every payment that was made plus b) one row each for missed payment.",
        "One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to current loan."
    ] 
}

global_feature_importance = ['AT_EXT_SOURCE_2', 'AT_EXT_SOURCE_3', 'AT_DAYS_EMPLOYED',
    'AT_AMT_GOODS_PRICE', 'AT_AMT_CREDIT', 'AT_CODE_GENDER',
    'AT_DAYS_BIRTH', 'B_DAYS_CREDIT_max', 'AT_DAYS_ID_PUBLISH',
    'AT_AMT_ANNUITY', 'PC_CNT_INSTALMENT_FUTURE_mean_mean',
    'AT_NAME_EDUCATION_TYPE', 'PC_CNT_INSTALMENT_FUTURE_mean_max',
    'IP_AMT_PAYMENT_mean_max', 'B_AMT_CREDIT_SUM_max',
    'PA_CNT_PAYMENT_mean', 'PA_HOUR_APPR_PROCESS_START_mean',
    'IP_AMT_PAYMENT_mean_mean', 'AT_DAYS_REGISTRATION',
    'B_AMT_CREDIT_SUM_DEBT_max', 'AT_TOTALAREA_MODE',
    'PA_SELLERPLACE_AREA_mean', 'B_CREDIT_TYPE_Mortgage_mean',
    'PA_NAME_CLIENT_TYPE_New_mean', 'PC_SK_DPD_DEF_max_mean'
]