# Customer Credit Risk Report

## Customer Summary

**1. Demographic & Personal Profile**
The client is a 42-year-old male (15,400 days old) with a secondary or secondary special education level. He demonstrates high stability in his personal documentation and residency, having been registered at his current address for approximately 24.2 years (8,842 days) and having updated his identity document roughly 12.3 years (4,474 days) prior to the application. Data regarding the client’s living area (TOTALAREA_MODE) is currently unavailable.

**2. Employment & Income Situation**
The client has been at his current place of employment for approximately 5.7 years (2,091 days), indicating a stable professional situation. While specific total income data is not provided in this snapshot, the loan annuity for the current application is set at 13,891.5.

**3. Current Loan Application**
The client is applying for a credit amount of 269,550.0 to finance goods valued at 225,000.0. This results in a loan-to-value ratio where the credit exceeds the goods price by approximately 20%. The loan carries an annuity of 13,891.5. The external credit score (EXT_SOURCE_2) is 0.467, which is a moderate score; however, the EXT_SOURCE_3 score is missing, limiting the breadth of the external risk assessment.

**4. Credit History Overview**
The client's history with external financial institutions (Bureau data) is unavailable, as all relevant fields (DAYS_CREDIT, AMT_CREDIT_SUM) contain missing values. However, internal records show the client has one previous application with the institution. This previous loan was a consumer credit or cash loan with a defined term of 20 installments.

**5. Payment Behavior & History**
The client demonstrates an excellent internal payment record. Based on 20 recorded installment payments, the client consistently paid 6,141.015 per month (with one minor variation of 6,019.875). There are no recorded instances of days past due (SK_DPD_DEF = 0) across the entire history of the previous POS/CASH balance, indicating a high level of reliability in meeting scheduled obligations.

**6. Ongoing Credit Obligations**
The previous credit application had a term of 20 payments. The POS_CASH balance history shows the "installments left to pay" (CNT_INSTALMENT_FUTURE) decreasing from 20 down to 0. This suggests that the previous obligation has been fully satisfied or is in its final stages. There is no evidence of other active debt in the provided data.

**7. Overall Behavioral Insights & Risk Signals**
- **Positive Signals:** The client exhibits strong behavioral consistency, evidenced by a 5.7-year employment tenure and a perfect internal repayment history with no late payments (0 DPD). His long-term residency (24 years) further suggests a stable lifestyle.
- **Concerning Signals:** The primary risk factor is the lack of data from the Credit Bureau and the missing EXT_SOURCE_3 score, which creates a "thin file" profile regarding his behavior with other lenders. Additionally, the EXT_SOURCE_2 score of 0.467 is relatively average.
- **Summary:** The client is a middle-aged individual with a proven track record of meeting internal financial commitments. While his external credit footprint is invisible in this data, his internal stability and payment discipline are strong positive indicators.

## Customer Credit Risk Explanation

**1. Late Payment Risk Prediction**
- The predicted probability of a late payment for this customer is 11.89%.
- The Machine Learning model predicts that the customer will have a late payment (True).
- The customer's risk is 4.67% higher than the average customer baseline of 7.22%.

**2. Key Drivers of the Prediction**
The prediction is primarily influenced by the following factors, all of which contribute to an increase in the predicted risk:

- **Gender (CODE_GENDER):** The customer is male ('M'). This feature is the strongest driver in this profile, increasing the late payment risk by 0.0215.
- **Goods Price (AMT_GOODS_PRICE):** The price of the goods for which the loan is given is 225,000.0. This specific value increases the risk by 0.0072.
- **External Data Score (EXT_SOURCE_2):** The customer has a normalized score of 0.467 from an external data source. This score contributes to an increase in risk of 0.0065.
- **Education Level (NAME_EDUCATION_TYPE):** The customer’s highest education level is 'Secondary / secondary special'. This level of education increases the risk by 0.0028.
- **Credit Bureau Records (CREDIT_DAY_OVERDUE):** The data regarding days past due on previous credits is missing (nan). The absence of this specific history increases the risk by 0.002.
- **Credit Card Activity (CNT_DRAWINGS_OTHER_CURRENT and CNT_DRAWINGS_CURRENT):** There is no recorded history of the number of drawings on previous credit cards (nan). These missing values collectively increase the risk by approximately 0.003.
- **Previous Application Details (NAME_CLIENT_TYPE and NAME_SELLER_INDUSTRY):** The customer was a 'New' client during their previous application, and the seller was in the 'Furniture' industry. These factors increase the risk by 0.0015 and 0.0013, respectively.
- **Organization Type (ORGANIZATION_TYPE):** Working for a 'Business Entity Type 3' increases the risk by 0.0009.

**3. Overall Risk Assessment**
The customer presents a higher-than-average risk of late payment, with a predicted probability of 11.89% compared to the 7.22% baseline. The primary drivers for this elevated risk are demographic factors (specifically gender and education level) and the score provided by external data sources. Additionally, the specific loan amount for the goods purchased and the lack of detailed historical data regarding credit card drawings and Credit Bureau overdue status contribute to the model's cautious prediction. The combination of being a new client in previous interactions and the type of employment organization further reinforces the predicted likelihood of a late payment.