"""

"""
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from typing import TypedDict, List, Any, Annotated, Optional
import pandas as pd
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.constant_values import table_definitions, global_feature_importance
from src.utils.agent_utils import create_feature_description, get_table_definitions, create_feature_dict, GeminiModel
from src.agent.tools import search_rag
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import filter_messages

#========
# Define constant values used for the agent's state and workflow
#========
load_dotenv(override=True)  # Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEMINI_API = os.getenv('GEMINI_API')

# Define Agent State
class AgentState(TypedDict):
    global_feature_contribution: List[str] # Những đặc trưng quan trọng nhất cho mọi điểm dữ liệu, đây là các feature mà Tree models cho là quan trọng nhất trong quá trình huấn luyện (global interpretation)
    optimal_threshold: float      # Ngưỡng tối ưu để phân loại khách hàng có khả năng trễ hạn thanh toán, dựa trên ML model
    prob_positive: float          # Xác suất dự đoán của nhãn Positive
    baseline_value: float         # Giá trị baseline từ SHAP
    late_payment: str             # Có thanh toán trễ hay không (Boolean)
    top_feature_contribution: dict# Đóng góp của các đặc trưng quan trọng nhất (theo SHAP) vào xác suất dự đoán (local interpretation)
    messages: Annotated[List[BaseMessage], add_messages] # Danh sách các tin nhắn đã được gửi và nhận trong quá trình hội thoại với LLM
    is_sufficient: bool # Các thông tin từ RAG đã đủ để tổng hợp thành các khuyến nghị cho khách hàng hay chưa
    queries_for_rag: List[str] # Các câu truy vấn được tạo ra từ planner_node để gửi đến RAG nhằm lấy thông tin chính sách liên quan đến tình huống của khách hàng
    additional_information_needed: Optional[str] # Nếu thông tin từ RAG chưa đủ, thì cần thêm thông tin gì và tại sao cần thêm thông tin đó
    react_iterations: int # Số lần lặp lại của quá trình Reasoning-Acting-Observing (RAG) để thu thập thông tin từ RAG, nhằm đảm bảo rằng các thông tin từ RAG là đủ để đưa ra khuyến nghị cuối cùng cho khách hàng
    
    customer_summary: str         # Một bản tóm tắt thông tin khách hàng
    customer_credit_risk_explaination: str # Một bản báo cảo xác xuất khách hàng trễ hạn thanh toán trong đơn vay hiện tại, kèm theo lời giải thích
    final_recommendation: Optional[str] # Nếu thông tin từ RAG đủ, thì đưa ra khuyến nghị cuối cùng cho khách hàng
    final_report: str # Một bản báo cáo tổng hợp cuối cùng, bao gồm customer_summary, customer_credit_risk_explaination, và final_recommendation (nếu có)

# Define agent workflow
class CreditRiskAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI|ChatOpenAI, df_map:dict[str, pd.DataFrame]):
        self.llm = llm
        self.df_map = df_map
    
    def customer_summarize_node(self, state: AgentState):
        """
        Tạo ra một bản tóm tắt hồ sơ khách hàng, về các lịch sử giao dịch trong quá khứ, hành vi khách hàng, lý lịch của khách hàng,...
        """
        print("--- Creating customer summarisation report ---")
        global_feature_contribution = state["global_feature_contribution"] # output a dictionary
        
        customer_info = create_feature_dict(global_feature_contribution, self.df_map)

        table_definitions_text = get_table_definitions(customer_info)

        feature_descriptions = create_feature_description(for_dict=customer_info, description_file_path="/Users/thuyduc/Projects/AI/AI Sandbox/Services/CreditRiskAgent/data/raw/HomeCredit_columns_description.csv")
        
        sys_message = SystemMessage(
            content=f"""
            You are a professional Credit Risk Analyst Agent.

            Your task is to carefully read the customer's data (provided by user) and generate a clear, structured, and insightful summary of the customer.

            ### Input Data Structure
            You will receive a dictionary with the following format:
            {{
                "table_name": {{
                    "feature_name": [value1, value2, ...],
                    ...
                }},
                ...
            }}

            - Some features contain a single value (from `application_train`).
            - Other features contain a list of values (from related tables like `bureau`, `installments_payments`, `POS_CASH_balance`, etc.).

            ### Table Definitions
            {table_definitions_text}

            ### Feature Definitions
            {feature_descriptions}

            ### Instructions
            1. Analyze every feature in given dictionary.
            2. Use the table and feature definitions to correctly interpret the meaning of each value.
            3. Pay special attention to:
            - Missing values (NaN)
            - Negative values (especially days-related features → they represent days before application)
            - Multiple values in related tables (summarize patterns, averages, extremes, trends)
            4. Write a professional and comprehensive customer summary that covers all useful insights. Structure the summary into the following sections (use these exact headings):
                **1. Demographic & Personal Profile**
                - Age, gender, education level, and other personal characteristics.
                - Housing / registration related information (if available).

                **2. Employment & Income Situation**
                - Current employment duration and stability.
                - Any available income or annuity information.

                **3. Current Loan Application**
                - Loan amount, goods price, annuity, and other key application details.
                - External credit scores (EXT_SOURCE) if available.

                **4. Credit History Overview**
                - Number and types of previous credits (from bureau / previous applications).
                - Credit amounts, outstanding debt, and credit types.
                - How recent the previous credits are.

                **5. Payment Behavior & History**
                - Payment patterns from installments (whether the customer paid the full installment amount, underpaid, or overpaid).
                - Frequency and consistency of payments.
                - Any signs of late payments or difficulties (if available from DPD or similar features).

                **6. Ongoing Credit Obligations**
                - Remaining installments (CNT_INSTALMENT_FUTURE).
                - Current status of previous POS / cash loans or other active products.

                **7. Overall Behavioral Insights & Risk Signals**
                - Summarize the customer's overall credit behavior (responsible / risky / mixed).
                - Highlight any positive signals (good payment history, stable employment, etc.).
                - Highlight any concerning signals (high debt, missing data, irregular payments, short credit history, etc.).
                - Do NOT give a final approve/reject decision — only describe observations.

            ### Output Format
            Write in clear, professional English. Be specific and use numbers from the data whenever possible. Do not include any extra explanation or markdown code blocks.
            """
        )

        human_message = HumanMessage(
            content=f"""Here is the customer's detailed data: 
            {customer_info}
            Please generate a professional summary of this customer.""")
        
        messages = [
            sys_message,
            human_message
        ]
        
        result = self.llm.invoke(messages)
        
        return {"customer_summary": result.content}
    
    def customer_credit_risk_explaination_node(self, state: AgentState):
        """
        Tạo ra một bản báo cáo xác xuất khách hàng trễ hạn thanh toán trong đơn vay hiện tại từ mô hình ML, kèm theo lời giải thích tại sao khách hàng có xác suất trễ hạn thanh toán cao hoặc thấp, dựa trên các đặc trưng quan trọng nhất (theo SHAP).
        """
        print("--- Creating customer credit risk explaination report ---")
        df_name_map = {
            "AT": "application_train",
            "B": "bureau",
            "BB": "bureau_balance",
            "CC": "credit_card_balance",
            "IP": "installments_payments",
            "PC": "POS_CASH_balance",
            "PA": "previous_application",
        }
        
        prob_positive = state["prob_positive"]
        baseline_value = state["baseline_value"]
        late_payment = state["late_payment"]
        top_feature_contribution = state["top_feature_contribution"]
        
        dict_ = create_feature_dict(list(top_feature_contribution.keys()), self.df_map)
        
        final_dict = {
            table: {
                feature: {
                    "value": values
                }
                for feature, values in features.items()
            }
            for table, features in dict_.items()
        }

        for key, value in top_feature_contribution.items():
            prefix = key.split("_")[0]
            remain = key.split("_")[1:]
            filtered_remain = [x for x in remain if x.isupper() or x.isnumeric()]
            table_name = df_name_map.get(prefix)
            feature_name = "_".join(filtered_remain)
            final_dict[table_name][feature_name]["contribution"] = float(value)

        table_definitions_text = get_table_definitions(final_dict)
        
        feature_descriptions = create_feature_description(for_dict=final_dict, description_file_path="/Users/thuyduc/Projects/AI/AI Sandbox/Services/CreditRiskAgent/data/raw/HomeCredit_columns_description.csv")
        
        sys_message = SystemMessage(
            content=f"""
            You are a professional Credit Risk Analyst Agent specialized in explaining late payment risk predictions.
            Your task is to generate a clear, professional, and easy-to-understand report about the late payment risk of a specific customer, based on the machine learning model prediction and SHAP explanations.
            ### Available Information
            - **prob_positive**: The Machine Learning model's predicted probability that this customer will have a late payment on the current loan (value between 0 and 1).
            - **baseline_value**: The average predicted probability across all customers (SHAP baseline). This is the starting point before considering this customer's specific features.
            - **late_payment**: Boolean indicating whether the Machine Learning model predicts this customer will have a late payment (`True`) or not (`False`).
            - **final_dict**: Dictionary containing the top features that most influence the prediction.
                Structure:
                {{
                    "table_name": {{
                        "feature_name": {{
                            "value": [actual value(s) of the customer],
                            "contribution": shap_value  # positive = increases risk, negative = decreases risk
                        }},
                        ...
                    }},
                    ...
                }}
            - **feature_descriptions**: Definitions of the features.
            - **table_definitions**: Description of each data table.
            ### Instructions for the Report
            Write a professional report with the following structure (use these exact section headings):

            **1. Late Payment Risk Prediction**
            - Clearly state the predicted probability (`prob_positive`) as a percentage.
            - State whether the Machine Learning model predicts the customer will have a late payment or not (`late_payment`).
            - Compare the customer's probability with the baseline (`baseline_value`).  
            Example: "The customer's risk is X% higher/lower than the average customer."

            **2. Key Drivers of the Prediction**
            - Explain the most important features from `final_dict` that push the prediction up or down from the baseline.
            - For each important feature:
            - Mention the actual value of the customer.
            - Explain what the feature means (use `feature_descriptions`).
            - State whether its contribution **increases** or **decreases** the late payment risk.
            - Quantify the impact when useful (e.g., "This feature increases the risk by approximately 0.018").
            - Group the explanation by positive contributors (increase risk) and negative contributors (decrease risk) if it improves clarity.
            - Focus on the top features that have the largest absolute contribution.

            **3. Overall Risk Assessment**
            - Summarize the main reasons why this customer has higher or lower risk than average.
            - Highlight any notable patterns (e.g., education level, income type, credit history status, loan amount, etc.).
            - Keep the tone professional, objective, and neutral. Do **not** give a final approve/reject recommendation.

            ### Important Rules
            - Always use the actual numbers from `prob_positive`, `baseline_value`, and the `contribution` values.
            - Explain in clear business language (avoid pure technical jargon).
            - When a feature has multiple values (e.g. from bureau), summarize the pattern instead of listing every single value.
            - Do not invent information that is not provided.
            - Return **only** the report using the exact section headings above.  
            Do not add any introductory sentence, conclusion outside the structure, or markdown code blocks.
            """
        )
        
        human_message = HumanMessage(
            content=f"""
            Here is the information for this customer:

            - Predicted probability of late payment (prob_positive): {prob_positive}
            - Baseline probability (baseline_value): {baseline_value}
            - Machine Learning Model prediction (late_payment): {late_payment}

            Top contributing features (final_dict):
            {final_dict}

            Feature definitions:
            {feature_descriptions}

            Table definitions:
            {table_definitions_text}

            Please generate the late payment risk report.
        """
        )

        messages = [
            sys_message,
            human_message
        ]
        result = self.llm.invoke(messages)
        
        return {'customer_credit_risk_explaination': result.content}

    def planner_node(self, state: AgentState):
        """
        Đối chiếu thông tin từ customer_summary và customer_credit_risk_explaination cùng với
        optimal_threshold và lý luận xem khách hàng bị dự đoán trễ hạn thanh toán có đáng bị như thế không, do quyết định từ mô hình ML
        đôi khi không chính xác, vì có những khách hàng có xác xuất trễ hạn cao hơn threshold nhưng vẫn có khả năng thanh toán đúng hạn, và ngược lại.
        Từ đó đưa ra kết luận khách hàng này có cần manual review hay không, cuối cùng, tạo ra những câu truy vấn có liên quan đến tình huống để dùng cho RAG sau này
        """
        class PlannerOuput(BaseModel):
            """
            Give a clear conclusion on whether the customer requires manual review or not, and generate a list of relevant queries for RAG.
            """
            conclusion_explaination: Optional[str]
            '''The conclusion and reasoning on whether the customer requires manual review or not, based on the analysis of customer_summary, customer_credit_risk_explaination, and optimal_threshold.'''
            queries_for_rag: List[str]
            '''A list of relevant queries that could be sent to a Retrieval-Augmented Generation (RAG) system to gather additional information or policy clarifications for this specific customer case.'''
        
        customer_summary = state["customer_summary"]
        customer_credit_risk_explaination = state["customer_credit_risk_explaination"]
        optimal_threshold = state["optimal_threshold"]
        additional_information_needed = state.get("additional_information_needed", None)

        if additional_information_needed is None:
            system_message = SystemMessage(
                content=f"""
                You are a professional Credit Risk Analyst Agent specialized in planning and decision-making based on customer summaries and credit risk explanations.

                ### Available Information
                - **customer_summary**: A detailed summary of the customer's profile, behavior, and history.
                - **customer_credit_risk_explaination**: A report explaining the predicted late payment risk for the customer, including key contributing features and their impacts.
                - **optimal_threshold**: The threshold probability used to classify customers as high or low risk for late payments.

                ### Instructions
                1. Analyze both the customer summary and the credit risk explanation.
                2. Determine if the customer's predicted late payment risk (from the ML model) is justified based on their profile and history.
                3. Decide whether this customer should be flagged for manual review or if they can be approved without further scrutiny.
                4. Generate three relevant queries that could be sent to a Retrieval-Augmented Generation (RAG) system to gather additional information or policy clarifications for this specific customer case.
            """
            )
            human_message = HumanMessage(
                content=f"""
                Here is the information for this customer:

                - Customer Summary:
                {customer_summary}

                - Customer Credit Risk Explanation:
                {customer_credit_risk_explaination}

                - Optimal Threshold for late payment risk classification: {optimal_threshold}

                Please analyze the information and provide your conclusion on whether this customer requires manual review. Additionally, generate three relevant queries for RAG.
            """
            )
        else:
            system_message = SystemMessage(
                content=f"""
                You are a professional Credit Risk Analyst Agent specialized in planning based on customer summaries and credit risk explanations.

                ### Available Information
                - **customer_summary**: A detailed summary of the customer's profile, behavior, and history.
                - **customer_credit_risk_explaination**: A report explaining the predicted late payment risk for the customer, including key contributing features and their impacts.
                - **additional_information_needed**: Specific information or clarification that is required to make a final recommendation for this customer.

                ### Instructions
                1. Analyze both the customer summary, the credit risk explanation and additional_information_needed.
                2. Based on additional_information_needed and context from customer_summary and customer_credit_risk_explaination, generate three relevant queries that could be sent to a Retrieval-Augmented Generation (RAG) system to gather the required information or policy clarifications for this specific customer case.
                3. Ensure that the queries are specific, clear, and directly related to the additional_information_needed.
            """
            )
            human_message = HumanMessage(
                content=f"""
                Here is the information for this customer:
                - Customer Summary:
                {customer_summary}
                - Customer Credit Risk Explanation:
                {customer_credit_risk_explaination}
                - Additional Information Needed:
                {additional_information_needed}

                Please generate three relevant queries for RAG to gather the required information or policy clarifications.
            """
            )
        
        planner_llm = self.llm.with_structured_output(PlannerOuput, method="function_calling")
        response = planner_llm.invoke([system_message, human_message])
        ai_message = AIMessage(content=response.conclusion_explaination)

        return {'messages': [ai_message], 'queries_for_rag': response.queries_for_rag}

    def tool_execution_node(self, state: AgentState):
        """
        Execute the RAG tool to retrieve relevant information based on the queries generated in the planner_node.
        """
        queries_for_rag = state["queries_for_rag"]
        rag_results = search_rag.invoke({"queries": queries_for_rag})
        
        return {
            'messages': [HumanMessage(content=f"Retrieved information from RAG:\n{rag_results}")]
        }

    def synthesizer_node(self, state: AgentState):
        '''
        Read information from customer_summary, customer_credit_risk_explaination, and information from history messagges (include: previous AI messages, Tool messages)
        Then decide whether the information from RAG is sufficient to generate a final recommendation for the customer. 
        If sufficient, generate a final recommendation; 
        if not, request more information or clarification by specifying what additional information is needed and why.
        '''
        class JudgementOutput(BaseModel):
            is_sufficient: bool
            '''A boolean indicating whether the information retrieved from RAG is sufficient to generate a final recommendation for the customer.'''
            final_recommendation: Optional[str]
            '''If is_sufficient is True, provide a clear and concise final conclusion, explanation, recommendation for the customer based on all available information.'''
            additional_information_needed: str
            '''If is_sufficient is False, specify what additional information or clarification is needed and why it is important for making a final recommendation.'''

        messages = state["messages"]
        customer_summary = state["customer_summary"]  #[0]['text']
        customer_credit_risk_explaination = state["customer_credit_risk_explaination"]  #[0]['text']

        # create messages with format: AI Message: ...., Tool Message: ....
        clear_messages = []
        ai_messages = filter_messages(messages, include_types=[AIMessage])
        human_messages = filter_messages(messages, include_types=[HumanMessage])
        
        for msg in ai_messages:
            clear_messages.append(f"AI Message: {msg.content}")
            break
        for msg in human_messages:
            clear_messages.append(f"Human Message: {msg.content}")

        clear_messages_text = "\n".join(clear_messages)

        system_message = SystemMessage(
            content=f"""
            You are a professional Credit Risk Analyst Agent specialized in synthesizing information and making final recommendations.

            ### Available Information
            - **customer_summary**: A detailed summary of the customer's profile, behavior, and history.
            - **customer_credit_risk_explaination**: A report explaining the predicted late payment risk for the customer, including key contributing features and their impacts.
            - **messages**: A list of previous messages, including AI messages and Human messages (from RAG), containing additional information or clarifications.

            ### Instructions
            1. Analyze the customer summary, credit risk explanation, and all previous messages.
            2. Determine if the information retrieved from RAG is sufficient to make a final recommendation regarding the customer's credit risk.
            3. If sufficient, provide a clear and concise final recommendation for the customer, do not fabricate information, use only the information provided.
            4. If not sufficient, specify what additional information or clarification is needed and explain why it is important for making a final recommendation.
        """
        )
        human_message = HumanMessage(
            content=f"""
            Here is the information for this customer:

            - Customer Summary:
            {customer_summary}

            - Customer Credit Risk Explanation:
            {customer_credit_risk_explaination}

            - Previous Messages:
            {clear_messages_text}

            Please analyze the information and provide your judgement on whether the information from RAG is sufficient to generate a final recommendation. If sufficient, provide the final conclusion, explanation and recommendation; if not, specify what additional information is needed and why.
        """
        )

        messages = [system_message, human_message]
        synthesizer_llm = self.llm.with_structured_output(JudgementOutput, method="function_calling")
        response = synthesizer_llm.invoke(messages)
        react_iterations = state.get("react_iterations", 0)
        if not response.is_sufficient:
            react_iterations += 1
            print(f'ReAct iterations: {react_iterations}')
        
        return {
            'is_sufficient': response.is_sufficient,
            'final_recommendation': response.final_recommendation,
            'additional_information_needed': response.additional_information_needed,
            'react_iterations': react_iterations
        }

    def final_report_node(self, state: AgentState):
        """
        Generate a final report that includes the customer summary, credit risk explanation, and the final recommendation (if available).
        """
        customer_summary = state["customer_summary"]
        customer_credit_risk_explaination = state["customer_credit_risk_explaination"]
        final_recommendation = state.get("final_recommendation", None)

        report_sections = [
            "# Customer Credit Risk Report",
            "## Customer Summary",
            customer_summary,
            "## Customer Credit Risk Explanation",
            customer_credit_risk_explaination
        ]

        if final_recommendation:
            report_sections.append("## Final Recommendation")
            report_sections.append(final_recommendation)

        final_report = "\n\n".join(report_sections)

        return {'final_report': final_report}
    
    def should_continue_react(self, state: AgentState):
        """
        Determine whether the agent should continue the Reasoning-Acting-Observing (RAG) loop based on the current state.
        """
        is_sufficient = state["is_sufficient"]
        react_iterations = state["react_iterations"]
        
        # Define a maximum number of iterations to prevent infinite loops
        MAX_ITERATIONS = 3
        
        if not is_sufficient and react_iterations < MAX_ITERATIONS:
            return "planner_node"
        else:
            return "final_report_node"

        