from dotenv import load_dotenv
import os

# tránh lỗi segmentation fault khi sử dụng nhiều thư viện ML/DL
# do khi load lightgbm (ML) và FlagEmbedding (DL) cùng lúc, thì 2 thư viện này đều sử dụng OpenMP trong cùng 1 process, 
# dẫn đến lỗi segmentation fault khi chạy chương trình.
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.utils.agent_utils import GeminiModel, HFModel
from src.agent.credit_agent import CreditRiskAgent, AgentState
from langgraph.graph import StateGraph, START, END
import sys
from src.utils.get_data_from_DB import get_data_based_on_id
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from src.datapipeline.custom_transformer import *
from src.datapipeline.feature_engineering_pipeline import *
from src.datapipeline.main_pipelines import *
from src.datapipeline.merge_pipeline import *
from src.agent.constant_values import global_feature_importance
from src.utils.personal_shap import LocalInterpretation
import logging

merge_pipeline:Pipeline
final_dataset:pd.DataFrame

logging.getLogger("google_genai").setLevel(logging.ERROR)
load_dotenv(override=True)  # Load environment variables from .env file
GEMINI_API = os.getenv('GEMINI_API')
HF_TOKEN = os.getenv('HF_TOKEN')
DATABASE_URL = "postgresql+psycopg://thaithuyduc:Duc12345@localhost:5400/credit_risk"
# ID = 373139 # id for False Positive
# ID = 187760 # id for False Positive
# ID = 418142 # id for True Positive
# ID = 225853 # id for True Negative
ID = 427696 # id for False Negative
PROJECT_ROOT = Path(__file__).resolve().parents[0]

merge_pipeline = joblib.load(PROJECT_ROOT / "models/artifacts/merge_pipeline.joblib")
calibrated_lgb_with_weight_tuned_pipeline = joblib.load(PROJECT_ROOT / "models/saved_models/calibrated_lgb_with_weight_tuned_pipeline.joblib")
backgound_data = pd.read_csv(PROJECT_ROOT / "data/processed/background_data_1000.csv")
optimal_threshold_calibrated = 0.08644900452917655

def test_main():
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    gemini_llm = GeminiModel(GEMINI_API, model_name="gemini-3-flash-preview").init_gemini_model()
    hf_llm = HFModel(api_key=HF_TOKEN, model_name="deepseek-ai/DeepSeek-V4-Flash", provider="novita").init_hf_model()

    dict_result = get_data_based_on_id(db, ID)

    bureau_balance = dict_result["bureau_balance"]
    POS_CASH_balance = dict_result["POS_CASH_balance"]
    application_train = dict_result["application_train"]
    credit_card_balance = dict_result["credit_card_balance"]
    previous_application = dict_result["previous_application"]
    bureau = dict_result["bureau"]
    installments_payments = dict_result["installments_payments"]
    
    final_dataset = merge_pipeline.transform(bureau_balance, POS_CASH_balance, application_train, credit_card_balance, \
                                        previous_application, bureau, installments_payments)
    X_test = final_dataset.drop(columns=['AT_TARGET', 'SK_ID_CURR'])
    y_test = final_dataset['AT_TARGET']

    local_interpreter = LocalInterpretation(background_data=backgound_data, estimator=calibrated_lgb_with_weight_tuned_pipeline, optimal_threshold=optimal_threshold_calibrated)
    prediction_result = local_interpreter.get_top_features_with_shap_values(X_test, top_features=10)
    
    initial_state = {
        'global_feature_contribution': global_feature_importance,
        'optimal_threshold': optimal_threshold_calibrated,
        'prob_positive': prediction_result['probability for positive class'],
        'baseline_value': prediction_result['baseline value'],
        'late_payment': prediction_result['late payment?'],
        'top_feature_contribution': prediction_result['top feature attribution']
    }

    df_map = {
        'application_train': application_train,
        'bureau_balance': bureau_balance,
        'bureau': bureau,
        'credit_card_balance': credit_card_balance,
        'installments_payments': installments_payments,
        'POS_CASH_balance': POS_CASH_balance,
        'previous_application': previous_application
    }

    credit_risk_agent = CreditRiskAgent(hf_llm, df_map=df_map)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("customer_summarisation_node", credit_risk_agent.customer_summarize_node)
    workflow.add_node("customer_risk_explaination_node", credit_risk_agent.customer_credit_risk_explaination_node)
    workflow.add_node("planner_node", credit_risk_agent.planner_node)
    workflow.add_node('tool_execution_node', credit_risk_agent.tool_execution_node)
    workflow.add_node("final_report_node", credit_risk_agent.final_report_node)
    workflow.add_node('synthesizer_node', credit_risk_agent.synthesizer_node)

    workflow.add_edge(START, "customer_summarisation_node")
    workflow.add_edge(START, "customer_risk_explaination_node")
    workflow.add_edge("customer_summarisation_node", "planner_node")
    workflow.add_edge("customer_risk_explaination_node", "planner_node")
    workflow.add_edge("planner_node", 'tool_execution_node')
    workflow.add_edge('tool_execution_node', 'synthesizer_node')

    workflow.add_conditional_edges(
        'synthesizer_node',
        credit_risk_agent.should_continue_react,
        {
            "planner_node": 'planner_node',
            "final_report_node": 'final_report_node'
        }
    )
    
    workflow.add_edge("final_report_node", END)

    app = workflow.compile()

    result = app.invoke(initial_state)

    # report = "\n\n".join([
    #     "# Customer Credit Risk Report",
    #     "## Customer Summary",
    #     result["customer_summary"][0]['text'],
    #     "## Customer Credit Risk Explanation",
    #     result["customer_credit_risk_explaination"][0]['text']
    # ])

    # with open(PROJECT_ROOT / "reports/report.md", "w", encoding="utf-8") as f:
    #     f.write(report)

    # print("Report generated successfully. Check the 'reports/report.md' file for details.")

    return result

if __name__ == "__main__":
    result = test_main()
    joblib.dump(result, PROJECT_ROOT / "reports/result.joblib")
    print("Result saved successfully. Check the 'reports/result.joblib' file for details.")
    
    
    
    
    
    
    

    
    

    

    
    
