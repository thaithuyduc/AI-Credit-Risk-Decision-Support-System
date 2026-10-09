import pandas as pd
from src.agent.constant_values import table_definitions
from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import InferenceClient
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Any, Sequence, Optional
from pydantic import Field, ConfigDict
import json
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

class HFModel:
    def __init__(self, api_key: str, model_name: str = "deepseek-ai/DeepSeek-V4-Flash", provider: str = "novita"):
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        self.base_url = 'https://router.huggingface.co/v1'

    def init_hf_model(self):
        hf_model = ChatOpenAI(
            model = f'{self.model_name}:{self.provider}',
            api_key = self.api_key,
            max_tokens = 50000,
            base_url = self.base_url
        )
        return hf_model

class GeminiModel:
    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        self.api_key = api_key
        self.model_name = model_name

    def init_gemini_model(self):
        gemini = ChatGoogleGenerativeAI(
            model=self.model_name,
            max_output_tokens=50000,
            temperature=1.0,
            top_p=0.8,
            google_api_key=self.api_key # Use your actual API key here
        )
        return gemini

def create_feature_description(for_dict:dict, description_file_path:str) -> dict:
    description = pd.read_csv(description_file_path, encoding='latin1')
    desc_lookup = {}
    feature_descriptions = {}
    
    # Normalize table names if needed (optional but safer)
    table_name_map = {
        "application_train": "application_{train|test}",
        "bureau": "bureau",
        "bureau_balance": "bureau_balance",
        "credit_card_balance": "credit_card_balance",
        "installments_payments": "installments_payments",
        "POS_CASH_balance": "POS_CASH_balance",
        "previous_application": "previous_application",
    }
    
    for _, row in description.iterrows():
        table = row["Table"]
        feature = row["Row"]
        desc = row["Description"]
        special = row["Special"]
        desc_lookup[f"{table}_{feature}"] = f"{desc} (Special: {special})"
    
    for table, features in for_dict.items():
        feature_descriptions[table] = {}
        
        # Get the official table name used in the description file
        official_table = table_name_map.get(table)
        
        for feature, values in features.items():
            # Try exact match first
            desc = desc_lookup.get(f"{official_table}.csv_{feature}", 1)
            
            feature_descriptions[table][feature] = desc
    
    return feature_descriptions

def get_table_definitions(detailed_dict:dict):
    selected_tables = detailed_dict.keys()
    table_definitions_text = "\n".join(
        f"- {table_name}:\n" +
        "\n".join(f"    - {definition}" for definition in table_definitions[table_name])
        for table_name in selected_tables
    )

    return table_definitions_text

def create_feature_dict(features_name: List[str], df_map: dict) -> dict:
    dict_ = {
        'application_train': [],
        'bureau_balance': [],
        'bureau': [],
        'credit_card_balance': [],
        'installments_payments': [],
        'POS_CASH_balance': [],
        'previous_application': []
    }
    detailed_dict_ = {}

    for feature in features_name:
        prefix = feature.split("_")[0]
        remain = feature.split("_")[1:]
        filtered_remain = [x for x in remain if x.isupper() or x.isnumeric()]
        feature = "_".join(filtered_remain)

        if prefix == "AT":
            dict_['application_train'].append(feature)
        elif prefix == "B":
            dict_['bureau'].append(feature)
        elif prefix == "BB":
            dict_['bureau_balance'].append(feature)
        elif prefix == "CC":
            dict_['credit_card_balance'].append(feature)
        elif prefix == "IP":
            dict_['installments_payments'].append(feature)
        elif prefix == "PC":
            dict_['POS_CASH_balance'].append(feature)
        elif prefix == "PA":
            dict_['previous_application'].append(feature)

    dict_ = {k: list(dict.fromkeys(v)) for k, v in dict_.items() if len(v) > 0}

    for name, features in dict_.items():          # features is a list
        detailed_dict_[name] = {}
        df = df_map[name]
        
        for feature in features:                  # iterate over each column name
            # .values → numpy array, .tolist() → Python list
            detailed_dict_[name][feature] = df[feature].values.tolist()

    return detailed_dict_