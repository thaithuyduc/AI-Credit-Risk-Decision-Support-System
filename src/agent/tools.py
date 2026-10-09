from langchain_core.tools import tool
from typing import List
from src.rag.indexing_retrieve import RAGRetriever
import os

vector_db_host = os.getenv("VECTOR_DB_HOST")
rag_indexing = RAGRetriever(vector_db_host=vector_db_host, model_name="BAAI/bge-m3")

@tool
def search_rag(queries: List[str]|str) -> str:
    """
    Search for relevant information using RAG (Retrieval-Augmented Generation) approach.
    
    Args:
        queries (List[str]): A list of query strings to search for relevant information.
    
    Returns:
        str: A string containing the merged information retrieved based on the queries.
    """
    if isinstance(queries, str):
        queries = [queries]

    total_infomation = rag_indexing.retrieve(queries)
    total_infomation = [info.page_content for info in total_infomation]
    total_infomation = list(set(total_infomation))  # Remove duplicates

    merged_information = "\n\n".join(total_infomation)

    return merged_information