from dotenv import load_dotenv
import os
from langchain_classic.retrievers import MultiVectorRetriever
from FlagEmbedding import FlagAutoModel
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.indexes import SQLRecordManager, index
from langchain_core.documents import Document
import re
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from pathlib import Path
from typing import List

class FlagEmbedding(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        super().__init__()
        self.model = FlagAutoModel.from_finetuned(model_name_or_path=model_name, use_fp16=True, normalize_embeddings=True # L2 normalization
                                        # query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                        )
    
    def embed_documents(self, documents: list[str]):
        return self.model.encode(documents)['dense_vecs']
    
    def embed_query(self, query: str):
        return self.model.encode(query)['dense_vecs']

class DatabaseManager:
    def __init__(self, vector_db_host: str, model_name: str = "BAAI/bge-m3"):
        self.embedding_model = FlagEmbedding(model_name=model_name)
        self.qdrant_client = QdrantClient(prefer_grpc=False, url=vector_db_host)

    def create_database_instances(self, collection_namespace_name: str):
        qdrant_vector_store = QdrantVectorStore(
            client=self.qdrant_client, 
            collection_name=collection_namespace_name, 
            embedding=self.embedding_model
        )

        return qdrant_vector_store

class LoadProcessData:
    def __init__(self):
        pass

    def load_data(self, data_path: str):
        # loader = WebBaseLoader(data_path, encoding='utf-8')
        text = Path(data_path).read_text(encoding='utf-8')
        documents = [Document(page_content=text, metadata={"source": data_path})]
        
        return documents

    def clean_text(self, text: str) -> str:
        # 1. Replace tabs and multiple spaces with single space
        text = re.sub(r'\t+', ' ', text)
        text = re.sub(r' +', ' ', text)
        
        # 2. Normalize newlines: multiple \n → single \n, but keep paragraphs
        text = re.sub(r'\n\s*\n', '\n\n', text)   # preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 3. Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_text(self, documents, chunk_size=600, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        return texts

    def create_data_chunks(self, data_path: str, chunk_size=600, chunk_overlap=200):
        documents = self.load_data(data_path)
        cleaned_documents = [Document(page_content=self.clean_text(doc.page_content), metadata=doc.metadata) for doc in documents]
        text_chunks = self.split_text(cleaned_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_chunks

class RAGIndexing:
    """
    This is naive implementation of RAG indexing. It only uses Qdrant as vector database
    """
    def __init__(self, vector_db_host: str, model_name: str = "BAAI/bge-m3"):
        self.embedding_model = FlagEmbedding(model_name=model_name)
        self.qdrant_client = QdrantClient(prefer_grpc=False, url=vector_db_host)
        self.load_process_data = LoadProcessData()
        
    def init_vector_db(self):
        # 1. Check if the collection exists in Qdrant
        if not self.qdrant_client.collection_exists(collection_name='credit_risk_policy'):
            # 2. Create the collection if it's missing (bge-m3 uses 1024 dimensions)
            self.qdrant_client.create_collection(
                collection_name='credit_risk_policy',
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

        # 3. Initialize Langchain's QdrantVectorStore now that we know the collection exists
        qdrant_vector_store = QdrantVectorStore(
            client=self.qdrant_client, 
            collection_name='credit_risk_policy', 
            embedding=self.embedding_model
        )
        
        return qdrant_vector_store
    
    def import_and_index_data(self, data_path: str):
        """ This function will import data from the given path, process it, and index it into Qdrant and SQL database. 
        It also implements MultiVectorRetriever with MongoDB as docstore to store the actual text chunks and use Qdrant just for vector search.
        In MultiVectorRetriever, we use Parent-Child Chunking strategy.
        """
        qdrant_vector_store = self.init_vector_db()
        parent_text_chunks = self.load_process_data.create_data_chunks(data_path, chunk_size=600, chunk_overlap=300)
        
        qdrant_vector_store.add_documents(parent_text_chunks)
        
        print(f"Successfully indexed data from {data_path} into Qdrant collection")

class RAGRetriever:
    def __init__(self, vector_db_host: str, model_name: str = "BAAI/bge-m3"):
        db_manager = DatabaseManager(vector_db_host=vector_db_host, model_name=model_name)
        self.qdrant_vector_store = db_manager.create_database_instances(collection_namespace_name='credit_risk_policy')

    def retrieve(self, query:List[str]) -> List[Document]:
        total_results = []
        for q in query:
            results = self.qdrant_vector_store.similarity_search(q, k=3)
            total_results.extend(results)
        return total_results
            
# def main():
#     load_dotenv(override=True)  # Load environment variables from .env file
#     data_path = Path(__file__).resolve().parent / 'credit_risk_policy.txt'
#     vector_db_host = os.getenv("VECTOR_DB_HOST")

#     rag_indexing = RAGIndexing(vector_db_host=vector_db_host)
#     rag_indexing.import_and_index_data(data_path)

# if __name__ == "__main__":
#     main()