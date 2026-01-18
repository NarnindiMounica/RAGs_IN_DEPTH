"Vectorstore module for document embedidng and retrieval"

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorStoreManage:
    "manages vectorstore application"

    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever = None

    def create_retriever(self, documents:List[Document]):
        """
        Docstring for create_retriever
        
        create vectorstore from documents

        Args:
        documents: list of documents to embed

        """

        self.vectorstore = FAISS.from_documents(documents=documents,
                                                embedding = self.embedding)
        
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        """
        Docstring for get_retriever
        
        get the retriever instance

        Returns:
        retriever instance
        """
        if not self.retriever:
            raise ValueError("Vectorstore not initialized. Call create_retriever first.")
        return self.retriever
    
    def retrieve(self, query:str, k:int=4)->List[Document]:
        """
        Docstring for retrieve
        
        Retrieve relevant documents for a query

        Args:
        query:search query
        k: Number of documents to retrieve

        Returns:
        List of relevant documents

        """
        if self.retriever is None:
            raise ValueError("Vectorstore not intialized, Call create_retriever first.")
        return self.retriever.invoke(query)