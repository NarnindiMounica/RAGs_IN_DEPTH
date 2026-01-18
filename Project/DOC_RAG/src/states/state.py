"RAG state definition for langgraph"

from pydantic import BaseModel
from typing import List
from langchain_core.documents import Document

class RAGState(BaseModel):

    question:str
    retrieved_documents:List[Document]=[]
    answer:str=""