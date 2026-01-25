"""Langgraph nodes for RAG workflow + React agent inside generate_content"""

from typing import List, Optional
from src.states.state import RAGState

from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

class ReactRAGNodes:
    "contains the node functions for RAG workflow"

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None #Lazy init agent

    def  retrieve_docs(self, state:RAGState)->RAGState:
        "classic retrivere node"
        docs = self.retriever.invoke(state.question) 
        return RAGState(question=state.question,
                        retrieved_docs= docs) 
    
    def _build_tools(self):
        "build retriever + wikipedia tools"



    
    def _build_agent(self):
        "react agent with tools"
        tools = self._build_tools()
        system_prompt = """
            you are a helpful RAG agent. Prefer 'retriever' for
            user-provided docs; use 'wikipedia' for general knowledge.
            Return only the final useful answer."""
        self._agent = create_agent(
            self.llm,
            tools = tools,
            prompt = system_prompt
        )
        