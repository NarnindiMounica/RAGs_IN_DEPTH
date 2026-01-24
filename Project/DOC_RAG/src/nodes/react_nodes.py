"""Langgraph nodes for RAG workflow + React agent inside generate_content"""

from typing import List, Optional
from src.states.state import RAGState

from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

