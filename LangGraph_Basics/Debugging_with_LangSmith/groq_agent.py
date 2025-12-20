# importing libraries

#pip install langgraph-cli[inmem]

import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

from typing_extensions import TypedDict
from typing import Annotated, List
from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, START, END

load_dotenv()

os.environ['GROQ_API_Key'] = os.getenv("GROQ_API_KEY")
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGSMITH_TRACING_V2'] = "true"
os.environ['PROJECT_NAME'] = "Debugging Demo" 

#state schema

class State(TypedDict):
    messages: Annotated[List, add_messages]

#graph function definition

def graph_builder_func(state:State):
    graph = StateGraph(State)

    def simple_bot(state:State):
        model = init_chat_model(model="groq:llama-3.1-8b-instant")
        return {"messages": model.invoke(state['messages'])}

    graph.add_node("simple_bot", simple_bot)

    graph.add_edge(START, "simple_bot")
    graph.add_edge("simple_bot", END)

    graph_builder = graph.compile()
    return graph_builder

# graph invoking

agent=graph_builder_func(State)



