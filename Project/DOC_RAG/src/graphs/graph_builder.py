from langgraph.graph import StateGraph, START, END
from src.states.state import RAGState
from src.nodes.rag_nodes import RAGNodes

class GraphBuilder:
    "builds and manages langgraph workflow"
    def __init__(self):
        self.builder = StateGraph(RAGState)
        self.nodes = RAGNodes()

    def build(self):
        self.builder.add_node("retriever", self.nodes.retrieve_docs)
        self.builder.add_node("generator", self.nodes.generator)

        self.builder.add_edge(START, "retriever")
        self.builder.add_edge("retriever", "generator")
        self.builder.add_edge("generator", END)

        self.graph = self.builder.compile()
        return self.graph
    
    def run(self, question:str)->dict:

        if self.graph is None
            self.build()

        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)



    
