from src.states.state import RAGState

class RAGNodes:
    "contains node functions for RAG workflow"

    def __init__(self, retriever, llm):
        """intialize RAG nodes

        Args:
        retriever: Document retriever instance
        llm: Language model instance
        
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state:RAGState)->RAGState :
          """
          Docstring for retrieve_docs
          
          Retrieve relevant documents node

          Args:
          state: current RAG state

          Returns:
          updated RAG state with retrieved documents
          """
          docs = self.retriever.invoke(state.question)

          return RAGState(question=state.question, retrieved_documents = docs)
    
    def generator(self, state:RAGState)->RAGState:
         
         "generate answer based on given context"

         context = "\n\n".join([doc.page_content for doc in state.retrieved_documents])

         prompt = f"""Answer the given question only using the given context information.
         
                    context : {context}
                    question: {state.question}
                    """
         
         response = self.llm.invoke(prompt)

         return RAGState(
              question= state.question,
              retrieved_documents= state.retrieved_documents,
              answer = response.content
         )