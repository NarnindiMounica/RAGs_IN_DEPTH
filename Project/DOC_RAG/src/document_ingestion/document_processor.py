"""Document processing module for loading and splitting documents"""

from typing import List, Union
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pathlib import Path

class DocumentProcessor:
    "Handle document loading and processing"

    def __init__(self, chunk_size:int=500, chunk_overlap:int=50):
        """
        Initialize document processor

        Args:
        chunk_size : size of text chunks
        chunk_overlap: overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )


    def load_from_url(self, url:str)->List[Document]:
        "load the documents from urls"
        loader = WebBaseLoader(web_path=url) 
        return loader.load()
    
    def load_from_pdf_dir(self, directory:Union[str, Path])->List[Document]:
        "load documents from all pdf's inside directory"
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()
    
    def load_from_text(self, file_path:Union[str, Path])->list[Document]:
        "load documents from given text doc file path"
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()
    
    def load_from_pdf(self, file_path:Union[str, Path])->list[Document]:
        "load documents for give pdf file"
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    

    def load_documents(self, sources:List[str])->List[Document]:
       """
       Docstring for load_documents
       
       Load documents from URLs, PDF Directories or TXT Files

       Args:

       sources: List of URLs, PDF folder paths, or Txt file paths

       Returns:
       List of loaded documents
       """ 
       docs = List[Document]=[]
       for src in sources:
           if src.startswith("https://") or src.startswith("http://"):
               docs.extend(self.load_from_url(src))

           path = Path("data")
           if path.is_dir(): #PDF Directory
               docs.extend(self.load_from_pdf_dir(path))
           elif path.suffix.lower()==".txt" :
               docs.extend(self.load_from_txt(path))
           else:
               raise ValueError(f"Unsupported source type: {src}.")

           return docs   


       def split_documents(self, documents:List[Document])->List[Document]:
           """
           Docstring for split_documents
           
           Splits documents into chunks

           Args:
           documents: list of documents to split

           Returns:
           List of split documents
           """  
           return self.splitter.split_documents(documents)
       
       def process_url(self, urls:List[str])->List[Document]:
           """
           Docstring for process_url
           
           Complete pipeline to load and split documents

           Args:
           urls: List of URLs to process

           Returns:
           List of processed document chunks

           """
           docs = self.load_document(urls)
           return self.split_documents(docs)
