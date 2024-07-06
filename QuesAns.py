import os                                                                    #used for File, Directory, path, environment variable, process management Operations
import streamlit as st                                                       #for frontend
from langchain_groq import ChatGroq                                          #for chatbot 
from langchain.text_splitter import RecursiveCharacterTextSplitter           #chunking for RAG
from langchain.chains.combine_documents import create_stuff_documents_chain  #for prioritizing the chunk based on relevancy
from langchain.chains import create_retrieval_chain                          #for creating own prompts using templates
from langchain.chains import create_retrieval_chain                          #for retrieval from RAG
from langchain_community.vectorstores import FAISS                           #FAISS is a Vector DB by facebook  for semantic and similarity search
from langchain_community.document_loaders import PyPDFDirectoryLoader        #for reading PDFs as input
from langchain_google_genai import GoogleGenerativeAIEmbeddings              #for converting text to embeddings (vectors)

from dotenv import load_dotenv                                               #to load environment variables

load_dotenv()










