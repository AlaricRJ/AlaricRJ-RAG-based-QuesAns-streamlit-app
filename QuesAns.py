import os                                                                    #used for File, Directory, path, environment variable, process management Operations
import streamlit as st                                                       #for frontend
from langchain_groq import ChatGroq                                          #for chatbot 
from langchain.text_splitter import RecursiveCharacterTextSplitter           #chunking for RAG
from langchain.chains.combine_documents import create_stuff_documents_chain  #for prioritizing the chunk based on relevancy
from langchain.chains import create_retrieval_chain                          #for creating own prompts using templates
from langchain.chains import create_retrieval_chain                          #for retrieval from RAG
from langchain_community.vectorstores import FAISS                           #FAISS is a Vector DB by facebook  for semantic and similarity search
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader        #for reading PDFs as input
from langchain_google_genai import GoogleGenerativeAIEmbeddings              #for converting text to embeddings (vectors)

from dotenv import load_dotenv                                               #to load environment variables

load_dotenv()                                                                #this fuction call looks for a .env file in the current directory & Load environment variables from .env file

#loading environment variables 
groq_api = os.getenv("GROQ_API_KEY")
google_api = os.getenv("GOOGLE_API_KEY")


st.title("Document Question Answering ")                                     #creats title heading in frontend

llm = ChatGroq(groq_api_key= groq_api, model_name="Gemma-7b-it")             #creating an object of ChatGroq class by name llm





#creating a prompt with help of existing prompt templates
prompt = ChatPromptTemplate.from_template(                                  
"""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>

Question: {input}

"""
)


#function for reading doc, splitting in chunks, creating embeddings of chuncks, storing in vector DB
def vector_embedding():

    if "vectors" not in st.session_state:                                                                               #st.session_state is used in Streamlit to store variables between reruns of the app,consider it as a dictionary 
                                                                                                                        #"vectors" is a string and here working as a key, and it checks if this key is already present in the st.session_state dictionary    
        st.session_state.loader = PyPDFDirectoryLoader("./PDFs")                                                        #Creates an instance of PyPDFDirectoryLoader, intialised with a path
        st.session_state.docs = st.session_state.loader.load()                                                          #Uses the loader object, to load the pdfs

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)           #Creates an instance of RecursiveCharacterTextSplitter
        st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.docs)              #Uses the text_splitter object to split the loaded document into smaller chunks

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")                        #Creates an instance of GoogleGenerativeAIEmbeddings with the specified model 
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_doc, st.session_state.embeddings )       #Uses the FAISS library to create a vector index from the document chunks using the generated embeddings




prompt1 = st.text_input("Ask your question based on document?")                                                         #makes placeholder in frontend, and saves the query in variable                                                        


if st.button("Creating Vector Store"):                                                                                  #if the buttons is clicked
    vector_embedding()
    st.write("Vector Store DB is ready")



import time


if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)                                                         #Create a chain for passing a list of Documents to a model
    retriever = st.session_state.vectors.as_retriever()                                                                #creates retriever object from dv "vector"
    retrieval_chain = create_retrieval_chain(retriever, document_chain)                                                #Create retrieval chain that retrieves documents and then passes them on

    start = time.process_time()

    response = retrieval_chain.invoke({'input': prompt1})                                                              #Generate responses
    
    st.write(response['answer'])




#following will be part of explanation which is generated by Gemma model




















