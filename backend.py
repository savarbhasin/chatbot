__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import io
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

model = ChatGroq(model="llama3-70b-8192")
# model = ChatOpenAI(model='gpt-4o-mini')

prompt = ChatPromptTemplate.from_template("""
    You are an invoice reader AI.
    Your goal is to answer the questions based on the context provided.
    <pdfcontent>{context}</pdfcontent>   
    <question>{input}</question>
    Only answer in json format with the required keys as mention in question and their values. 
    If you think that the context provied is not from an invoice then just return "error":this is not an invoice
""")


embeddings = OpenAIEmbeddings()

vision_model = genai.GenerativeModel('gemini-1.5-flash')


st.title("Invoice Reader AI")

uploaded_file = st.file_uploader("Upload file", type=["pdf", "png", "jpg", "jpeg"])



if uploaded_file is not None:
    
    file_type = uploaded_file.type.split('/')[1]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=75)
    
    if file_type in ["png", "jpg", "jpeg"]:
        image = Image.open(uploaded_file)
        with st.spinner("Processing"):
            response = vision_model.generate_content(["Extract all text from the image", image, "You are an invoice reader."])
            documents = splitter.split_text(response.text)
            vector_store = FAISS.from_texts(documents, embedding=embeddings)
            

    if file_type == 'pdf':
        with st.spinner("Processing"):
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            docs = PyPDFLoader(file_path=temppdf).load()
            documents = splitter.split_documents(docs)
            vector_store = FAISS.from_documents(documents, embedding=embeddings)
            

    retriever = vector_store.as_retriever()    
    
    first_chain = create_stuff_documents_chain(model, prompt, output_parser=JsonOutputParser())
    rag_chain = create_retrieval_chain(retriever, first_chain)

    with st.spinner("Fetching invoice details"):
        answer = rag_chain.invoke({"input": "who is the invoice billed to (customer) give full details? what are the items? what is the total amount of the invoice?"})['answer']
        st.json(answer)

    st.title("Chat with the document")
    question = st.text_input("Ask a question about the invoice:")
    if question:
        with st.spinner("Generating response"):
            answer = rag_chain.invoke({"input": question})['answer']
            st.json(answer)
else:
    st.info("Please upload a PDF or image file to get started.")
        

    
   

