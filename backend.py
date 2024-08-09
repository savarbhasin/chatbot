import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify   
from flask_cors import cross_origin, CORS

app = Flask(__name__)   
CORS(app)

load_dotenv()


os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = PyPDFLoader(file_path="./invoice.pdf").load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)

documents = splitter.split_documents(docs)

from langchain_groq import ChatGroq
model = ChatGroq(model="llama3-70b-8192")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
    You are an invoice reader AI.
    Your goal is to answer the questions based on the context provided.
    <pdfcontent>{context}</pdfcontent>   
    <question>{input}</question>
    Only answer in json format with the required keys as mention in question and their values.
""")

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(documents,embedding=embeddings)
retriever = vector_store.as_retriever()

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.output_parsers import JsonOutputParser
first_chain = create_stuff_documents_chain(model,prompt,output_parser=JsonOutputParser())
retrieval_chain = create_retrieval_chain(retriever, first_chain)

@app.route('/api/question', methods=['POST'])
@cross_origin()
def question():
    try:
        question = request.json['question']
        answer = retrieval_chain.invoke({"input": question})['answer']
        print(f"Answer: {answer}")
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/api/details',methods=['GET'])
@cross_origin()
def get_details():
    answer = retrieval_chain.invoke({"input":"customer details? products? total amount?"})['answer']
    return jsonify({"answer":answer})
    
if __name__ == "__main__":
    app.run(debug=True)

