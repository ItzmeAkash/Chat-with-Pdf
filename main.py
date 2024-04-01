import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


#Extracting text from pdf
def get_pdf_text(pdf_docs):
    """
    Extracts text from a  PDF files.

    Args:
        pdf_files :  PDF files.

    Returns:
        str: Concatenated text extracted from all the pages of the PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_readr  =PdfReader(pdf)
        for page in pdf_readr.pages:
            text+=page.extract_text()
    return text

#Divide the text into smaller chunks
def get_text_chunks(text):
    """
    Splits a  text into smaller chunks.

    Args:
        text (str): The text to be split into chunks.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)     
    chunks = text_splitter.split_text(text)   
    return chunks

# Convert the text chunks into Vectors
def get_vector_store(text_chunks):
    """
    Generates a vector store from the text chunks.

    Args:
        text_chunks: A list of text chunks.

    Returns:
        FAISS: A vectorstore database containing embeddings of the text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")


#Conversational question-answering chain for contextual queries
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# User input a question and  generate a response.
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



#Main Function the UI (Streamlit)
def main():
    st.set_page_config("Chat PDF📝")
    st.header("Question and Answer with PDF ⭐")

    user_question = st.text_input("Ask a Question from the PDF")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and  Submit", accept_multiple_files=True)
        if st.button("Submit "):
            with st.spinner("loading..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Successful")



if __name__ == "__main__":
    main()
