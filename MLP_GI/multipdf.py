import streamlit as st
from PyPDF2 import PdfReader #to read those pdf 
from langchain.text_splitter import RecursiveCharacterTextSplitter #to convert into vectors
import os
#chroma db is mainly for vector embedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

#now importing some useful lib
from langchain.vectorstores import FAISS #vector embeding 
from langchain_google_genai import ChatGoogleGenerativeAI #
from langchain.chains.question_answering import load_qa_chain #this helps to do chat 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#CREATING THE FUNCTION FOR UPLOADING THE FILES

def get_pdf_text(pdf_docs): 
    text=""
    for pdf in pdf_docs: #read all the pages in the pdf
        pdf_reader=PdfReader(pdf) #with the help of PDfreader we are reading it will readit
        for page in pdf_reader.pages:
            text+=page.extract_text()# we are extracting the info from pages
    return text

#converting into chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

#getting the vectors using google embedding
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") #this models/embedding is atechnique
    vector_store =FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")


# #
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide the 
    context just say, "answer is not available in the context", don't provide the 
    context: \n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain =get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat with multiple PDF")
    st.header("Chat with multiple pdf using Gemini ")

    user_question = st.text_input("Ask a question from the pdf files ")

    if user_question:
        user_input(user_question)

    
        

    with st.sidebar:
        st.title("ALL FILES:")
        pdf_docs = st.file_uploader("upload the pdfs", accept_multiple_files=True)
        if st.button("submit and process"):
            with st.spinner("processing...."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()



