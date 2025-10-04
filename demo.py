import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Function to process PDF and return vectordb + QA chain
def process_pdf(uploaded_file):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)

    llm = ChatOpenAI(model="gpt-3.5-turbo")  # you can change to gpt-4 if you have access
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    return vectordb, qa


st.title("📘 AI Research Co-Pilot (Demo)")

# Upload two PDFs
pdf1 = st.file_uploader("Upload first PDF", type="pdf")
pdf2 = st.file_uploader("Upload second PDF", type="pdf")

if pdf1 and pdf2:
    st.success("PDFs uploaded! Building knowledge bases...")

    vectordb1, qa1 = process_pdf(pdf1)
    vectordb2, qa2 = process_pdf(pdf2)

    st.success("Knowledge bases ready! You can now ask questions.")

    query = st.text_input("Ask a question about the PDFs")

    if query:
        st.write("🔎 Searching first PDF...")
        result1 = qa1.run(query)
        st.write("**Answer from PDF 1:**", result1)

        st.write("🔎 Searching second PDF...")
        result2 = qa2.run(query)
        st.write("**Answer from PDF 2:**", result2)

        # Compare answers
        st.subheader("📊 Comparison of both PDFs")
        st.write("PDF 1 says:", result1)
        st.write("PDF 2 says:", result2)
