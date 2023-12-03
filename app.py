import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    OPENAI_API_KEY=st.secrets[OPENAI_API_KEY]
    
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # uploading file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # reading the contents ( extracts the data page wise )
    if pdf is not None:
        st.write(pdf)
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split input data into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,# character
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        # create embeddings -> encode of info for effective search
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # create input field for asking questions
        user_question = st.text_input("Ask a question about your PDF: ")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()  # large language model - openai in our case
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.write(response)


if __name__ == "__main__":
    main()