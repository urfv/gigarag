import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.chat_models import GigaChat
from langchain.schema import ChatMessage
from langchain.chains import RetrievalQA

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.gigachat import GigaChatEmbeddings

#frontend
st.title("GigaChain RAG")
input_field = st.text_input('Пиши сюда')

#llm
load_dotenv()
GIGACHAT_API_KEY = os.environ.get('GIGACHAT_API_KEY')
llm = GigaChat(
    credentials=GIGACHAT_API_KEY,
    scope="GIGACHAT_API_PERS",
    temperature=0,
    verify_ssl_certs=False,
    model="GigaChat-Pro-preview",
    stream=True,
    profanity=True,
    timeout=600,
    verbose=True
)

#текст сплиттер и эмбеддинги
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
loader = TextLoader("анна.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(documents)
embeddings_model = GigaChatEmbeddings(credentials=GIGACHAT_API_KEY, verify_ssl_certs=False)
db = FAISS.from_documents(documents, embeddings_model)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

if st.button('Go'):
    if input_field:
        with st.spinner('Generating response...'):
            prompt = input_field
            response = qa_chain({"query": prompt})
            otvet = response['result']
            st.write(otvet)
    else:
        st.warning('Please enter your prompt')
