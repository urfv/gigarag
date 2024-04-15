import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.chat_models import GigaChat
from langchain.schema import ChatMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain.schema import ChatMessage
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
GIGACHAT_API_KEY = os.environ.get('GIGACHAT_API_KEY')

#ui
st.title("Gigachat RAG Machine")

# инициалиация истории чата
if "messages" not in st.session_state:
    st.session_state.messages = [
        ChatMessage(
            role="system",
            content="You're a smart RAG bot, always ready to help a user find necessary information",
        ),
        ChatMessage(role="assistant", content="Ask away!"),
    ]

# отображение сообщений чата из истории при повторном запуске приложения
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)

#текст сплиттер и эмбеддинги
loader = TextLoader("text.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(documents)
embeddings_model = GigaChatEmbeddings(credentials=GIGACHAT_API_KEY, verify_ssl_certs=False)
db = FAISS.from_documents(documents, embeddings_model)

#логика чата
if prompt := st.chat_input():
    chat = GigaChat(
        credentials=GIGACHAT_API_KEY,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
        stream=True,
        profanity=True,
        timeout=600,
        verbose=False
    )

    message = ChatMessage(role="user", content=prompt)
    st.session_state.messages.append(message)

    with st.chat_message(message.role):
        st.markdown(message.content)

    qa_chain = RetrievalQA.from_chain_type(llm=chat, retriever=db.as_retriever())
    response = qa_chain.invoke({"query": prompt})
    otvet = response['result']

    message = ChatMessage(role="assistant", content=otvet)
    st.session_state.messages.append(message)

    with st.chat_message(message.role):
        message_placeholder = st.empty()
        message_placeholder.markdown(message.content)

    st.session_state.token = chat._client.token
    chat._client.close()