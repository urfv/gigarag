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

# Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         ChatMessage(
#             role="system",
#             content="Ты - product manager и специалист по customer development, который всегда готов помочь пользователю своей экспертизой.",
#         ),
#         ChatMessage(role="assistant", content="Как я могу помочь вам?"),
#     ]

#llm
load_dotenv()
GIGACHAT_API_KEY = os.environ.get('GIGACHAT_API_KEY')
llm = GigaChat(
    credentials=GIGACHAT_API_KEY,
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False,
    model="GigaChat-Pro-preview",
    stream=True,
    profanity=True,
    timeout=600,
    verbose=True
)


#текст сплиттер
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
# embedding_size = 1024
# index = faiss.IndexFlatL2(embedding_size)
# vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})
# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# memory = VectorStoreRetrieverMemory(retriever=retriever)
db = FAISS.from_documents(documents, embeddings_model)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

if st.button('Go'):
    if input_field:
        with st.spinner('Generating response...'):
            prompt = input_field
            response = qa_chain({"query": prompt})
            otvet = response['result']
            # output = response.return_values['output']
            # start_index = output.find("Final Answer:") + len("Final Answer:")
            # end_index = output.find("\"]", start_index)
            # final_answer = output[start_index:end_index].strip()
            # print(f"Final Answer: {final_answer}")
            # st.write(f"Final Answer: {final_answer}")
            st.write(otvet)
    else:
        st.warning('Please enter your prompt')





    

    # message = ChatMessage(role="user", content=prompt)
    # st.session_state.messages.append(message)

    # with st.chat_message(message.role):
    #     st.markdown(message.content)

    # message = ChatMessage(role="assistant", content="")
    # st.session_state.messages.append(message)

    # with st.chat_message(message.role):
    #     message_placeholder = st.empty()
    #     for chunk in chat.stream(st.session_state.messages):
    #         message.content += chunk.content
    #         message_placeholder.markdown(message.content + "▌")
    #     message_placeholder.markdown(message.content)

    # # Каждый раз, когда пользователь нажимает что-то в интерфейсе весь скрипт выполняется заново.
    # # Сохраняем токен и закрываем соединения
    # st.session_state.token = chat._client.token
    # chat._client.close()