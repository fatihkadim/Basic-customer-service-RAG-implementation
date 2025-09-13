import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import tempfile

# Streamlit başlık ve ikon
st.set_page_config(page_title="Müşteri Destek Botu", page_icon="🤖")
st.title("📑 Müşteri Destek Botu (RAG + Memory)")
st.write("Bir PDF yükleyin ve içerik hakkında sorular sorun. Türkçe desteklidir.")

# .env yükle
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# PDF yükleme
uploaded_file = st.file_uploader("PDF Dosyanızı yükleyin", type="pdf", key="pdf_uploader")

if uploaded_file is not None:
    if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
        with st.spinner("📂 PDF işleniyor..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            embedding = OpenAIEmbeddings(model="text-embedding-3-large")
            vectordb = FAISS.from_documents(docs, embedding)

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            llm = ChatOpenAI(model="gpt-4", temperature=0)

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
                memory=memory
            )

            st.session_state.qa_chain = qa_chain
            st.session_state.messages = []  # chatbot mesajları için
            st.session_state.last_uploaded_name = uploaded_file.name

        st.success("✅ PDF başarıyla işlendi")

# Chatbot ekranı
if "qa_chain" in st.session_state:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Sorunuzu yazın..."):
        # Kullanıcı mesajı göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Model cevabı al
        response = st.session_state.qa_chain.invoke(prompt)
        answer = response["answer"]

        # Model cevabını göster
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
