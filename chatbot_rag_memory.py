"""
problem tanimi: akıllı müşteri destek sistemi
    -müsteriler sık sık benzer sorular sorar.

    -çözüm
        - .pdf dosyasını (sıkça sorulan sorular) vektör veritabanına dönüştürme(faiss)
        - kullanıcıdan gelen sorular veri tabanında sorgulanır ve gpt türkçe cevaplar üretir



Kullanılan teknolojiler
    -langchain: rag mimarisi kurmak için
    - faiss : embdedingleri saklmamak için hızlı bir vektör veri tabanı
    - openai: soru cevap için llm
    - streamlit: web arayüzü , son kullanıcı ile interaktif kullanıcı deneyimi


veriseti
    - Soru : yurtdısı satırslarınız bulunuyor mu
    - Cevap : hayır.

plan
    - SSS bilgilerini içeren pdf
    - kullanıcı dosyayı arayüzden yükleyecek
    - pdf metni parçalara ayrılacak ve embeddingler çıkarılacak
    - kullanıcı soru sorduğu zaman vektör db den benzer içerikler getirilir. gpt ile cevap oluşturulur.
    - konuşma geçmişi memory ile saklanır ve sonraki yanıtlara bağlam oluşturulur.


install libraries: freeze

"""
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain # rag+ sohbet zinciri
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory # konusma gecmisini tutan hafiza yapisi
from dotenv import load_dotenv
import os

load_dotenv() #ortam degiskenlerini .env den ükle

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPEN_AI_API key is not found.")

os.environ["OPENAI_API_KEY"] =api_key

#embdeding modeli baslat (text -> vektor)
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

#daha once olusturulmus vektor veri tabanını yükle
vectordb = FAISS.load_local(
    "faq_vectorstore", #kaydedilmis faiss veri tabanı klasörü
    embedding, #embedding modeli
    allow_dangerous_deserialization=True #pickle güvenlik uyarısı bastırma
)

#konusma gecmisi icin memory olusturma
memory = ConversationBufferMemory(
    memory_key="chat_history", #konusma gecmisi bu anahtarla saklanır
    return_messages=True # gecmis mesajlar tam haliyle geri döner
)

# sıfır rsatlantısalık ile çalışır,sabit cevaplar verir.
llm = ChatOpenAI(
    model_name = "gpt-4",
    temperature = 0.8 #deterministik(aynı girdiye aynı çıktıyı verir)
)


"""
rag + memory zincir oluştur
- llm
- faiss retriever : en benzer 3 belge getirilsin (k=3)
- memory
"""
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever = vectordb.as_retriever(search_kwargs = {"k": 3}),
    memory= memory,
    verbose = True
)

print("Müşteri destek botuna hoşgeldiniz")
while True:
    user_input = input("Siz: ")
    if user_input.lower() == "çık":
        break

    #kullanıcı sorusu llm +rag + memory zincirine verilir
    response = qa_chain.run(user_input)
    print("Müşteri destek botu: ",response)