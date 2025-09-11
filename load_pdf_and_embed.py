from langchain.embeddings import OpenAIEmbeddings # langchain openai tabanlı vektor temsili
from langchain.vectorstores import FAISS #vector db
from langchain.document_loaders import PyPDFLoader #pdf dosyasından metin cıkarma işlemi için gerekli
from langchain.text_splitter import RecursiveCharacterTextSplitter # metni daha küçük parçalar bölme(chunk)
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise  ValueError("OPENAI_API_KEY is not found.")

os.environ["OPENAI_API_KEY"] = api_key

# SSS dosyasını yükle
loader = PyPDFLoader("musteri_destek_faq.pdf")

documents = loader.load() #langchain documents objesi olustur

# metinleri parcalamak için
#splitter metni anlamlı parçalara ayırırken cümle veua paragraf bütünlüğünü korumaya çalışıyor
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Her bir metin parçasının maksimum uzunluğu (karakter sayısı)
    chunk_overlap=50  # Parçalar arasında tekrarlanan karakter sayısı (örtüşme miktarı)
)

# chunkları olustur
docs = text_splitter.split_documents(documents)

#openai embedding modeli
#text-embedding-3-large modeli kaliteli ve tr destegi cok iyi
embedding = OpenAIEmbeddings(model= "text-embedding-3-large")

# faiss vektör veri tabanı parcalara ayrılmış metni embedding ile vektör haline getirir ve index oluşturur
vectordb = FAISS.from_documents(docs,embedding)

# olusturulan vektor veritabanını yerel diske kaydet
vectordb.save_local("faq_vectorstore")

print("embedding ve vektor veritabanı basarılı bir sekilde olusturuldu.")