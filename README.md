# Basic Customer Service RAG Implementation

A simple Retrieval-Augmented Generation (RAG) system with **Turkish language support** for customer service.  
You can upload an FAQ PDF and ask questions to get accurate answers.  

<img width="800" height="600" alt="Ekran görüntüsü 2025-09-13 215528" src="https://github.com/user-attachments/assets/f530f025-48e6-45be-aec4-0a0e115d68f2" />

## Features
- 📄 Upload your own FAQ PDF (`musteri_destek_faq.pdf` as example included)
- 🤖 Ask questions in Turkish, get contextual answers
- 💾 Vectorstore-based document retrieval
- 🧠 Conversational memory support
- 🌐 Simple Streamlit interface

## File Structure
- `faq_vectorstore/` → Stores vectorized FAQ data  
- `chatbot_rag_memory.py` → RAG system with conversational memory  
- `load_pdf_and_embed.py` → Loads and embeds PDF into vectorstore  
- `streamlit_app.py` → Streamlit frontend for interaction  
- `musteri_destek_faq.pdf` → Example FAQ file  
- `requirements.txt` → Dependencies  

## Installation
```
git clone https://github.com/your-username/Basic-customer-service-RAG-implementation.git

cd Basic-customer-service-RAG-implementation

pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```

streamlit run streamlit_app.py

```

Then, upload your FAQ PDF and start asking questions in Turkish. 

## Requirements
- Python 3.9+

- Streamlit

- LangChain

- FAISS

- OpenAI API key (for embeddings & LLM)

## License
This project is licensed under the MIT License.
