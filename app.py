
import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

st.set_page_config(page_title='הבוט התרופתי - DEBUG', layout='wide')

st.markdown('## 🧪 גרסת בדיקה - טעינת קבצים')

# Load API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 מפתח API של OpenAI לא הוגדר.")
    st.stop()

# Display which files exist in docs/
folder_path = "docs"
if not os.path.exists(folder_path):
    st.warning("📁 התיקייה docs לא קיימת כלל!")
    st.stop()

pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
if not pdf_files:
    st.warning("⚠️ לא נמצאו קבצי PDF בתיקייה docs.")
    st.stop()

st.success(f"✅ נמצאו {len(pdf_files)} קבצים בתיקייה docs:")
for f in pdf_files:
    st.markdown(f"- {f}")

# Try to load documents and build vectorstore
try:
    documents = []
    for file_name in pdf_files:
        path = os.path.join(folder_path, file_name)
        st.markdown(f"🔄 טוען את הקובץ: `{file_name}`")
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)

    st.success(f"📄 נטענו {len(documents)} עמודים בסך הכול.")

    # Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    st.info(f"🧩 מחולקים ל־{len(texts)} מקטעים")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(texts, embeddings)
    st.success("✅ ה־Vectorstore נבנה בהצלחה!")
except Exception as e:
    st.error(f"❌ שגיאה בטעינה או עיבוד הקבצים: {e}")
