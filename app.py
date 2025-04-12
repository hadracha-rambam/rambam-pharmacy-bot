
import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title='הבוט התרופתי של רמב"ם', layout='wide')

st.markdown('''
<style>
body {
    direction: rtl;
    text-align: right;
    font-family: 'Assistant', sans-serif;
}
</style>
''', unsafe_allow_html=True)

st.markdown('## 🧠 הבוט החכם של רמב"ם')
st.markdown('מחפש מיהולים של תרופות או פרוטוקולים של בית המרקחת? הגעת למקום הנכון... שאל את הבוט שלנו.')

# Load OpenAI API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 יש להגדיר מפתח API של OpenAI על מנת להשתמש בבוט.")
    st.stop()

@st.cache_resource
def load_vectorstore():
    folder_path = "docs"
    if not os.path.exists(folder_path):
        return None

    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file_name))
            documents.extend(loader.load())

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

if vectorstore:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    user_question = st.text_input("✍️ מה ברצונך לברר?", placeholder="לדוגמה: איך למהול Vancomycin בילדים?")
    if user_question:
        with st.spinner("חושב..."):
            result = qa_chain.run(user_question)
            st.markdown("#### 💬 תשובת הבוט:")
            st.write(result)
else:
    st.warning("⚠️ לא נטענו קבצים או שהמידע אינו זמין.")
