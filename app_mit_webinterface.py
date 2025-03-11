import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import re


def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator = "\n", chunk_size = 128, chunk_overlap = 25,
                                          length_function = len)
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts = text_chunks, embedding = embeddings)


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    return ConversationalRetrievalChain.from_llm(llm = llm, retriever = vectorstore.as_retriever(), memory = memory)


def extract_tarif_info(pdf_folder):
    pattern = re.compile(r"BED_([A-Z]+)_([0-9]{2}_[0-9]{4})")
    tarif_dict = {}

    for file in os.listdir(pdf_folder):
        match = pattern.search(file)
        if match:
            tarif, generation = match.groups()
            tarif_dict[file] = (tarif, generation)

    return tarif_dict


def main():
    load_dotenv()
    pdf_folder = "C:/Users/nemah/OneDrive/Dokumente/Masterstudium/Vergangene Semester/BDBA WS2425/Analyse semi- & unstrukturierter Daten/Projekt/Daten/alle"  # Ändere den Pfad zu deinem festen PDF-Ordner
    tarif_dict = extract_tarif_info(pdf_folder)

    st.set_page_config(page_title = "AVB-Chatbot", page_icon = "🐳")
    st.header("AVB-Chatbot 🐳")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if tarif_dict:
        options = {v: k for k, v in tarif_dict.items()}  # Umkehren für Dropdown-Auswahl
        selected_tarif = st.selectbox("Wähle einen Tarif und die Tarifgeneration:", list(options.keys()))
        selected_pdf = options[selected_tarif]

        if st.button("Laden"):
            with st.spinner('Verarbeite das Dokument...'):
                raw_text = get_pdf_text(os.path.join(pdf_folder, selected_pdf))
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Dokument erfolgreich geladen!")

    user_question = st.text_input("Stelle eine Frage:")
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            st.write(f"{'❓' if i % 2 == 0 else '❗'} {message.content}")


if __name__ == '__main__':
    main()