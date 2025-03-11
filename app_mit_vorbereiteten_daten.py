import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=128, chunk_overlap=25, length_function=len)
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)


def main():
    load_dotenv()
    json_path = "Ressources/AVB_Datenkorpus.json"  # Passe den Pfad an

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    st.set_page_config(page_title="AVB-Chatbot", page_icon="🐳")
    st.header("AVB-Chatbot 🐳")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    unique_tarife = df[['tarif', 'tarifgeneration']].drop_duplicates()
    selected_tarif = st.selectbox("Wähle einen Tarif und die Tarifgeneration:",
                                  unique_tarife.apply(lambda x: f"{x['tarif']} ({x['tarifgeneration']})", axis=1))

    if st.button("Laden"):
        with st.spinner('Verarbeite das Dokument...'):
            selected_tarif_value, selected_generation_value = selected_tarif.split(" (")
            selected_generation_value = selected_generation_value.rstrip(")")

            relevant_df = df[
                (df['tarif'] == selected_tarif_value) & (df['tarifgeneration'] == selected_generation_value)]
            if relevant_df.empty:
                st.error("Keine passenden Daten gefunden.")
                return

            relevant_texts = "\n".join(relevant_df['chapter_text'].dropna().astype(str))
            text_chunks = get_text_chunks(relevant_texts)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Daten erfolgreich geladen!")

    user_question = st.text_input("Stelle eine Frage:")
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            st.write(f"{'❓' if i % 2 == 0 else '❗'} {message.content}")


if __name__ == '__main__':
    main()