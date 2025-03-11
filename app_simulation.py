import multiprocessing
import os
import random
import re
import time

import pandas as pd
import torch
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Lade Umgebungsvariablen aus .env-Datei
load_dotenv()

# SBERT-Modell laden
sbert_model = SentenceTransformer('all-mpnet-base-v2')

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=128, chunk_overlap=25, length_function=len)
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

def load_questions(question_file):
    df = pd.read_csv(question_file)
    return df['Frage'].tolist()

def get_relevance_score(question, passage, answer):
    llm = ChatOpenAI()
    prompt = (f"Frage: {question}\nTextpassage: {passage}\nAntwort: {answer}\n\nWie gut passt die Antwort zur Frage "
              f"im Kontext der Textpassage? Bewerte die Passung auf einer Skala von 0 (schlecht) bis 1 (gut) "
              f"und gib nur den numerischen Wert zurück.")
    response = llm.invoke(prompt)

    match = re.search(r'\d+\.?\d*', response.content)
    if match:
        score_str = match.group(0)
        try:
            score = float(score_str)
            return score
        except ValueError:
            print(f"Konnte numerischen Wert nicht extrahieren: {score_str}")
            return 0.0
    else:
        print(f"Kein numerischer Wert in der Antwort gefunden: {response.content}")
        return 0.0

def get_coherence_score(passage, answer):
    llm = ChatOpenAI()
    prompt = (f"Textpassage: {passage}\nAntwort: {answer}\n\nWie kohärent ist die Antwort im Kontext der Textpassage? "
              f"Bewerte die Kohärenz auf einer Skala von 0 (schlecht) bis 1 (gut) "
              f"und gib nur den numerischen Wert zurück.")
    response = llm.invoke(prompt)

    match = re.search(r'\d+\.?\d*', response.content)
    if match:
        score_str = match.group(0)
        try:
            score = float(score_str)
            return score
        except ValueError:
            print(f"Konnte numerischen Wert nicht extrahieren: {score_str}")
            return 0.0
    else:
        print(f"Kein numerischer Wert in der Antwort gefunden: {response.content}")
        return 0.0

def get_sbert_score(text1, text2):
    embeddings1 = sbert_model.encode(text1, convert_to_tensor=True)
    embeddings2 = sbert_model.encode(text2, convert_to_tensor=True)
    cosine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=0).item()
    return cosine_similarity

def process_document(pdf_file, pdf_folder, questions, question_file):
    try:
        raw_text = get_pdf_text(os.path.join(pdf_folder, pdf_file))
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)

        chat_history = []
        random_questions = random.choices(questions, k=10) # Anzahl der zu stellenden Fragen

        results = []

        for question in random_questions:
            response = conversation_chain.invoke({'question': question, 'chat_history': chat_history})
            answer = response.get('answer', '')
            passage = response.get('text', '')

            relevance_score = get_relevance_score(question, passage, answer)
            coherence_score = get_coherence_score(passage, answer)
            sbert_qk_score = get_sbert_score(question, passage)
            sbert_qv_score = get_sbert_score(question, answer)

            results.append({
                'question': question,
                'answer': answer,
                'passage': passage,
                'relevance_score': relevance_score,
                'coherence_score': coherence_score,
                'sbert_qk_score': sbert_qk_score,
                'sbert_qv_score': sbert_qv_score
            })

        return {
            "document": pdf_file,
            "results": results
        }

    except Exception as e:
        print(f"Error processing document {pdf_file}: {e}")
        return None

def simulate_for_all_documents(pdf_folder, question_file):
    print(f"Starte Simulation für Dateien in: {pdf_folder}")
    tarif_dict = extract_tarif_info(pdf_folder)
    questions = load_questions(question_file)

    if not tarif_dict:
        print("Keine Dateien zum Verarbeiten gefunden. Simulation beendet.")
        return

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as pool:
        try:
            results = pool.starmap(process_document, [(pdf_file, pdf_folder, questions,
                                                       question_file) for pdf_file in tarif_dict.keys()])
        except KeyboardInterrupt:
            print("Prozess unterbrochen")
            pool.terminate()
            pool.join()

    all_results = [result for result in results if result is not None]

    # Ergebnisse in eine DataFrame umwandeln
    data = []
    for result in all_results:
        document = result['document']
        for row in result['results']:
            data.append({
                'document': document,
                'question': row['question'],
                'answer': row['answer'],
                'passage': row['passage'],
                'relevance_score': row['relevance_score'],
                'coherence_score': row['coherence_score'],
                'sbert_qk_score': row['sbert_qk_score'],
                'sbert_qv_score': row['sbert_qv_score']
            })
    results_df = pd.DataFrame(data)

    results_df.to_csv('simulation_results' + str(time.time()) + '.csv', index=False)
    print("Simulation abgeschlossen! Ergebnisse gespeichert.")

def extract_tarif_info(pdf_folder):
    pattern = re.compile(r"BED_([A-Z]+)_([0-9]{2}_[0-9]{4})")
    tarif_dict = {}

    if not os.path.exists(pdf_folder):
        print(f"Fehler: Ordner '{pdf_folder}' existiert nicht.")
        return tarif_dict

    for file in os.listdir(pdf_folder):
        match = pattern.search(file)
        if match:
            tarif, generation = match.groups()
            tarif_dict[file] = (tarif, generation)

    if not tarif_dict:
        print(f"Warnung: Keine passenden Dateien im Ordner '{pdf_folder}' gefunden.")
    return tarif_dict

def main():
    pdf_folder = "C:/Users/nemah/OneDrive/Dokumente/Masterstudium/Vergangene Semester/BDBA WS2425/Analyse semi- & unstrukturierter Daten/Projekt/Daten/AVB"  # fester PDF-Ordner
    question_file = "C:/Users/nemah/OneDrive/Dokumente/Masterstudium/Vergangene Semester/BDBA WS2425/Analyse semi- & unstrukturierter Daten/Projekt/AVB_Fragenkatalog.csv" # Katalog vordefinierter Fragen

    simulate_for_all_documents(pdf_folder, question_file)

if __name__ == '__main__':
    main()