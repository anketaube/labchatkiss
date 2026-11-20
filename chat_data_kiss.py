import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from io import BytesIO
import re

# 📌 API-Key aus Secrets laden
API_KEY = st.secrets["ACADEMICCLOUD_API_KEY"]

# 📌 Liste der verfügbaren Modelle
MODEL_OPTIONS = [
    "meta-llama-3.1-8b-instruct",
    "internvl2.5-8b",
    "deepseek-r1",
    "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-instruct",
    "llama-3.1-sauerkrautlm-instruct",
    "llama-3.1-nemotron-70b",
    "mistral-large-instruct",
    "codestral-22b",
    "e5-mistral-7b-instruct",
    "qwen-2.5-72b-instruct",
    "qwen-2.5-vl-72b-instruct",
    "qwen-2.5-coder-32b-instruct"
]

API_ENDPOINT = "https://chat-ai.academiccloud.de/v1/chat/completions"

# 📥 Lade die FAQ-Datei von GitHub
@st.cache_data
def load_faq_data():
    url = "https://github.com/anketaube/LabChatAPP/raw/main/DNBLab_FAQ.xlsx"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content))
    return df

# 🧠 Lade oder erstelle den Vektorindex
@st.cache_data
def load_or_create_index():
    # Versuche, den Index von GitHub zu laden
    index_url = "https://github.com/anketaube/DNBLabChat/raw/main/dnblab_index.json"
    try:
        response = requests.get(index_url)
        response.raise_for_status()
        data = response.json()
        st.info("✅ Vektorindex erfolgreich geladen.")
        return data
    except:
        st.warning("⚠️ Kein Index gefunden. Erstelle ihn jetzt...")
        # Erstelle den Index neu
        df = load_faq_data()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        documents = []
        for idx, row in df.iterrows():
            text = row['Antwort']
            embedding = model.encode(text).tolist()
            documents.append({
                "id": str(idx),
                "text": text,
                "metadata": {
                    "question": row['Frage'],
                    "answer": row['Antwort']
                },
                "embedding": embedding
            })
        index_data = {"documents": documents}
        # Speichere den Index (kann in GitHub gespeichert werden)
        st.info("✅ Vektorindex wurde erfolgreich erstellt.")
        return index_data

# 🧠 Funktion: Suche relevante Abschnitte
def retrieve_relevant_context(query, documents, top_k=2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).astype(np.float32)

    similarities = []
    for doc in documents:
        doc_embedding = np.array(doc["embedding"], dtype=np.float32)
        sim = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        similarities.append((sim, doc))

    similarities.sort(reverse=True)
    top_docs = similarities[:top_k]

    context = "\n".join([doc["text"] for sim, doc in top_docs])
    sources = [doc["metadata"] for sim, doc in top_docs]
    return context, sources

# 🧠 Funktion: Generiere Antwort mit RAG
def generate_answer_rag(question, model, top_k=2):
    index_data = load_or_create_index()
    documents = index_data["documents"]
    context, sources = retrieve_relevant_context(question, documents, top_k=top_k)

    prompt = f"""Du bist ein hilfreicher Assistent für das DNBLab. Beantworte die Frage ausschließlich basierend auf den folgenden Informationen:

{context}

Frage: {question}
Antwort:"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    json_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Du bist ein hilfreicher textbasierter Assistent."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=json_data)
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer, sources
    except Exception as e:
        return f"Fehler bei der Antwortgenerierung: {e}", []

# 🎯 Haupt-App
st.title("🔍 DNBLab FAQ Chat mit RAG (Retrieval-Augmented Generation)")

# Auswahl-Menü für das Modell
selected_model = st.selectbox("Wähle ein Sprachmodell:", MODEL_OPTIONS, index=0)

user_question = st.text_input("Ihre Frage an das Lab-FAQ:")

if user_question:
    with st.spinner("🔍 Suche nach relevanten Informationen..."):
        answer, sources = generate_answer_rag(user_question, selected_model)

    st.markdown("### 💬 Antwort:")
    st.write(answer)

    # 📌 Quellenprüfung
    st.markdown("---")
    st.markdown("🔍 **Quelle der Antwort:**")

    if sources:
        st.markdown("✅ Diese Antwort basiert auf der offiziellen FAQ:")
        for src in sources:
            st.markdown(f"🔹 **Frage:** {src['question']}")
            st.markdown(f"   **Antwort:** {src['answer']}")
        st.markdown(
            f"[🔗 DNBLab_FAQ.xlsx (GitHub)](https://github.com/anketaube/LabChatAPP/raw/main/DNBLab_FAQ.xlsx)"
        )
    else:
        st.markdown("🤖 LLM (keine passende FAQ gefunden)")
        st.markdown("💡 Die Antwort wurde vom Modell generiert, basierend auf allgemeinem Wissen.")
        st.markdown("🔗 [🔗 Academic Cloud API](https://chat-ai.academiccloud.de)")
