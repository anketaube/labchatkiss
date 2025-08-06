import streamlit as st
import pandas as pd
import requests

# API-Key und Modell aus Streamlit Secrets laden
API_KEY = st.secrets["ACADEMICCLOUD_API_KEY"]
MODEL = st.secrets.get("ACADEMICCLOUD_MODEL", "meta-llama-3.1-8b-instruct")  # Default-Modell (korrekte Modell-ID!)

API_ENDPOINT = "https://chat-ai.academiccloud.de/v1/chat/completions"

@st.cache_data
def load_data():
    url = "https://github.com/anketaube/LabChatAPP/raw/main/DNBLab_FAQ.xlsx"
    df = pd.read_excel(url)
    return df

faq_df = load_data()

st.title("Lab Chat FAQ mit Academic Cloud API")

user_question = st.text_input("Ihre Frage an das Lab-FAQ:")

def generate_answer_academiccloud(question, faq_df):
    # FAQ-Kontext aus Frage-Antwort-Paaren als Text zusammenstellen
    faq_context = "\n".join(f"{q}: {a}" for q, a in zip(faq_df['Frage'], faq_df['Antwort']))
    prompt = f"""Du bist ein hilfreicher Assistent für Lab-FAQs. Benutze diese FAQ als Wissen:
{faq_context}

Beantworte folgende Nutzerfrage möglichst präzise und textbasiert:

Frage: {question}
Antwort:"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    json_data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Du bist ein hilfreicher textbasierter Assistent."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500,
        # "n" Parameter entfernt (optional, je nach API)
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=json_data)
        response.raise_for_status()
        data = response.json()
        
        # Debug: Falls dir der JSON-Inhalt unklar ist, kannst du ihn hier ausgeben lassen:
        # st.write(data)
        
        # Antwort extrahieren:
        # Die Antwort liegt unter "choices"[0]["message"]["content"]
        answer = data["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        return f"Fehler bei der Antwortgenerierung: {e}"

if user_question:
    with st.spinner("Antwort wird generiert..."):
        answer = generate_answer_academiccloud(user_question, faq_df)
    st.markdown("### Antwort:")
    st.write(answer)
