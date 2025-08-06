import streamlit as st
import pandas as pd
from openai import OpenAI

# OpenAI-Client mit API-Key aus Streamlit Secrets initialisieren
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Excel Datei (als Datenbasis) von Github laden
@st.cache_data
def load_data():
    url = "https://github.com/anketaube/LabChatAPP/raw/main/DNBLab_FAQ.xlsx"
    df = pd.read_excel(url)
    return df

faq_df = load_data()

st.title("Lab Chat FAQ mit GPT-4o")

# Eingabefeld für die Frage
user_question = st.text_input("Ihre Frage an das Lab-FAQ:")

def generate_answer(question, faq_df):
    """Generiert eine textbasierte Antwort mit GPT-4o, basierend auf FAQ-Daten."""
    # FAQ-Kontext als Text zusammenbauen (Frage & Antwort)
    faq_context = "\n".join(f"{q}: {a}" for q, a in zip(faq_df['Frage'], faq_df['Antwort']))
    
    prompt = f"""Du bist ein hilfreicher Assistent für Lab-FAQs. Benutze diese FAQ als Wissen:
{faq_context}

Beantworte folgende Nutzerfrage möglichst präzise und textbasiert:

Frage: {question}
Antwort:"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher textbasierter Assistent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
            n=1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Fehler bei der Antwortgenerierung: {e}"

if user_question:
    with st.spinner("Antwort wird generiert..."):
        answer = generate_answer(user_question, faq_df)
    st.markdown("### Antwort:")
    st.write(answer)

