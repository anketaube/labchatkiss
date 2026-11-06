import streamlit as st
import pandas as pd
import requests
import re

# API-Key aus Secrets laden
API_KEY = st.secrets["ACADEMICCLOUD_API_KEY"]

# Liste der verfügbaren Modelle (bitte exakt die Modell-IDs aus der Doku verwenden)
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

# 📥 Lade die FAQ-Datei **nur von GitHub**
@st.cache_data
def load_data():
    url = "https://github.com/anketaube/LabChatAPP/raw/main/DNBLab_FAQ.xlsx"
    df = pd.read_excel(url)
    return df

faq_df = load_data()

# 🎯 Sicherstellen, dass die Spalten existieren
if 'Frage' not in faq_df.columns or 'Antwort' not in faq_df.columns:
    st.error("❌ Die Datei auf GitHub hat keine Spalten 'Frage' oder 'Antwort'.")
    st.stop()

# 🧠 Funktion: Prüfe, ob Antwort aus der FAQ stammt
def is_answer_from_faq(answer, faq_df):
    # Normalisiere Antwort: entferne Groß-/Kleinschreibung, Satzzeichen
    clean_answer = re.sub(r'[^\w\s]', '', answer.lower().strip())
    
    # Prüfe, ob die Antwort in den FAQ-Antworten vorkommt
    for faq_answer in faq_df['Antwort'].astype(str):
        clean_faq = re.sub(r'[^\w\s]', '', faq_answer.lower().strip())
        if clean_answer in clean_faq or clean_faq in clean_answer:
            return True
    return False

st.title("DNBLab FAQ Chat mit Academic Cloud API")

# Auswahl-Menü für das Modell
selected_model = st.selectbox("Wähle ein Sprachmodell:", MODEL_OPTIONS, index=0)

user_question = st.text_input("Ihre Frage an das Lab-FAQ:")

def generate_answer_academiccloud(question, faq_df, model):
    faq_context = "\n".join(f"{q}: {a}" for q, a in zip(faq_df['Frage'], faq_df['Antwort']))
    prompt = f"""Du bist ein hilfreicher Assistent für FAQs zum DNBLab. Benutze diese FAQ als Wissen:
{faq_context}

Beantworte folgende Nutzerfrage möglichst präzise und textbasiert:

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
        return answer
    except Exception as e:
        return f"Fehler bei der Antwortgenerierung: {e}"

if user_question:
    with st.spinner("Antwort wird generiert..."):
        answer = generate_answer_academiccloud(user_question, faq_df, selected_model)

    st.markdown("### Antwort:")
    st.write(answer)

    # 📌 Quellenprüfung
    st.markdown("---")
    st.markdown("🔍 **Quelle der Antwort:**")

    if is_answer_from_faq(answer, faq_df):
        st.markdown(
            "[🔗 DNBLab_FAQ.xlsx (GitHub)](https://github.com/anketaube/LabChatAPP/raw/main/DNBLab_FAQ.xlsx)"
        )
        st.markdown("💡 Diese Antwort basiert auf der offiziellen FAQ-Datei des DNBLab.")
    else:
        st.markdown("🤖 LLM (keine passende FAQ gefunden)")
        st.markdown("💡 Die Antwort wurde vom Modell generiert, basierend auf allgemeinem Wissen.")
        st.markdown("🔗 [🔗 Academic Cloud API](https://chat-ai.academiccloud.de)")
