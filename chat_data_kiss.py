import streamlit as st
import pandas as pd
import requests

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

@st.cache_data
def load_data():
    url = "https://github.com/anketaube/LabChatAPP/raw/main/DNBLab_FAQ.xlsx"
    df = pd.read_excel(url)
    return df

faq_df = load_data()

st.title("Lab Chat FAQ mit Academic Cloud API")

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
