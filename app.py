import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv 
from agents.quantitative_agent import QuantitativeAgent

from agents.qualitative_agent import QualitativeAgent # <-- TEMPORALMENTE DESACTIVADO

load_dotenv() 
print("GOOGLE_API_KEY cargada:", os.getenv("GOOGLE_API_KEY") is not None)

st.set_page_config(layout="wide")

st.title("⚽ Director Deportivo IA")

@st.cache_resource
def load_quantitative_agent():
    return QuantitativeAgent(csv_path='data/stats.csv')

@st.cache_resource
def load_qualitative_agent():
     return QualitativeAgent(documents_path='data/articles')

print("Cargando agente cuantitativo...")
stats_agent = load_quantitative_agent()
qualitative_agent = load_qualitative_agent()
print("Agente cuantitativo cargado.")


def route_query(query: str):
    query_lower = query.lower()


    quantitative_keywords = ["gol", "asist", "ranking", "xg", "estadíst", "top", "partido", "prgp"]
    qualitative_keywords = ["noticia", "rumor", "historia", "lesión", "evento", "transferencia"]

    if any(word in query_lower for word in quantitative_keywords):
        return "quantitative"
    elif any(word in query_lower for word in qualitative_keywords):
        return "qualitative"
    else:
        return "both" 


st.subheader("Haz tu consulta:")
query = st.text_input("Escribe tu pregunta")

if st.button("Buscar"):
    if query:
        mode = route_query(query)
        st.info(f"Modo detectado: {mode}")

        if mode in ["quantitative", "both"]:
            with st.spinner("Buscando en estadísticas..."):
                player_data = stats_agent.get_player_stats(query) 
                if "error" in player_data:
                    st.warning(player_data["error"])
                else:
                    st.success("Respuesta cuantitativa:")
                    st.json(player_data)

        if mode in ["qualitative", "both"]:
            with st.spinner("Buscando en artículos..."):
                respuesta = qualitative_agent.answer_question(query)
                st.success("Respuesta cualitativa:")
                st.write(respuesta)