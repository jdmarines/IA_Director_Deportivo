# main_app.py

import streamlit as st
from dotenv import load_dotenv

# Importamos las clases de nuestros agentes
from agents.quantitative_agent import QuantitativeAgent
from agents.qualitative_agent import QualitativeAgent

# Importamos lo necesario de LangChain y Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Cargar la variable de entorno con la API Key de Google
load_dotenv()

# --- CONFIGURACIÓN DE LA APLICACIÓN DE STREAMLIT ---
st.set_page_config(
    page_title="ScoutAI - Asistente de Scouting",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ ScoutAI: Tu Asistente de Scouting")
st.caption("Una herramienta para análisis cuantitativo y cualitativo de jugadores.")

# --- INICIALIZACIÓN DE LOS AGENTES (USANDO CACHÉ DE STREAMLIT) ---
# st.cache_resource permite que los agentes se carguen una sola vez y no en cada recarga.
# Esto es crucial para no reconstruir la base de datos vectorial cada vez.
@st.cache_resource
def initialize_agents():
    """
    Carga y prepara ambos agentes para ser utilizados.
    """
    print("Inicializando agentes por primera vez...")
    # Asegúrate de que la ruta al CSV es correcta
    quantitative_agent = QuantitativeAgent(csv_path="data/stats.csv") 
    qualitative_agent = QualitativeAgent()
    print("Agentes listos.")
    return quantitative_agent, qualitative_agent

quantitative_agent, qualitative_agent = initialize_agents()

# --- LÓGICA DEL ENRUTADOR (ROUTER) CON GEMINI ---
@st.cache_resource
def initialize_router_chain():
    """
    Crea la cadena de LangChain que decidirá qué agente usar.
    """
    print("Inicializando cadena de enrutamiento...")
    
    # Usamos Gemini Flash por su velocidad y bajo costo
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.0)

    # El prompt es la "inteligencia" del enrutador. Le enseñamos a clasificar la pregunta.
    router_template = """
    Tu tarea es clasificar la pregunta de un usuario para un sistema de scouting de fútbol.
    Debes decidir si la pregunta es 'cuantitativa' o 'cualitativa'.

    - 'cuantitativa': Se refiere a estadísticas, números, métricas, rankings, datos específicos.
      Ejemplos: "¿Quién es el máximo goleador?", "¿Cuántos años tiene Messi?", "Top 5 asistidores".
    - 'cualitativa': Se refiere a opiniones, descripciones de habilidades, estilo de juego, potencial, comparación de perfiles.
      Ejemplos: "¿Cómo juega Lamine Yamal?", "Fortalezas y debilidades de Mbappé", "¿Qué jugador es un buen regateador?".

    Basado en la pregunta, responde únicamente con la palabra 'cuantitativa' o 'cualitativa'.

    Pregunta del usuario:
    {user_question}

    Clasificación:
    """

    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["user_question"]
    )

    return LLMChain(llm=llm, prompt=router_prompt)

router_chain = initialize_router_chain()


# --- INTERFAZ DE USUARIO ---
user_question = st.text_input(
    "Haz tu pregunta sobre un jugador:",
    placeholder="Ej: ¿Cuáles son las estadísticas de Haaland? o ¿Cómo es el estilo de juego de Bellingham?"
)

if user_question:
    with st.spinner("Analizando pregunta y buscando información..."):
        # 1. Usamos el enrutador para clasificar la pregunta
        st.info("Clasificando pregunta...")
        route = router_chain.run(user_question).strip().lower()
        st.write(f"**Tipo de consulta detectada:** `{route}`")

        # 2. Ejecutamos el agente correspondiente
        if "cuantitativa" in route:
            # Para preguntas cuantitativas, intentamos extraer la métrica o el jugador
            # Esta es una lógica simple, se podría mejorar con otro llamado al LLM
            if "top" in user_question.lower() or "máximo" in user_question.lower():
                # Asumimos una métrica común como Goles (Gls) si no se especifica
                metric_to_find = "Gls" # Podrías hacer esto más inteligente
                response = quantitative_agent.find_top_players(metric=metric_to_find)
            else:
                # Extraemos el nombre del jugador (lógica simple)
                # Se asume que el nombre viene después de "de" o es la última parte
                parts = user_question.split("de ")
                player_name = parts[-1].replace("?", "").strip()
                response = quantitative_agent.get_player_stats(player_name=player_name)
            
            st.json(response) # Mostramos el JSON/dict de forma bonita

        elif "cualitativa" in route:
            response = qualitative_agent.answer_question(query=user_question)
            st.markdown(response)
        
        else:
            # Fallback si el enrutador no responde lo esperado
            st.error("No se pudo determinar el tipo de pregunta. Intenta reformularla.")
