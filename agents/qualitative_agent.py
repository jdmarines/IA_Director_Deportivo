import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
# Fallback opcional si no hay clave de Google (descomenta si lo usarás)
# from langchain_community.llms import Ollama

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


class QualitativeAgent:
    def __init__(
        self,
        documents_path: str = "data/articles",
        chunk_size: int = 1000,
        chunk_overlap: int = 120,
        top_k: int = 6,  # recupera pocos chunks para no exceder el contexto
        model_name_embed: str = "sentence-transformers/all-MiniLM-L6-v2",
        gemini_model: str = "gemini-1.5-flash",  # evita -latest
        temperature: float = 0.2,
    ):
        print("Inicializando Agente Cualitativo con FAISS...")

        # --- 1) Carga y split de documentos
        loader = DirectoryLoader(documents_path, glob="**/*.txt", show_progress=True)
        self.documents = loader.load()
        print(f"Se cargaron {len(self.documents)} documentos.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # ayuda a cortar limpio
        )
        self.texts = text_splitter.split_documents(self.documents)
        print(f"Se dividieron los documentos en {len(self.texts)} fragmentos.")

        # --- 2) Embeddings y vector store
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name_embed)
        print("Modelo de embeddings cargado.")

        self.vector_store = FAISS.from_documents(self.texts, self.embeddings)
        print("Base de datos vectorial FAISS creada en memoria.")

        # --- 3) LLM (Gemini) + safety + timeout
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if google_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                api_key=google_api_key,
                temperature=temperature,
                # max_output_tokens opcional; algunos bindings lo soportan:
                # max_output_tokens=1024,
               # safety_settings={
                #    "HARASSMENT": "BLOCK_ONLY_HIGH",
                 #   "HATE_SPEECH": "BLOCK_ONLY_HIGH",
                  #  "SEXUAL": "BLOCK_ONLY_HIGH",
                   # "DANGEROUS": "BLOCK_ONLY_HIGH",
                #},
                # client_options={"timeout": 120.0},  # si lo soporta tu versión
            )
            print(f"LLM: {gemini_model} (Google Generative AI)")
        else:
            # Fallback local (si quieres trabajar sin API key). Requiere Ollama corriendo.
            # self.llm = Ollama(model="llama3", temperature=temperature)
            # print("LLM: llama3 via Ollama (fallback)")
            raise RuntimeError(
                "Falta GOOGLE_API_KEY. Exporta la variable o habilita el fallback de Ollama."
            )

        # --- 4) Prompt en español, pensado para map_reduce
        prompt_template = """
Eres un analista experto en fútbol (Premier League).
Responde SIEMPRE en español, con precisión y concisión.
Usa SOLO la información del contexto. Si no es suficiente, dilo explícitamente.

Contexto:
{context}

Pregunta:
{question}

Respuesta (en español):
"""
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        # --- 5) QA chain en modo map_reduce (mejor para textos largos que stuff)
        self.chain = load_qa_chain(
            self.llm,
            chain_type="map_reduce",  # <- clave para no reventar el contexto
            prompt=PROMPT,            # se usa en la fase "map"; LangChain gestiona el reduce
            verbose=False,
        )

        # --- 6) Retriever (k corto para no inflar el prompt)
        self.retriever_k = top_k

        print("Agente Cualitativo listo.")

    def _retrieve(self, query: str):
        # Usa similaridad con un k bajo para controlar tamaño de prompt
        return self.vector_store.similarity_search(query, k=self.retriever_k)

    def answer_question(self, query: str) -> str:
        if not self.documents:
            return "No hay documentos cargados para realizar la búsqueda."

        print(f"Buscando documentos relevantes para: '{query}'")
        relevant_docs = self._retrieve(query)
        print(f"Se encontraron {len(relevant_docs)} fragmentos relevantes (k={self.retriever_k}).")

        try:
            # load_qa_chain con map_reduce acepta run(...) con estos kwargs:
            response = self.chain.run(input_documents=relevant_docs, question=query)
            return response.strip()
        except Exception as e:
            # Log amigable para diagnosticar problemas de cuota, safety, timeout o input grande
            return (
                "Ocurrió un error consultando el LLM.\n"
                f"Tipo: {type(e).__name__}\n"
                f"Detalle: {str(e)}\n"
                "Sugerencias: verifica GOOGLE_API_KEY, modelo ('gemini-1.5-flash'), "
                "reduce top_k, y evita documentos demasiado grandes."
            )
 