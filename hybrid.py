"""
HYBRID PDF-RAG + WEATHER CHATBOT (Router LLM + Direct Tool Calls)
-----------------------------------------------------------------
What "hybrid" means here:
1) A small LLM "router" decides: weather vs pdf, and extracts location if needed.
2) Python executes the chosen tool DIRECTLY (no AgentExecutor, no agent loops).
3) (Optional) A final LLM formats the tool output into a nice response.

Pros:
- No agent iteration loops / iteration-limit errors
- Still uses @tool for clean interfaces
- You keep full control + easy debugging

Env vars required:
- OPENAI_API_KEY
- OPENWEATHER_API_KEY

Install:
pip install -U langchain langchain-openai langchain-community langchain-text-splitters \
               langchain-chroma chromadb pypdf requests
"""

import os
import json
import re
import requests
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent

def _load_environment() -> None:
    env_file = os.getenv("ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)
        return
    load_dotenv(BASE_DIR / ".env", override=False)
    load_dotenv(override=False)


_load_environment()


# -----------------------
# 0) ENV CHECKS
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

print("OPENWEATHER key loaded?", bool(OPENWEATHER_API_KEY))
print("First 6 chars:", (OPENWEATHER_API_KEY or "")[:6])

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment.")
if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found in environment.")


# -----------------------
# 1) CONFIG
# -----------------------
PDF_PATH = Path(os.getenv("PDF_PATH", BASE_DIR / "docs" / "transformers.pdf"))
PERSIST_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "pdf_rag"


# -----------------------
# 2) BUILD / LOAD CHROMA VECTOR STORE
# -----------------------
def build_or_load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("📦 Loading existing Chroma vector store...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found at: {PDF_PATH}. Place your file there or set PDF_PATH in .env."
        )

    print("📄 Building Chroma vector store from PDF (first run)...")
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    print("✅ Vector store ready.")
    return vectordb


vectordb = build_or_load_vectorstore()
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)


# -----------------------
# 3) TOOLS (capabilities)
# -----------------------
def _normalize_city(city: str) -> str:
    city = (city or "").strip()
    city = re.sub(r"[?.!,;:]+$", "", city)
    city = re.sub(r"\s+", " ", city)
    # helpful default for OpenWeather ambiguity
    if city.lower() in {"new york", "new york city"}:
        return "New York,US"
    return city


@tool
def get_current_weather(city: str, units: str = "imperial") -> Dict[str, Any]:
    """
    Return raw weather JSON fields (not a friendly paragraph).
    Hybrid pattern: tool returns structured data; LLM formats it later.
    Uses OpenWeather free-tier endpoint /data/2.5/weather.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "Missing OPENWEATHER_API_KEY."}

    city = _normalize_city(city)
    if not city:
        return {"error": "No city provided. Try 'Seattle' or 'New York,US'."}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": units}

    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return {
                "error": f"Weather request failed: {r.status_code}",
                "details": r.text,
            }

        data = r.json()
        main = data.get("main") or {}
        wind = data.get("wind") or {}
        weather = (data.get("weather") or [{}])[0]

        return {
            "location": f"{data.get('name','')}, {(data.get('sys') or {}).get('country','')}".strip().strip(
                ","
            ),
            "conditions": weather.get("description"),
            "temp": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "wind_speed": wind.get("speed"),
            "units": units,
        }
    except Exception as e:
        return {"error": "Weather tool exception", "details": str(e)}


@tool
def retrieve_pdf_chunks(question: str) -> Dict[str, Any]:
    """
    Retrieve top-k relevant PDF chunks for a question.
    Hybrid pattern: tool returns context + sources; LLM writes final answer.
    """
    question = (question or "").strip()
    if not question:
        return {"error": "Empty question."}

    docs = retriever.get_relevant_documents(question)
    if not docs:
        return {"error": "No relevant PDF content found."}

    context = "\n\n---\n\n".join(
        [f"[Page {d.metadata.get('page', 'NA')}]\n{d.page_content}" for d in docs]
    )

    sources = []
    for d in docs:
        page = d.metadata.get("page", "?")
        snippet = (d.page_content[:160] + "...").replace("\n", " ")
        sources.append({"page": page, "snippet": snippet})

    return {"question": question, "context": context, "sources": sources}


# -----------------------
# 4) ROUTER (LLM decides which tool to call)
# -----------------------
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=120)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route user messages to one of two tools:\n"
            "1) weather (current weather)\n"
            "2) pdf (questions about the PDF)\n\n"
            "Return STRICT JSON ONLY (no markdown, no extra text) with one of:\n"
            '{{"route":"weather","city":"<city or empty>","units":"imperial|metric"}}\n'
            '{{"route":"pdf"}}\n\n'
            "Rules:\n"
            "- If user asks about weather/temperature/forecast/rain/snow/wind/humidity -> route=weather.\n"
            '- Extract the city if present; if missing, city="".\n'
            "- Use imperial unless the user asks for Celsius/metric.\n",
        ),
        ("human", "{question}"),
    ]
)


def route(question: str) -> Dict[str, Any]:
    raw = router_llm.invoke(
        router_prompt.format_messages(question=question)
    ).content.strip()
    try:
        obj = json.loads(raw)
        if obj.get("route") not in {"weather", "pdf"}:
            return {"route": "pdf"}
        return obj
    except json.JSONDecodeError:
        # Safe fallback
        return {"route": "pdf"}


# -----------------------
# 5) ANSWER WRITER (LLM formats tool outputs)
# -----------------------
writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=350)

weather_writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use ONLY the provided weather JSON to answer. "
            "If it contains an error, ask for a valid city. Keep it 2-5 sentences.",
        ),
        ("human", "User question: {question}\nWeather JSON: {weather_json}"),
    ]
)

pdf_writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using ONLY the provided PDF context. "
            "If the answer isn't in the context, say: \"I don't know based on the PDF.\" "
            "Write 1-3 short paragraphs. End with a short 'Sources' list using the provided sources.",
        ),
        (
            "human",
            "Question: {question}\n\nPDF Context:\n{context}\n\nSources JSON: {sources_json}",
        ),
    ]
)


def answer_weather(question: str, city: str, units: str) -> str:
    data = get_current_weather.invoke({"city": city, "units": units})
    return writer_llm.invoke(
        weather_writer_prompt.format_messages(
            question=question, weather_json=json.dumps(data)
        )
    ).content.strip()


def answer_pdf(question: str) -> str:
    payload = retrieve_pdf_chunks.invoke({"question": question})
    if payload.get("error"):
        return payload["error"]

    return writer_llm.invoke(
        pdf_writer_prompt.format_messages(
            question=question,
            context=payload["context"],
            sources_json=json.dumps(payload["sources"]),
        )
    ).content.strip()


# -----------------------
# 6) MAIN LOOP
# -----------------------
def main():
    print("\n✅ HYBRID PDF + Weather Chatbot Ready. Type 'exit' to quit.\n")
    print("Try:")
    print("  - What does the PDF say about self-attention?")
    print("  - What's the weather in New York?")
    print("  - Weather in Seattle in Celsius")
    print()

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        decision = route(question)

        if decision.get("route") == "weather":
            city = decision.get("city", "")
            units = decision.get("units", "imperial")
            if not city.strip():
                print("\nBot: Which city should I check the weather for?\n")
                continue

            out = answer_weather(question, city=city, units=units)
            print("\nBot:", out)
            print("\nSources: Weather API\n")
            continue

        # Default: PDF
        out = answer_pdf(question)
        print("\nBot:", out)
        print("\nSources: PDF (RAG)\n")


if __name__ == "__main__":
    main()
