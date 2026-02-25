"""
FastAPI wrapper for hybrid.py

Run:
    uvicorn hybrid_api:app --reload --host 0.0.0.0 --port 8000
"""

import json
from pathlib import Path
from typing import Generator
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from hybrid import (
    answer_pdf,
    answer_weather,
    get_current_weather,
    pdf_writer_prompt,
    retrieve_pdf_chunks,
    route,
    weather_writer_prompt,
    writer_llm,
)


app = FastAPI(title="Hybrid PDF+Weather API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class ChatResponse(BaseModel):
    route: Literal["pdf", "weather"]
    answer: str
    metadata: Dict[str, Any]


class RouteResponse(BaseModel):
    route: Literal["pdf", "weather"]
    city: Optional[str] = None
    units: Optional[Literal["imperial", "metric"]] = None


def _normalize_route(decision: Dict[str, Any]) -> Literal["pdf", "weather"]:
    selected = decision.get("route", "pdf")
    if selected not in {"pdf", "weather"}:
        return "pdf"
    return selected


def _stream_llm_content(messages: Any) -> Generator[str, None, None]:
    for chunk in writer_llm.stream(messages):
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            if content:
                yield content
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        yield text


@app.get("/", response_class=FileResponse)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def decide_route(payload: ChatRequest) -> RouteResponse:
    decision = route(payload.question)
    selected = _normalize_route(decision)

    city = decision.get("city")
    units = decision.get("units")
    if units not in {"imperial", "metric"}:
        units = "imperial"

    return RouteResponse(route=selected, city=city, units=units)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    decision = route(question)
    selected = _normalize_route(decision)

    if selected == "weather":
        city = (decision.get("city") or "").strip()
        units = decision.get("units", "imperial")
        if units not in {"imperial", "metric"}:
            units = "imperial"

        if not city:
            raise HTTPException(
                status_code=400,
                detail="Weather route selected but no city found. Include a city in your question.",
            )

        raw_weather = get_current_weather.invoke({"city": city, "units": units})
        answer = answer_weather(question=question, city=city, units=units)

        return ChatResponse(
            route="weather",
            answer=answer,
            metadata={
                "city": city,
                "units": units,
                "raw_weather": raw_weather,
                "source": "OpenWeather API",
            },
        )

    retrieval = retrieve_pdf_chunks.invoke({"question": question})
    answer = answer_pdf(question)

    if retrieval.get("error"):
        return ChatResponse(
            route="pdf",
            answer=retrieval["error"],
            metadata={"source": "PDF (RAG)", "error": retrieval["error"]},
        )

    return ChatResponse(
        route="pdf",
        answer=answer,
        metadata={
            "source": "PDF (RAG)",
            "sources": retrieval.get("sources", []),
        },
    )


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    decision = route(question)
    selected = _normalize_route(decision)

    if selected == "weather":
        city = (decision.get("city") or "").strip()
        units = decision.get("units", "imperial")
        if units not in {"imperial", "metric"}:
            units = "imperial"

        if not city:
            raise HTTPException(
                status_code=400,
                detail="Weather route selected but no city found. Include a city in your question.",
            )

        raw_weather = get_current_weather.invoke({"city": city, "units": units})
        messages = weather_writer_prompt.format_messages(
            question=question, weather_json=json.dumps(raw_weather)
        )
        return StreamingResponse(
            _stream_llm_content(messages), media_type="text/plain; charset=utf-8"
        )

    retrieval = retrieve_pdf_chunks.invoke({"question": question})
    if retrieval.get("error"):
        return StreamingResponse(
            iter([retrieval["error"]]), media_type="text/plain; charset=utf-8"
        )

    messages = pdf_writer_prompt.format_messages(
        question=question,
        context=retrieval["context"],
        sources_json=json.dumps(retrieval["sources"]),
    )
    return StreamingResponse(
        _stream_llm_content(messages), media_type="text/plain; charset=utf-8"
    )
