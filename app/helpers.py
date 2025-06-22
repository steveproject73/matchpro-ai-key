"""helpers.py – utility functions for MatchPro AI (OpenAI >= 1.26)

⚠️ Dipendenze:
    - openai>=1.26.0,<2.0
    - tiktoken
    - numpy, scikit‑learn, pdfplumber, newspaper3k, lxml==4.9.3

Il modulo espone:
    • embed(text)                → embedding vettoriale (async)
    • job_text(url)              → testo pulito dell'annuncio (async)
    • score(pdf_bytes, job_url)  → similarity CV/annuncio + testi in chiaro (async)
    • gpt_suggestions(cv_txt, job_txt) → lista bullet + cover letter (async)

Tutte le funzioni sono asincrone: vanno attese con «await». 
"""
from __future__ import annotations

import os
import json
import ast
import io
import asyncio
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from newspaper import Article
from openai import AsyncOpenAI

__all__ = [
    "embed",
    "job_text",
    "score",
    "gpt_suggestions",
]

# ---------------------- OpenAI client ---------------------- #
client = AsyncOpenAI(  # legge OPENAI_API_KEY dalle env vars
    # api_key=os.getenv("OPENAI_API_KEY"),  # opzionale se usi nome diverso
    # organization=os.getenv("OPENAI_ORG"),
)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ------------------ prompts per ChatCompletion ------------------ #
BULLET_PROMPT = (
    "Agisci come career assistant. Migliora ESATTAMENTE cinque bullet point di questo CV "
    "in modo che combacino con la job description. Ogni bullet <15 parole. "
    "Rispondi solo con un JSON list di 5 stringhe.\nCV:\n###\n{cv}\n###\n"  # noqa: E501
    "JOB DESCRIPTION:\n###\n{job}\n###\n"
)

LETTER_PROMPT = (
    "Scrivi una cover letter in italiano che colleghi questo CV alla job description. "
    "Tono: professionale ma entusiasta. Cita 2 hard‑skills presenti sia nel CV che nella job."
)

# ------------------------ utilities ------------------------ #
async def embed(text: str) -> List[float]:
    """Ottieni l'embedding di **text** usando OpenAI (async)."""
    resp = await client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding


async def cv_text(pdf_bytes: bytes) -> str:
    """Estrae il testo da un PDF in bytes."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)


async def job_text(url: str) -> str:
    """Scarica e pulisce l'annuncio di lavoro con newspaper3k (async wrapper)."""
    loop = asyncio.get_running_loop()

    def _download() -> str:
        art = Article(url, language="en")  # newspaper usa sync I/O
        art.download()
        art.parse()
        return art.text

    return await loop.run_in_executor(None, _download)


async def gpt_suggestions(cv_txt: str, job_txt: str) -> Tuple[List[str], str]:
    """Ritorna bullet points migliorati + cover letter. """
    # ---------- bullets ---------- #
    bullets_raw = await client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": BULLET_PROMPT.format(cv=cv_txt, job=job_txt)}],
        max_tokens=300,
    )

    # response_format=json_object garantisce JSON valido
    try:
        bullets = json.loads(bullets_raw.choices[0].message.content)
        if isinstance(bullets, list):
            # normalizza spazi
            bullets = [b.strip("\n ") for b in bullets]
        else:
            raise ValueError
    except Exception:
        # fallback alla stringa grezza separata da '\n'
        bullets = [ln.strip() for ln in bullets_raw.choices[0].message.content.split("\n") if ln]

    # ---------- cover letter ---------- #
    letter_raw = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": LETTER_PROMPT},
            {"role": "assistant", "content": "CV:" + cv_txt[:4000]},
            {"role": "assistant", "content": "JOB:" + job_txt[:4000]},
        ],
        max_tokens=350,
    )

    letter = letter_raw.choices[0].message.content.strip()
    return bullets, letter


async def score(pdf_bytes: bytes, job_url: str) -> Tuple[float, str, str]:
    """Calcola similarity tra CV e annuncio, e restituisce testi puliti (cv_txt, job_txt)."""
    cv_task = asyncio.create_task(cv_text(pdf_bytes))
    job_task = asyncio.create_task(job_text(job_url))

    cv_txt, job_txt = await asyncio.gather(cv_task, job_task)

    e_cv = await embed(cv_txt)
    e_job = await embed(job_txt)

    sim = float(cosine_similarity([e_cv], [e_job])[0][0] * 100)
    return sim, cv_txt, job_txt




