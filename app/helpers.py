"""helpers.py – utility functions for MatchPro AI (OpenAI >= 1.26)

Modifiche 22 giu 2025
• aggiunto fallback di scraping con requests + User‑Agent “browser”
• gestito 403/405 sollevando JobDownloadError (custom) per poter mostrare un messaggio chiaro lato API
"""
from __future__ import annotations

import os
import json
import ast
import io
import asyncio
from typing import List, Tuple

import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from newspaper import Article, ArticleException
from openai import AsyncOpenAI

__all__ = [
    "embed",
    "job_text",
    "score",
    "gpt_suggestions",
]

# ---------------------- costanti & client ---------------------- #
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    )
}

client = AsyncOpenAI()  # legge OPENAI_API_KEY dalle ENV vars
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ---------------------- prompt GPT ----------------------------- #
BULLET_PROMPT = (
    "Agisci come career assistant. Migliora ESATTAMENTE cinque bullet point di questo CV "
    "(max 15 parole ciascuno) in base alla descrizione della posizione. "
    "Rispondi SOLO con un JSON list di 5 stringhe.\nCV:\n###\n{cv}\n###\n"
    "JOB DESCRIPTION:\n###\n{job}\n###\n"
)

LETTER_PROMPT = (
    "Scrivi una cover letter in italiano che colleghi questo CV alla job description. "
    "Tono: professionale ma entusiasta. Cita 2 hard‑skills presenti sia nel CV che nell'annuncio."
)

# ------------------------ errori custom ------------------------ #
class JobDownloadError(RuntimeError):
    """Raised when the job post cannot be scraped (403/405 ecc.)."""

# ---------------------- funzioni core -------------------------- #
async def embed(text: str) -> List[float]:
    """Ottiene embedding del testo (async)."""
    resp = await client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


async def cv_text(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)


async def job_text(url: str) -> str:
    """Scrape annuncio. Se il sito blocca (403/405) tenta fallback requests."""
    loop = asyncio.get_running_loop()

    def _download() -> str:
        art = Article(url, language="en")
        try:
            art.download()
            art.parse()
            return art.text
        except ArticleException:
            # fallback con headers browser
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code >= 400:
                raise JobDownloadError(f"{resp.status_code} on {url}")
            art = Article(url, language="en")
            art.download(input_html=resp.text)
            art.parse()
            return art.text

    return await loop.run_in_executor(None, _download)


async def gpt_suggestions(cv_txt: str, job_txt: str) -> Tuple[List[str], str]:
    # bullets
    bullets_raw = await client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": BULLET_PROMPT.format(cv=cv_txt, job=job_txt)}],
        max_tokens=300,
    )
    try:
        bullets = json.loads(bullets_raw.choices[0].message.content)
        if not isinstance(bullets, list):
            raise ValueError
        bullets = [b.strip("\n ") for b in bullets]
    except Exception:
        bullets = [ln.strip() for ln in bullets_raw.choices[0].message.content.split("\n") if ln]

    # cover letter
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
    cv_task = asyncio.create_task(cv_text(pdf_bytes))
    job_task = asyncio.create_task(job_text(job_url))

    cv_txt, job_txt = await asyncio.gather(cv_task, job_task)

    e_cv = await embed(cv_txt)
    e_job = await embed(job_txt)
    sim = float(cosine_similarity([e_cv], [e_job])[0][0] * 100)
    return sim, cv_txt, job_txt





