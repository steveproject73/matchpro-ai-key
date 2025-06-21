# app/main.py – versione completa
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from app.helpers import score, gpt_suggestions   # ← già definiti in helpers.py

app = FastAPI(title="MatchPro AI – demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/match")
async def match(cv: UploadFile, job_url: str = Form(...)):
    pdf_bytes = await cv.read()
    sim, cv_txt, job_txt = await score(pdf_bytes, job_url)
    bullets, letter = await gpt_suggestions(cv_txt, job_txt)
    return {
        "score": sim,
        "bullets": bullets,
        "cover_letter": letter
    }
