# app/helpers.py  – versione completa e pulita (giugno 2025)

import io, re, os, json, ast, requests, bs4, pdfplumber, openai
from newspaper import Article
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# ─── 0. CONFIG OPENAI ─────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── 1. CV: estrai testo dal PDF ─────────────────────────────────
def cv_text(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return re.sub(r"\s+", " ", " ".join(pages))[:8000]

# ─── 2. JOB: estrai testo dalla job-description (URL) ────────────
def job_text(url: str) -> str:
    # A) newspaper3k
    try:
        art = Article(url, language="it")
        art.download(); art.parse()
        txt = f"{art.title} {art.text}".strip()
        if len(txt) > 300:
            return txt[:8000]
    except Exception:
        pass
    # B) fallback: requests + BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        html = requests.get(url, headers=headers, timeout=10).text
        soup = bs4.BeautifulSoup(html, "html.parser")
        p_text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        return p_text[:8000]
    except Exception:
        return ""

# ─── 3. EMBEDDING & SCORE 0-100 ─────────────────────────────────
async def embed(text: str) -> list[float]:
    res = await openai.Embedding.acreate(input=text, model="text-embedding-3-small")
    return res["data"][0]["embedding"]

async def score(pdf_bytes: bytes, url: str):
    cv_txt  = cv_text(pdf_bytes)
    job_txt = job_text(url)
    e_cv, e_job = await embed(cv_txt), await embed(job_txt or " ")
    sim = cosine_similarity([e_cv], [e_job])[0][0] if job_txt else 0
    return round(sim * 100, 1), cv_txt, job_txt

# ─── 4. PROMPT GPT-4o mini ───────────────────────────────────────
BULLET_PROMPT = (
    "You are a career assistant. Improve EXACTLY five bullet-points of this CV "
    "so that it matches the job description. Keep each bullet under 15 words "
    "and start with a verb. Reply with a JSON list of 5 strings.\n\n"
    "CV:\n###\n{cv}\n###\n\nJOB DESCRIPTION:\n###\n{job}\n###"
)

LETTER_PROMPT = (
    "Write a 150-word cover letter in Italian that matches this CV to the job "
    "description. Tone: professional but enthusiastic. Mention 2 hard skills "
    "and 1 soft skill that appear both in CV and job description."
)

# ─── 5. BULLET-POINT + COVER LETTER ──────────────────────────────
async def gpt_suggestions(cv_text: str, job_text: str):
    # ▸ bullet-points
    bullets_resp = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": BULLET_PROMPT.format(cv=cv_text, job=job_text)
        }],
        max_tokens=300
    )
    raw = bullets_resp.choices[0].message.content.strip()

    # ▸ pulizia preliminare ('json', '```json', ecc.)
    clean = raw.lower().lstrip('`').replace('```json', '').replace('json', '').strip()
    clean = clean[clean.find('['):] if '[' in clean else clean

    # ▸ parsing robusto
    bullets: list[str] | None = None
    for parser in (json.loads, ast.literal_eval):
        try:
            bullets = parser(clean)
            if isinstance(bullets, list):
                break
        except Exception:
            pass
    if not isinstance(bullets, list):
        bullets = re.findall(r'"([^"]{5,120}?)"', clean) or [clean]

    # ▸ normalizzazione finale
    bullets = [b.lstrip('\n-•* ').rstrip() for b in bullets][:5]

    # ▸ cover letter
    letter_resp = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": LETTER_PROMPT.format(cv=cv_text, job=job_text)
        }],
        max_tokens=250
    )
    letter = letter_resp.choices[0].message.content

    return bullets, letter



