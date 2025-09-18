from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from agent import agent_run

app = FastAPI(title="Article Summarizer Agent")

# --- CORS (safe for localhost dev) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten later if you deploy
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static /index.html ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    # serve the SPA
    return FileResponse("static/index.html")

# --- API ---
class SearchBody(BaseModel):
    query: str
    top_k: int = 3
    domain_only: str | None = None  # e.g., "ieee.org"

@app.post("/search")
def search(body: SearchBody):
    return agent_run(body.query, top_k=body.top_k, domain_only=body.domain_only)
