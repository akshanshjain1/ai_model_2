# app/main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.detect import calculate_perplexity

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/detect")
async def detect_ai(input: InputText):
    perplexity = calculate_perplexity(input.text)
    ai_score = max(0, min(100, (100 - perplexity)))  # Simple mapping
    return {"ai_score": round(ai_score, 2), "perplexity": round(perplexity, 2)}

@app.get("/")
async def root():
    return {"status": "ok"}
