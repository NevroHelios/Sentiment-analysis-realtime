from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import get_gentiment

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/sentiment")
def sentiment_analysis(request: SentimentRequest):
    return get_gentiment(request.text)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}
