from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import get_sentiment, load_pipe

app = FastAPI()
pipe = None

def reload_pipe():
    global pipe
    pipe = load_pipe()

class SentimentRequest(BaseModel):
    text: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    reload_pipe()

@app.get("/reload_model")
def reloading_pipe():
    reload_pipe()

@app.post("/predict")
def sentiment_analysis(request: SentimentRequest):
    assert pipe is not None, "Pipeline is loading..."
    return get_sentiment(request.text, pipe)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}
