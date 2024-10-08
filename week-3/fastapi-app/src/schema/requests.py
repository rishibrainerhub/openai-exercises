from pydantic import BaseModel


class TextGenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


class SentimentAnalysisRequest(BaseModel):
    text: str
