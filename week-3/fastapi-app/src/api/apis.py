from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from schema.requests import TextGenerationRequest, SentimentAnalysisRequest
from schema.responses import APIResponse
from services.openai_service import OpenAIService

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/generate-text")
@limiter.limit("5/minute")
async def generate_text_route(
    request: Request,
    text_request: TextGenerationRequest,
    service: OpenAIService = Depends(OpenAIService),
) -> APIResponse:
    """
    Generate text based on the given prompt using OpenAI's GPT model.
    """
    try:
        result = service.generate_text(text_request.prompt, text_request.max_tokens)
        return APIResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-sentiment")
@limiter.limit("5/minute")
async def analyze_sentiment_route(
    request: Request,
    sentiment_request: SentimentAnalysisRequest,
    service: OpenAIService = Depends(OpenAIService),
) -> APIResponse:
    """
    Analyze the sentiment of the given text using OpenAI's GPT model.
    """
    try:
        result = await service.analyze_sentiment(sentiment_request.text)
        return APIResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
