import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from exceptions.openai import handle_api_error

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)




class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @handle_api_error
    def generate_text(self, prompt: str, max_tokens: int) -> str:
        """
        Generate text based on the given prompt using OpenAI's GPT model.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=max_tokens,
        )
        generated_text = response.choices[0].message.content.strip()
        logger.info(f"Generated text for prompt: {prompt[:50]}...")
        return generated_text

    @handle_api_error
    async def analyze_sentiment(self, text: str) -> str:
        """
        Analyze the sentiment of the given text using OpenAI's GPT model.
        """

        prompt = f"Analyze the sentiment of the following text and respond with either 'positive', 'negative', or 'neutral': \"{text}\""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1,
            temperature=0.1,
        )
        sentiment = response.choices[0].message.content.strip().lower()
        logger.info(f"Analyzed sentiment for text: {text[:50]}...")
        return sentiment
