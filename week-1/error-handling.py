import time
from typing import Any, Callable, Dict, TypeVar
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

import os

os.environ["OPENAI_API_KEY"] = "your api key"

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


# Type variable for generic function
T = TypeVar("T") 

# Initialize the OpenAI client
client = OpenAI()


def handle_api_error(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle common API errors.
    """

    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
        except openai.APIConnectionError as e:
            logger.error(f"Network error: {e}")
        except openai.APITimeoutError as e:
            logger.error(f"Request timed out: {e}")
        except openai.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
        except openai.PermissionDeniedError as e:
            logger.error(f"Permission denied: {e}")
        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        return None  # type: ignore

    return wrapper


@retry(
    wait=wait_exponential(
        multiplier=1, min=4, max=10
    ),  # Exponential backoff with jitter
    stop=stop_after_attempt(5),  # Maximum of 5 attempts
    retry=retry_if_exception_type(openai.RateLimitError),  # Retry on rate limit errors
)
def make_api_call(messages: list[Dict[str, str]]) -> ChatCompletion:
    """
    Make an API call with exponential backoff for rate limit errors.
    """
    return client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)


@handle_api_error
def safe_api_call(messages: list[Dict[str, str]]) -> ChatCompletion:
    """
    Safely make an API call with error handling.
    """
    return make_api_call(messages)


def make_multiple_api_calls(
    message_list: list[list[Dict[str, str]]], delay: float = 1.0
) -> list[ChatCompletion | None]:
    """
    Make multiple API calls in succession, respecting rate limits.
    """
    results = []
    for messages in message_list:
        result = safe_api_call(messages)
        results.append(result)
        time.sleep(delay)  # Simple rate limiting
    return results


# Example usage
if __name__ == "__main__":
    messages = [
        [{"role": "user", "content": "Hello, how are you?"}],
        [{"role": "user", "content": "What's the weather like today?"}],
        [{"role": "user", "content": "Tell me a joke."}],
    ]

    results = make_multiple_api_calls(messages)
    for i, result in enumerate(results):
        if result:
            logger.info(f"Response {i + 1}: {result.choices[0].message.content}")
        else:
            logger.error(f"Response {i + 1}: Failed to get a response.")
