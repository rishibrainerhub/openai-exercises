import os
from typing import List, Tuple

import openai
from openai.types.chat import ChatCompletion

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = (
    "your-api-key-here"
)


def get_model_response(model: str, prompt: str) -> str:
    """
    Get a response from the specified model for a given prompt.

    Args:
        model (str): The name of the model to use.
        prompt (str): The user's input prompt.

    Returns:
        str: The model's response.
    """
    response: ChatCompletion = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a customer service assistant that categorizes customer inquiries.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def test_model(model: str, prompts: List[str]) -> List[Tuple[str, str]]:
    """
    Test a model with a list of prompts.

    Args:
        model (str): The name of the model to test.
        prompts (List[str]): A list of test prompts.

    Returns:
        List[Tuple[str, str]]: A list of (prompt, response) tuples.
    """
    results = []
    for prompt in prompts:
        response = get_model_response(model, prompt)
        results.append((prompt, response))
    return results


def print_test_results(model: str, results: List[Tuple[str, str]]):
    """
    Print the test results for a model.

    Args:
        model (str): The name of the model tested.
        results (List[Tuple[str, str]]): A list of (prompt, response) tuples.
    """
    print(f"\nTesting model: {model}")
    for prompt, response in results:
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")


def main():
    # Test prompts
    test_prompts = [
        "How do I change my email address?",
        "What's the status of my refund?",
        "Do you offer gift wrapping?",
    ]

    # Models to test (replace with your fine-tuned model name)
    models = ["gpt-3.5-turbo", "ft:gpt-3.5-turbo-0125:dudoxx-ug::AD9BgNKE"]

    # Test both models and print results
    for model in models:
        results = test_model(model, test_prompts)
        print_test_results(model, results)


if __name__ == "__main__":
    main()
