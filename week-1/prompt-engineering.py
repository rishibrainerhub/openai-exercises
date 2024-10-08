from typing import List, Dict, Any, Tuple
from openai import OpenAI
import time
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = (
    "your-api-key-here"
)

# Initialize the OpenAI client
client = OpenAI()


def generate_few_shot_prompt(
    task: str, examples: List[Dict[str, str]], query: str
) -> str:
    """
    Generate a few-shot prompt for a specific task.

    Args:
    task (str): Description of the task.
    examples (List[Dict[str, str]]): List of example input-output pairs.
    query (str): The input for which we want to generate an output.

    Returns:
    str: The generated few-shot prompt.
    """
    prompt = f"{task}\n\nExamples:\n"
    for example in examples:
        prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
    prompt += f"Input: {query}\nOutput:"
    return prompt


def generate_chain_of_thought_prompt(task: str, steps: List[str], query: str) -> str:
    """
    Generate a chain-of-thought prompt for a multi-step reasoning task.

    Args:
    task (str): Description of the task.
    steps (List[str]): List of steps to guide the reasoning process.
    query (str): The input for which we want to generate an output.

    Returns:
    str: The generated chain-of-thought prompt.
    """
    prompt = f"{task}\n\nLet's approach this step by step:\n"
    for i, step in enumerate(steps, 1):
        prompt += f"{i}. {step}\n"
    prompt += (
        f"\nNow, let's apply these steps to the following input:\n{query}\n\nReasoning:"
    )
    return prompt


def get_api_response(prompt: str) -> Tuple[str, float]:
    """
    Get a response from the OpenAI API.

    Args:
    prompt (str): The prompt to send to the API.

    Returns:
    Tuple[str, float]: The API response and the time taken.
    """
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    end_time = time.time()
    return response.choices[0].message.content.strip(), end_time - start_time


def compare_prompting_strategies(task: str, query: str) -> Dict[str, Any]:
    """
    Compare different prompting strategies for the same task.

    Args:
    task (str): The task description.
    query (str): The input query.

    Returns:
    Dict[str, Any]: A dictionary containing the results and timing for each strategy.
    """
    # Few-shot prompting
    few_shot_examples = [
        {"input": "The product exceeded my expectations!", "output": "Positive"},
        {"input": "I'm disappointed with the quality.", "output": "Negative"},
        {"input": "It's okay, but not great.", "output": "Neutral"},
    ]
    few_shot_prompt = generate_few_shot_prompt(task, few_shot_examples, query)
    few_shot_response, few_shot_time = get_api_response(few_shot_prompt)

    # Chain-of-thought prompting
    cot_steps = [
        "Identify the key words or phrases that indicate sentiment.",
        "Consider the overall tone of the text.",
        "Determine if there are any contradictions or nuances.",
        "Classify the sentiment as Positive, Negative, or Neutral.",
        "Provide a brief explanation for the classification.",
    ]
    cot_prompt = generate_chain_of_thought_prompt(task, cot_steps, query)
    cot_response, cot_time = get_api_response(cot_prompt)

    return {
        "few_shot": {
            "prompt": few_shot_prompt,
            "response": few_shot_response,
            "time": few_shot_time,
        },
        "chain_of_thought": {
            "prompt": cot_prompt,
            "response": cot_response,
            "time": cot_time,
        },
    }


# Example usage
if __name__ == "__main__":
    task = "Determine the sentiment of the following text:"
    query = "I bought this product last week. While it has some good features, I'm not entirely satisfied with its performance."

    results = compare_prompting_strategies(task, query)

    logger.info("Few-shot prompting results:")
    logger.info(f"Response: {results['few_shot']['response']}")
    logger.info(f"Time taken: {results['few_shot']['time']:.2f} seconds\n")

    logger.info("Chain-of-thought prompting results:")
    logger.info(f"Response: {results['chain_of_thought']['response']}")
    logger.info(f"Time taken: {results['chain_of_thought']['time']:.2f} seconds")
