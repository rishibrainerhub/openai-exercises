import os
import time
from typing import Optional, Callable, Any, TypeVar

import openai
from openai.types import FileObject
from openai.types.fine_tuning import FineTuningJob

import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

import os

# Type variable for generic function
T = TypeVar("T")


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


@handle_api_error
def upload_training_file(file_path: str) -> Optional[FileObject]:
    """
    Upload a training file to OpenAI.

    Args:
        file_path (str): The path to the training file.

    Returns:
        Optional[FileObject]: The uploaded file object if successful, None otherwise.
    """
    try:
        with open(file_path, "rb") as file:
            response = openai.files.create(file=file, purpose="fine-tune")
        logger.info(f"File uploaded successfully. File ID: {response.id}")
        return response
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return None


@handle_api_error
def create_fine_tuning_job(
    training_file: str, model: str = "gpt-3.5-turbo"
) -> Optional[FineTuningJob]:
    """
    Create a fine-tuning job using the specified training file and model.

    Args:
        training_file (str): The ID of the training file.
        model (str): The name of the base model to fine-tune.

    Returns:
        Optional[FineTuningJob]: The created fine-tuning job object if successful, None otherwise.
    """
    try:
        job = openai.fine_tuning.jobs.create(training_file=training_file, model=model)
        logger.info(f"Fine-tuning job created: {job.id}")
        return job
    except Exception as e:
        logger.error(f"Error creating fine-tuning job: {str(e)}")
        return None


@handle_api_error
def monitor_fine_tuning_job(job_id: str, polling_interval: int = 60) -> Optional[str]:
    """
    Monitor the progress of a fine-tuning job.

    Args:
        job_id (str): The ID of the fine-tuning job to monitor.
        polling_interval (int): The number of seconds to wait between status checks.

    Returns:
        Optional[str]: The name of the fine-tuned model if successful, None otherwise.
    """
    while True:
        try:
            job = openai.fine_tuning.jobs.retrieve(job_id)
            print(f"Status: {job.status}")

            if job.status == "succeeded":
                return job.fine_tuned_model
            elif job.status == "failed":
                print("Fine-tuning job failed.")
                print(f"Failure reason: {job.error.message}")
                return None

            time.sleep(polling_interval)
        except Exception as e:
            print(f"Error monitoring fine-tuning job: {str(e)}")
            return None


def main():
    file_path = "customer-inquiries.jsonl"

    uploaded_file = upload_training_file(file_path)
    if not uploaded_file:
        print("Failed to upload training file. Exiting.")
        return

    job = create_fine_tuning_job(uploaded_file.id)
    if not job:
        print("Failed to create fine-tuning job. Exiting.")
        return

    fine_tuned_model = monitor_fine_tuning_job(job.id)

    if fine_tuned_model:
        print(f"Fine-tuned model name: {fine_tuned_model}")
    else:
        print("Fine-tuning failed or was interrupted.")

        # Fetch and print detailed job information
        try:
            detailed_job = openai.fine_tuning.jobs.retrieve(job.id)
            print("\nDetailed job information:")
            print(f"Status: {detailed_job.status}")
            print(f"Error: {detailed_job.error}")
            print(f"Created at: {detailed_job.created_at}")
            print(f"Finished at: {detailed_job.finished_at}")
            print(f"Model: {detailed_job.model}")
            print(f"Fine-tuned model: {detailed_job.fine_tuned_model}")
            print(f"Organization ID: {detailed_job.organization_id}")
            print(f"Result files: {detailed_job.result_files}")
            print(f"Trained tokens: {detailed_job.trained_tokens}")
            print(f"Validation file: {detailed_job.validation_file}")
        except Exception as e:
            print(f"Error fetching detailed job information: {str(e)}")


if __name__ == "__main__":
    main()
