from typing import Optional
from openai import OpenAI
import os
import io
import logging
from PIL import Image
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class OpenAiService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_image(self, prompt: str) -> Optional[Image.Image]:
        """
        Generate an image based on the given text prompt using DALL-E.

        Args:
            prompt (str): The text description for image generation.

        Returns:
            Optional[Image.Image]: The generated image, or None if generation fails.
        """
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url

            # Download the image
            import requests

            image_data = requests.get(image_url).content
            image = Image.open(io.BytesIO(image_data))

            logger.info(f"Generated image for prompt: {prompt[:50]}...")
            return image
        except Exception as e:
            logger.error(f"Error in image generation: {str(e)}")
            return None

    def edit_image(
        self, image: Image.Image, mask: Image.Image, prompt: str
    ) -> Optional[Image.Image]:
        """
        Edit an existing image based on the given mask and prompt using DALL-E.

        Args:
            image (Image.Image): The original image to edit.
            mask (Image.Image): The mask indicating areas to edit.
            prompt (str): The text description for image editing.

        Returns:
            Optional[Image.Image]: The edited image, or None if editing fails.
        """
        try:
            # Convert images to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()

            mask_byte_arr = io.BytesIO()
            mask.save(mask_byte_arr, format="PNG")
            mask_byte_arr = mask_byte_arr.getvalue()

            response = self.client.images.edit(
                model="dall-e-2",
                image=img_byte_arr,
                mask=mask_byte_arr,
                prompt=prompt,
                n=1,
                size="1024x1024",
            )
            image_url = response.data[0].url

            # Download the image
            import requests

            image_data = requests.get(image_url).content
            edited_image = Image.open(io.BytesIO(image_data))

            logger.info(f"Edited image with prompt: {prompt[:50]}...")
            return edited_image
        except Exception as e:
            logger.error(f"Error in image editing: {str(e)}")
            return None
