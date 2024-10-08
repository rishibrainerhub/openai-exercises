# Streamlit DALL-E Image Generator and Editor

A Streamlit web application that allows users to generate and edit images using OpenAI's DALL-E API. The app provides an intuitive interface for both image generation from text prompts and interactive image editing with a drawing tool for masking.

## Features

- 🎨 Generate images using DALL-E 3 from text descriptions
- ✏️ Edit existing images using DALL-E 2
- 🎭 Interactive mask drawing tool
- 📤 Image upload support
- 🖼️ Real-time preview of generated and edited images

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── __init__.py         # Python package initializer
├── openai_service.py   # OpenAI API service wrapper
```

## Prerequisites

- Python 3.12 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

## Required Dependencies

```
streamlit
pillow
streamlit-drawable-canvas
numpy
python-dotenv
openai
requests
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. The app will open in your default web browser with two main sections:

   - **Image Generation**: Enter a text prompt to generate a new image using DALL-E 3
   - **Image Editing**: Upload an image, draw a mask, and enter a prompt to edit the selected area using DALL-E 2

## Features in Detail

### Image Generation
- Enter a descriptive prompt in the text input field
- Click "Generate Image" to create a new image
- The generated image will be displayed below the input field

### Image Editing
1. Upload an image using the file uploader
2. Use the drawing tool to create a mask:
   - White areas indicate regions to be edited
   - The stroke width can be adjusted for precise masking
3. Enter a prompt describing the desired changes
4. Click "Edit Image" to apply the changes

## Error Handling

The application includes comprehensive error handling for:
- Invalid API responses
- Image generation/editing failures
- Missing input validations
- File upload issues

## Logging

The application includes logging functionality that tracks:
- Image generation attempts
- Editing operations
- Error occurrences

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI's DALL-E](https://openai.com/dall-e-3)