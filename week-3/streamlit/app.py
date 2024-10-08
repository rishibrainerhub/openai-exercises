import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
from openai_service import OpenAiService


def main():
    # Initialize the OpenAI service
    openai_service = OpenAiService()

    st.title("DALL-E Image Generator and Editor")

    # Image Generation
    st.header("Generate Image")
    generation_prompt = st.text_input("Enter a description for image generation:")
    if st.button("Generate Image"):
        if generation_prompt:
            with st.spinner("Generating image..."):
                generated_image = openai_service.generate_image(generation_prompt)
                if generated_image:
                    st.image(
                        generated_image,
                        caption="Generated Image",
                        use_column_width=True,
                    )
                else:
                    st.error("Failed to generate image. Please try again.")
        else:
            st.warning("Please enter a description for image generation.")

    # Image Editing
    st.header("Edit Image")
    uploaded_image = st.file_uploader(
        "Upload an image to edit:", type=["png", "jpg", "jpeg"]
    )
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.subheader("Create Mask")
        st.write(
            "Use the drawing tool to create a mask. The white area will be edited."
        )

        # Create a canvas for the mask
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",  # Semi-transparent white
            stroke_width=30,
            stroke_color="#fff",
            background_color="#000",
            background_image=image,
            height=image.height,
            width=image.width,
            drawing_mode="freedraw",
            key="canvas",
        )

        editing_prompt = st.text_input("Enter a description for image editing:")
        if st.button("Edit Image"):
            if editing_prompt:
                if canvas_result.image_data is not None:
                    # Convert the drawing to a PIL Image
                    mask = Image.fromarray(
                        (canvas_result.image_data[:, :, 3] > 0).astype(np.uint8) * 255
                    )

                    with st.spinner("Editing image..."):
                        edited_image = openai_service.edit_image(
                            image, mask, editing_prompt
                        )
                        if edited_image:
                            st.image(
                                edited_image,
                                caption="Edited Image",
                                use_column_width=True,
                            )
                        else:
                            st.error("Failed to edit image. Please try again.")
                else:
                    st.warning("Please draw a mask before editing.")
            else:
                st.warning("Please enter a description for image editing.")


if __name__ == "__main__":
    main()
