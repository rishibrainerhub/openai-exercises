import os

os.environ["OPENAI_API_KEY"] = (
    "your-api-key-here"
)

from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

def main():
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "who is the president of the United States?",
            }
        ],
        model="gpt-3.5-turbo",
    )
    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    main()
