import os
import logging
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class CustomerServiceChatbot:
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.context: Dict[str, Any] = {}
        self.persona = """
        You are a friendly and helpful customer service representative for a fictional online bookstore called "ReadMore".
        Your name is Alex. You're knowledgeable about books, reading habits, and general customer service.
        Always maintain a polite and professional tone. If you don't know an answer, it's okay to say so.
        Try to understand the customer's needs and provide relevant information or assistance.
        """

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role (str): The role of the message sender (e.g., "user" or "assistant")
            content (str): The content of the message
        """
        self.conversation_history.append({"role": role, "content": content})

    def get_chat_completion(self, user_input: str) -> str:
        """
        Get a chat completion from the OpenAI API.

        Args:
            user_input (str): The user's input message

        Returns:
            str: The chatbot's response

        Raises:
            Exception: If there's an error in getting the chat completion
        """
        try:
            self.add_message("user", user_input)

            messages = [
                {"role": "system", "content": self.persona}
            ] + self.conversation_history

            response = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, max_tokens=150
            )

            assistant_message = response.choices[0].message.content.strip()
            self.add_message("assistant", assistant_message)

            return assistant_message
        except Exception as e:
            logger.error(f"Error in getting chat completion: {str(e)}")
            raise

    def update_context(self, user_input: str) -> None:
        """
        Update the context based on user input.

        Args:
            user_input (str): The user's input message
        """
        # This is a simple implementation. In a real-world scenario, you might use
        # more sophisticated NLP techniques to extract and update context.
        lower_input = user_input.lower()
        if "book" in lower_input:
            self.context["topic"] = "books"
        elif "delivery" in lower_input or "shipping" in lower_input:
            self.context["topic"] = "delivery"
        elif "return" in lower_input or "refund" in lower_input:
            self.context["topic"] = "returns"

    def generate_response(self, user_input: str) -> str:
        """
        Generate a response based on user input and current context.

        Args:
            user_input (str): The user's input message

        Returns:
            str: The chatbot's response
        """
        self.update_context(user_input)

        context_prompt = f"The current topic seems to be about {self.context.get('topic', 'general inquiries')}. "
        contextualized_input = context_prompt + user_input

        return self.get_chat_completion(contextualized_input)


def chat_interface() -> None:
    """
    Implement a simple CLI interface for interacting with the chatbot.
    """
    chatbot = CustomerServiceChatbot()
    print("Welcome to ReadMore Customer Service! How can I assist you today?")
    print("(Type 'quit' to exit)")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            print("Thank you for using ReadMore Customer Service. Have a great day!")
            break

        try:
            response = chatbot.generate_response(user_input)
            print(f"Alex: {response}")
        except Exception as e:
            logger.error(f"Error in chat interface: {str(e)}")
            print(
                "I apologize, but I'm having trouble processing your request. Could you please try again?"
            )


if __name__ == "__main__":
    try:
        chat_interface()
    except Exception as e:
        logger.critical(f"Critical error in main program: {str(e)}")
        print("A critical error occurred. Please check the logs for details.")
