import logging
from typing import List, Dict, Any, Union
from functools import wraps
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Initialize the language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def error_handler(func):
    """Decorator for error handling"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


class TextProcessor:
    """Class to handle text processing tasks"""

    @error_handler
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize a long article."""
        logger.info(f"Summarizing text of length {len(text)}")
        prompt = ChatPromptTemplate.from_template(
            "Summarize the following text in no more than {max_length} words:\n\n{text}"
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"text": text, "max_length": max_length})

    @error_handler
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text while preserving context and nuance."""
        logger.info(f"Translating text to {target_language}")
        prompt = ChatPromptTemplate.from_template(
            "Translate the following text to {target_language}, preserving context and nuance:\n\n{text}"
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"text": text, "target_language": target_language})

    @error_handler
    def answer_question(self, context: str, question: str) -> str:
        """Answer a question based on the given context."""
        logger.info(f"Answering question: {question}")
        prompt = ChatPromptTemplate.from_template(
            "Answer the following question based on the given context:\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})


class FileReader:
    """Class to handle file reading operations"""

    @staticmethod
    @error_handler
    def read_file(filepath: Union[str, Path]) -> List[Document]:
        """Read content from a PDF or text file."""
        filepath = Path(filepath)
        logger.info(f"Reading file: {filepath}")

        if filepath.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(filepath))
        elif filepath.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(filepath))
        else:
            raise ValueError(f"Unsupported file type: {filepath.suffix}")

        return loader.load()

    @staticmethod
    @error_handler
    def process_long_text(documents: List[Document]) -> List[Document]:
        """Process long text by splitting it into smaller chunks."""
        logger.info("Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        return text_splitter.split_documents(documents)


class LanguageTaskManager:
    """Manager class for language tasks"""

    def __init__(self):
        self.text_processor = TextProcessor()
        self.file_reader = FileReader()

    async def process_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Process a file asynchronously"""
        documents = await asyncio.to_thread(self.file_reader.read_file, filepath)
        chunks = await asyncio.to_thread(self.file_reader.process_long_text, documents)

        full_text = " ".join(doc.page_content for doc in chunks)

        with ThreadPoolExecutor() as executor:
            summary_future = executor.submit(
                self.text_processor.summarize_text, full_text
            )
            translation_future = executor.submit(
                self.text_processor.translate_text, full_text[:500], "hindi"
            )
            answer_future = executor.submit(
                self.text_processor.answer_question,
                full_text,
                "What is the main topic of this text?",
            )

        return {
            "summary": summary_future.result(),
            "translation": translation_future.result(),
            "answer": answer_future.result(),
            "num_chunks": len(chunks),
        }


async def main():
    manager = LanguageTaskManager()

    try:
        # Process a PDF file
        # pdf_results = await manager.process_file('sample.pdf')
        # logger.info("PDF processing results:")
        # logger.info(f"Summary: {pdf_results['summary']}")
        # logger.info(f"Translation (first 500 chars): {pdf_results['translation']}")
        # logger.info(f"Answer to question: {pdf_results['answer']}")
        # logger.info(f"Number of chunks: {pdf_results['num_chunks']}")

        # Process a text file
        txt_results = await manager.process_file("virat.txt")
        logger.info("\nText file processing results:")
        logger.info(f"Summary: {txt_results['summary']}")
        logger.info(f"Translation (first 500 chars): {txt_results['translation']}")
        logger.info(f"Answer to question: {txt_results['answer']}")
        logger.info(f"Number of chunks: {txt_results['num_chunks']}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
