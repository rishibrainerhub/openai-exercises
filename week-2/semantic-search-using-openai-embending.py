import logging
from typing import List, Dict, Tuple
import os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_corpus() -> List[Dict[str, str]]:
    """
    Create a corpus of text documents.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing document id and content.
    """
    corpus = [
        {"id": "doc1", "content": "The quick brown fox jumps over the lazy dog."},
        {
            "id": "doc2",
            "content": "A journey of a thousand miles begins with a single step.",
        },
        {"id": "doc3", "content": "To be or not to be, that is the question."},
        {"id": "doc4", "content": "All that glitters is not gold."},
        {"id": "doc5", "content": "Where there's a will, there's a way."},
    ]
    logger.info(f"Created corpus with {len(corpus)} documents.")
    return corpus


def generate_embeddings(corpus: List[Dict[str, str]]) -> Dict[str, List[float]]:
    """
    Generate embeddings for each document using the OpenAI Embeddings API.

    Args:
        corpus (List[Dict[str, str]]): The corpus of documents.

    Returns:
        Dict[str, List[float]]: A dictionary mapping document ids to their embeddings.
    """
    embeddings = {}
    try:
        for doc in corpus:
            response = client.embeddings.create(
                model="text-embedding-ada-002", input=doc["content"]
            )
            embeddings[doc["id"]] = response.data[0].embedding
        logger.info(f"Generated embeddings for {len(embeddings)} documents.")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Args:
        a (List[float]): First vector.
        b (List[float]): Second vector.

    Returns:
        float: Cosine similarity between the two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_similar_documents(
    query: str, embeddings: Dict[str, List[float]], top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Find the most semantically similar documents given a query.

    Args:
        query (str): The search query.
        embeddings (Dict[str, List[float]]): The pre-computed embeddings for each document.
        top_k (int): The number of top similar documents to return.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing document id and similarity score.
    """
    try:
        query_embedding = (
            client.embeddings.create(model="text-embedding-ada-002", input=query)
            .data[0]
            .embedding
        )

        similarities = []
        for doc_id, doc_embedding in embeddings.items():
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((doc_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    except Exception as e:
        logger.error(f"Error finding similar documents: {str(e)}")
        raise


def search_interface():
    """
    Create a simple search interface that uses the semantic search functionality.
    """
    corpus = create_corpus()
    embeddings = generate_embeddings(corpus)

    while True:
        query = input("Enter your search query (or 'quit' to exit): ")
        if query.lower() == "quit":
            break

        try:
            results = find_similar_documents(query, embeddings)
            print("\nSearch Results:")
            for doc_id, similarity in results:
                doc_content = next(
                    doc["content"] for doc in corpus if doc["id"] == doc_id
                )
                print(f"Document: {doc_id}")
                print(f"Content: {doc_content}")
                print(f"Similarity: {similarity:.4f}\n")
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            print("An error occurred during the search. Please try again.")


if __name__ == "__main__":
    try:
        search_interface()
    except Exception as e:
        logger.critical(f"Critical error in main program: {str(e)}")
        print("A critical error occurred. Please check the logs for details.")
