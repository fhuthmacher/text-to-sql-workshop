from pydantic import BaseModel
from typing import List, Dict
from abc import ABC, abstractmethod
import chromadb
from chromadb.api.types import EmbeddingFunction
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from chromadb.utils.embedding_functions import AmazonBedrockEmbeddingFunction
from utils.splitter import RAGChunk

class RetrievalResult(BaseModel):
    id: str
    document: str
    embedding: List[float]
    distance: float
    metadata: Dict = {}

# Base retrieval class. Can be reused if you decide to implement a different retrieval class.
class BaseRetrievalTask(ABC):
    @abstractmethod
    def retrieve(self, query_text: str, n_results: int) -> List[RetrievalResult]:
        """
        Retrieve documents based on the given query.

        Args:
            query (str): The query string to search for.

        Returns:
            List[RetrievalResult]: A list of RetrievalResult objects that are relevant to the query.
        """
        pass



# Example of a concrete implementation
class ChromaDBRetrievalTask(BaseRetrievalTask):

    def __init__(self, chroma_client, collection_name: str, embedding_function, chunks: List[RAGChunk] = None):
        self.client = chroma_client
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.chunks = chunks

        # Create the collection
        self.collection = self._create_collection()

    def _create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def add_chunks_to_collection(self, batch_size: int = 20, num_workers: int = 10):
        batches = [self.chunks[i:i + batch_size] for i in range(0, len(self.chunks), batch_size)]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._add_batch, batch) for batch in batches]
            for future in as_completed(futures):
                future.result()  # This will raise an exception if one occurred during the execution
        print('Finished Ingesting Chunks Into Collection')

    def _add_batch(self, batch: List[RAGChunk]):
        self.collection.add(
            ids=[chunk.id_ for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch]
        )

    def retrieve(self, query_text: str, n_results: int = 5) -> List[RetrievalResult]:
        # Query the collection
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['embeddings', 'documents', 'metadatas', 'distances']
        )

        # Transform the results into RetrievalResult objects
        retrieval_results = []
        for i in range(len(results['ids'][0])):
            retrieval_results.append(RetrievalResult(
                id=results['ids'][0][i],
                document=results['documents'][0][i],
                embedding=results['embeddings'][0][i],
                distance=results['distances'][0][i],
                metadata=results['metadatas'][0][i] if results['metadatas'][0] else {}
            ))

        return retrieval_results