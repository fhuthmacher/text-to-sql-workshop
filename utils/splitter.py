from typing import List, Dict, Any
from pydantic import BaseModel
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Node
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
import pandas as pd
import re

# Create a class to use instead of LlamaIndex Nodes. This way we decouple our chroma collections from LlamaIndexes
class RAGChunk(BaseModel):
    id_: str
    text: str
    metadata: Dict[str, Any] = {}


class DataFrameChunkingStrategy:
    def __init__(self, df: pd.DataFrame, chunk_size: int = 512, chunk_overlap: int = 0):
        self.df = df
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self) -> IngestionPipeline:
        transformations = [
            SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
        ]
        return IngestionPipeline(transformations=transformations)

    def load_documents(self) -> List[Document]:
        documents = []
        for index, row in self.df.iterrows():
            combined_text = f"Question: {row['Question']}\nQuery: {row['Query']}"
            doc = Document(
                text=combined_text,
                metadata={
                    'index': str(index),  # Convert to string to reduce metadata size
                }
            )
            documents.append(doc)
        return documents

    def to_ragchunks(self, nodes: List[Node]) -> List[RAGChunk]:
        return [
            RAGChunk(
                id_=node.node_id,
                text=node.text,
                metadata=node.metadata
            )
            for node in nodes
        ]

    def process(self) -> List[RAGChunk]:
        documents = self.load_documents()
        nodes = self.pipeline.run(documents=documents)
        rag_chunks = self.to_ragchunks(nodes)
        
        print(f"Processing complete. Created {len(rag_chunks)} chunks.")
        return rag_chunks