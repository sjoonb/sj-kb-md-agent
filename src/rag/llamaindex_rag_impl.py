import logging
from typing import Sequence

import yaml
from llama_index.core import (PromptTemplate, Settings, SimpleDirectoryReader,
                              StorageContext, VectorStoreIndex,
                              get_response_synthesizer,
                              load_index_from_storage)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI

from src.rag.interfaces import IRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaIndexRAGImpl(IRAG):
    def __init__(self, input_dir="./data", index_dir="./index", model_name="gpt-3.5-turbo"):
        self.input_dir = input_dir
        self.index_dir = index_dir
        self.model_name = model_name
        self.index = None
        self.query_engine = None
        logger.info(f"RAG initialized with input_dir={input_dir}, index_dir={index_dir}, model_name={model_name}")

        self._initialize_rag()

    def _initialize_rag(self):
        logger.info("Initializing RAG System...")
        self._initialize_llm()
        self._load_or_build_index()
        self._create_retriever()
        self._create_query_engine()
        logger.info("RAG System initialization complete.")

    def query(self, prompt: str):
        if not self.query_engine:
            logger.error("RAG System not initialized. Call initialize() first.")
            raise ValueError("RAG System not initialized. Call initialize() first.")
        logger.info(f"Querying RAG System with prompt: {prompt[:50]}...")  # Log first 50 chars of prompt for brevity
        response = self.query_engine.query(prompt)

        source_nodes = response.source_nodes
        for node_with_score in source_nodes:
            node = node_with_score.node
            metadata = node.metadata
            text_content = node.text
            
            # Print metadata
            print("Document Metadata:")
            for key, value in metadata.items():
                print(f"{key}: {value}")
            
            # Print a snippet of the document's content
            print("\nDocument Content Snippet:")
            print(text_content[:500])  # Print the first 500 characters of the content
            print() 

        logger.info("Query completed successfully.")
        return response.response

    def _initialize_llm(self):
        logger.info(f"Initializing LLM with model={self.model_name}")
        Settings.llm = OpenAI(model=self.model_name)
        logger.info("LLM initialization complete.")

    def _load_documents(self):
        logger.info(f"Loading documents from {self.input_dir}")
        reader = SimpleDirectoryReader(input_dir=self.input_dir, recursive=True)
        docs = reader.load_data()
        logger.info(f"Loaded {len(docs)} documents.")
        return docs

    def _load_or_build_index(self):
        logger.info("Attempting to load existing index...")
        try:
            self.index = self._load_existing_index()
            logger.info("Existing index loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load existing index: {str(e)}. Building new index...")
            docs = self._load_documents()
            self.index = self._build_index(docs)
            self._store_index()
            logger.info("New index built and stored successfully.")

    def _load_existing_index(self):
        logger.info(f"Loading existing index from {self.index_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
        index = load_index_from_storage(storage_context)
        logger.info("Existing index loaded successfully.")
        return index

    def _build_index(self, docs: Sequence[Document]):
        logger.info(f"Building new index from {len(docs)} documents...")
        index = VectorStoreIndex.from_documents(docs, show_progress=True)
        logger.info("Index built successfully.")
        return index

    def _store_index(self):
        logger.info(f"Storing index to {self.index_dir}")
        self.index.storage_context.persist(self.index_dir)
        logger.info("Index stored successfully.")

    def _create_retriever(self):
        logger.info("Creating retriever...")
        self. retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
        )
    
    def _create_query_engine(self):
        # assemble query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=get_response_synthesizer(),
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        )

        # self.query_engine = self.index.as_query_engine(llm=None, verbose=True)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template":
              PromptTemplate(self._load_prompt_template('llamaindex_prompt'))
            }
        )

        # print updated prompt
        prompts = self.query_engine.get_prompts()
        for k, p in prompts.items():
            prompt_text = f"Prompt Key: {k}\nText:"
            print(prompt_text)
            print(p.get_template())
            print("")

        logger.info("Query engine created successfully.")
