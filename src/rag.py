# rag_system.py

from typing import Sequence
import yaml
import logging
from llama_index.core import (PromptTemplate, Settings, SimpleDirectoryReader,
                              StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, input_dir="./data", index_dir="./index", model_name="gpt-3.5-turbo", temperature=0.5):
        self.input_dir = input_dir
        self.index_dir = index_dir
        self.model_name = model_name
        self.temperature = temperature
        self.index = None
        self.query_engine = None
        logger.info(f"RAGSystem initialized with input_dir={input_dir}, index_dir={index_dir}, model_name={model_name}, temperature={temperature}")

    def initialize(self):
        logger.info("Initializing RAG System...")
        self._initialize_llm()
        self._load_or_build_index()
        self._create_query_engine()
        logger.info("RAG System initialization complete.")

    def query(self, prompt: str):
        if not self.query_engine:
            logger.error("RAG System not initialized. Call initialize() first.")
            raise ValueError("RAG System not initialized. Call initialize() first.")
        logger.info(f"Querying RAG System with prompt: {prompt[:50]}...")  # Log first 50 chars of prompt for brevity
        response = self.query_engine.query(prompt)
        logger.info("Query completed successfully.")
        return response

    def _initialize_llm(self):
        logger.info(f"Initializing LLM with model={self.model_name}, temperature={self.temperature}")
        Settings.llm = OpenAI(model=self.model_name, temperature=self.temperature)
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

    def _create_query_engine(self):
        logger.info("Creating query engine...")
        self.query_engine = self.index.as_query_engine(llm=None, verbose=True)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": self._load_prompt_template("prompt_template.yaml")}
        )
        logger.info("Query engine created successfully.")

    def _load_prompt_template(self, file_path: str) -> PromptTemplate:
        logger.info(f"Loading prompt template from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        prompt_template_str = data['mydata_prompt']['template']
        logger.info("Prompt template loaded successfully.")
        return PromptTemplate(prompt_template_str)