import logging
import os
import xml.etree.ElementTree as ET

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.rag.interfaces import IRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_LLM_RETRIEVER_RAG_DOCS_DIR = "./src/rag/llm_retriever_rag_resource/docs"


class LlmRetrieverRAGImpl(IRAG):
    def query(self, query: str):
        logger.info(f"Querying LLM Retriever RAG System with prompt: {query[:50]}...")
        document = self._retrieve_document(query)
        response = self.generation_chain.invoke({"document": document, "query": query})
        return response.content

    def __init__(self):
        self._initialize_retrieval_chain()
        self._initialize_generation_chain()

    def _initialize_retrieval_chain(self):
        file_name_list = self._get_md_file_list()
        self.retrieval_prompt = PromptTemplate(
            input_variables=["file_name_list", "query"],
            template=self._load_prompt_template(
                "llm_retrieval_prompt", "find_document_prompt"
            ),
        )
        self.retrieval_chain = (
            {
                "query": RunnablePassthrough(),
                "file_name_list": lambda _: "\n".join(file_name_list),
            }
            | self.retrieval_prompt
            | ChatOpenAI(temperature=0, model="gpt-4o")
            | self._parse_llm_response
        )

    def _get_md_file_list(self):
        return [f for f in os.listdir(_LLM_RETRIEVER_RAG_DOCS_DIR) if f.endswith(".md")]

    def _parse_llm_response(self, response):
        logger.info(f"Raw LLM response\n: {response.content}")
        try:
            # Remove any leading/trailing whitespace and ensure we're parsing valid XML
            cleaned_response = response.content.strip()
            root = ET.fromstring(cleaned_response)
            reasoning = root.find("reasoning").text.strip()
            file_name = root.find("file_name").text.strip()
            return {
                "reasoning": reasoning,
                "file_name": file_name if file_name.lower() != "null" else None,
            }
        except ET.ParseError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response.content}")
            return {"reasoning": "Error parsing LLM response", "file_name": None}

    def _retrieve_document(self, query):
        result = self.retrieval_chain.invoke({"query": query})
        if result["file_name"] and result["file_name"].lower() != "null":

            md_content = self._load_md_content(result["file_name"])
            result["content"] = md_content
        else:
            result["content"] = None

        return result

    def _load_md_content(self, file_name):
        file_path = os.path.join(_LLM_RETRIEVER_RAG_DOCS_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _initialize_generation_chain(self):
        self.generation_prompt = PromptTemplate(
            input_variables=["document", "query"],
            template=self._load_prompt_template(
                "llm_retrieval_prompt", "generation_template"
            ),
        )
        self.generation_chain = (
            {"document": RunnablePassthrough(), "query": RunnablePassthrough()}
            | self.generation_prompt
            | ChatOpenAI(temperature=0, model="gpt-4o-mini")
        )
