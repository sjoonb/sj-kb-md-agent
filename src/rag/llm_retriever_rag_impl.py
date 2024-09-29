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
        if not document["content"]:
            return self._generate_feedback_response(document.get("feedback", {}))

        response = self.generation_chain.invoke({"document": document["content"], "query": query})
        return response.content
    
    def _generate_feedback_response(self, feedback: dict) -> str:
        response = "관련된 문서를 찾을 수 없습니다.\n\n"
        print("feedback: ", feedback)
        
        clarification = feedback.get("clarification_request", "")
        if clarification:
            response += f"{clarification}\n\n"
        
        related_queries = feedback.get("related_queries", [])
        if related_queries:
            response += "관련 질문 제안:\n"
            for i, rq in enumerate(related_queries, 1):
                response += f"{i}. {rq}\n"
        
        return response

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
            result = {
                "reasoning": reasoning,
                "file_name": file_name if file_name.lower() != "null" else None,
            }
            
            # Parse feedback if file_name is null
            if result["file_name"] is None:
                feedback_elem = root.find("feedback")
                if feedback_elem is not None:
                    result["feedback"] = {
                        "clarification_request": feedback_elem.find("clarification_request").text.strip(),
                        "related_queries": [query.text.strip() for query in feedback_elem.findall("related_queries/query")]
                    }
            
            return result
        except ET.ParseError as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Raw response: {response.content}")
            return {"reasoning": "Error parsing LLM response", "file_name": None}

    def _retrieve_document(self, query):
        result = self.retrieval_chain.invoke({"query": query})
        if result["file_name"] and result["file_name"].lower() != "null":

            md_content = self._load_md_content(result["file_name"])
            result["content"] = md_content
        else:
            result["content"] = None
            if "feedback" in result:
                result["feedback"] = result["feedback"]

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
