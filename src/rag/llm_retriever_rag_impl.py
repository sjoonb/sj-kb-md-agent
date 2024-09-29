import logging
import os
import xml.etree.ElementTree as ET
import re

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
_FAQ_DATA_PATH = "./src/rag/llm_retriever_rag_resource/faqs/마이데이터 기술 가이드라인_6_Q&A.md"

class LlmRetrieverRAGImpl(IRAG):
    def query(self, query: str):
        logger.info(f"Querying LLM Retriever RAG System with prompt: {query[:50]}...")
        
        # First, try to find a match in FAQ
        faq_result = self._search_faq(query)
        if faq_result["match_found"]:
            return faq_result["content"]
        
        # If no FAQ match, proceed with document retrieval
        document = self._retrieve_document(query)
        if not document["content"]:
            return self._generate_feedback_response(document.get("feedback", {}))

        response = self.generation_chain.invoke({"document": document["content"], "query": query})
        return response.content
    
    def __init__(self):
        self._initialize_faq_chain()
        self._initialize_retrieval_chain()
        self._initialize_generation_chain()
        self._load_faq_data()

    def _initialize_faq_chain(self):
        self.faq_prompt = PromptTemplate(
            input_variables=["faq_list", "query"],
            template=self._load_prompt_template("llm_retrieval_prompt", "faq_search_prompt")
        )
        self.faq_chain = (
            {
                "query": RunnablePassthrough(),
                "faq_list": lambda _: self.faq_data
            }
            | self.faq_prompt
            | ChatOpenAI(temperature=0, model="gpt-4o")
            | self._parse_faq_response
        )

    def _load_faq_data(self):
        with open(_FAQ_DATA_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split the content into FAQ entries
        faq_entries = re.split(r'\n(?=# )', content)
        
        # Process each entry into a structured format
        self.faq_data = []
        for entry in faq_entries:
            lines = entry.strip().split('\n')
            if lines and lines[0].startswith('# '):
                question = lines[0][2:].strip()
                answer = '\n'.join(lines[1:]).strip()
                self.faq_data.append({"question": question, "answer": answer})

    def _search_faq(self, query):
        result = self.faq_chain.invoke({"query": query})
        logger.info(f"FAQ search result: {result}")
        return result

    def _parse_faq_response(self, response):
        try:
            root = ET.fromstring(response.content.strip())
            return {
                "match_found": root.find("match_found").text.lower() == "true",
                "reasoning": root.find("reasoning").text.strip(),
                "content": root.find("content").text.strip() if root.find("content").text else None
            }
        except ET.ParseError as e:
            logger.error(f"FAQ LLM 응답 파싱 중 오류 발생: {e}")
            return {"match_found": False, "reasoning": "LLM 응답 파싱 중 오류 발생", "content": None}

    def _generate_feedback_response(self, feedback: dict) -> str:
        response = "관련된 내용을 찾을 수 없습니다.\n\n"
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
            cleaned_response = response.content.strip()
            root = ET.fromstring(cleaned_response)
            reasoning = root.find("reasoning").text.strip()
            file_name = root.find("file_name").text.strip()
            result = {
                "reasoning": reasoning,
                "file_name": file_name if file_name.lower() != "null" else None,
            }
            
            if result["file_name"] is None:
                feedback_elem = root.find("feedback")
                if feedback_elem is not None:
                    result["feedback"] = {
                        "clarification_request": feedback_elem.find("clarification_request").text.strip(),
                        "related_queries": [query.text.strip() for query in feedback_elem.findall("related_queries/query")]
                    }
            
            return result
        except ET.ParseError as e:
            logger.error(f"LLM 응답 파싱 중 오류 발생: {e}")
            logger.error(f"Raw response: {response.content}")
            return {"reasoning": "LLM 응답 파싱 중 오류 발생", "file_name": None}

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
