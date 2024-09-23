from typing import Sequence

import yaml
from llama_index.core import (PromptTemplate, Settings, SimpleDirectoryReader,
                              VectorStoreIndex)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI


def initialize_llm(model_name="gpt-3.5-turbo", temperature=0.5):
    Settings.llm = OpenAI(model=model_name, temperature=temperature)

def load_documents(input_dir="./data"):
    reader = SimpleDirectoryReader(input_dir=input_dir, recursive=True)
    docs = reader.load_data()
    return docs

def build_index(docs: Sequence[Document],):
    return VectorStoreIndex.from_documents(docs)

def create_query_engine(index: VectorStoreIndex) -> BaseQueryEngine:
    query_engine = index.as_query_engine(llm=None, verbose=True)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": _load_prompt_template("prompt_template.yaml")}
    )
    return query_engine


def _load_prompt_template(file_path: str) -> PromptTemplate:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    prompt_template_str = data['mydata_prompt']['template']
    return PromptTemplate(prompt_template_str)