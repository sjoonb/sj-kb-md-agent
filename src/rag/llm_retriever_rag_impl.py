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

class LlmRetreivalRAGImpl(IRAG):
    def __init__(self):
      return

    def initialize(self):
      return

    def query(self, prompt: str):
        return """
        
        
        
        
        
        ehl
        
        
        
        x-api-tran-id는 다음과 같은 특징을 가집니다:

1. 정의: x-api-tran-id는 거래고유번호입니다.
2. 목적: API를 송수신한 기관 간 거래추적이 필요한 경우(민원대응, 장애처리 등) 거래를 식별하기 위해 사용됩니다.
3. 사용 위치: HTTP 요청 및 응답 헤더에 포함됩니다.
4. 데이터 형식: AN (25) 형식으로, 25자리의 영숫자로 구성됩니다.
5. 생성 규칙: 거래고유번호 생성 규칙은 [첨부14]에서 확인할 수 있습니다.
6. 사용 방법:
   - API 요청 시: 헤더에 x-api-tran-id 값을 설정하여 전송합니다.
   - API 응답 시: 요청에서 받은 x-api-tran-id 값을 그대로 응답 헤더에 포함하여 반환합니다."""