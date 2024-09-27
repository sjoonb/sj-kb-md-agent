import logging
from typing import Sequence

import yaml
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from src.rag.interfaces import IRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlmRetreivalRAGImpl(IRAG):
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        self.prompt = PromptTemplate(
            input_variables=["document", "query"],
            template=self._load_prompt_template("llm_retrieval_prompt")
        )
        self.chain = (
            {"document": RunnablePassthrough(), "query": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )

    def _retrieve_document(self):
        return """
                   금융분야 마이데이터 표준API 규격
  제2장. 표준 API 기본규격
2.1 기본 규격
 데이터 표현규격
ᄋ (JSON) 데이터를 교환하기 위한 API의 메시지 형식으로 JSON* 방식
을 사용함
ᄋ (메시지 인코딩 방식) 메시지 전송을 위한 인코딩 방식으로 UTF-8*을 사용함
ᄋ (명명 규칙) API 명세 내 URI와 파라미터의 명명으로‘Under Scores’ 표기법(각 소문자 영단어가 밑줄 문자(_)로 연결)을 사용함
- 단, OAuth 2.0과 같이 타 표준에서 요구하는 URI나 파라미터 명명 규칙 등이 존재하는 경우 해당 표준을 우선 준용함
ᄋ (통화코드 표현 형식) 통화코드 표현으로 ISO 4217 표준규격을 사용함 * 예시 : KRW (한국원화)
       * JSON(JavaScript Object Notation)
- 용량이 적은 메시지를 송수신하기 위해 데이터 객체를 속성·값(Key:Value) 형
식으로 표현하는 개방형 표준 메시지 형식
         * UTF-8
- ASCII 코드를 확장하여 전 세계의 모든 문자코드를 표현할 수 있는 표준 인
코딩 방식으로써, 범용성이 높아 호환성이 우수
   금융보안원 www.fsec.or.kr - 5 -

    데이터 통신규격
ᄋ (네트워크 구간 보호) 참여주체(정보제공자, 마이데이터사업자, 종합포털) 간 안전한 통신을 위한 보호 방안으로, TLS* 기반 상호인증(Mutual Authentication) 및 전송구간 암호화를 사용
- 기관 간 전용선 또는 VPN으로 연결한 경우 TLS 기반 상호인증 생략
ᄋ (메시지 교환방식) API 요청 및 응답(메시지) 교환방식으로 REST* 방식을 사용함
- API 명세(제5장~제7장)의 “필수”가 “N”인 항목의 경우, 회신할 값이 없으면 해당 항목은 응답·요청메시지에서 제외
* NULL, 빈값 등으로 회신하지 않음
ᄋ (URI 계층 구조) 업권별 웹서버 상의 자원(리소스)을 고유하게 식별하고, 위치를 지정할 수 있도록 URI 계층 구조는 다음과 같이 구성
금융분야 마이데이터 표준API 규격
         * TLS (Transport Layer Security)
- 전송(Transport) 계층과 응용(Application) 계층 사이에서 종단 간 인증, 전송
데이터의 암호화, 무결성을 보장하는 표준 프로토콜 - 안전한 전송을 위해 TLS 1.3이상 버전 사용
         * REST (REpresentational State Transfer)
- HTTP 기반으로 데이터를 전달하는 프레임워크로써, URI로 정보의 자원을 표현하고
자원에 대한 행위를 HTTP 메소드(GET, POST 등)로 표현
- 다만, 보안상 이유로 HTTP 메소드 중 GET, POST만 제한적 이용
    분류
URI 계층 구조
인증 API
<base path> / oauth / 2.0 / [authorize | token | revoke]
업권별 정보제공 API
<base path> / <version> / <industry> / <resource>
지원 API
<base path> / <version> / <resource>
            금융보안원 www.fsec.or.kr - 6 -

   금융분야 마이데이터 표준API 규격
  - <base path> 정보제공자의 API서버 도메인명(또는 IP)로써 종합포 털에 기관정보와 함께 등록한 정보
- <version> 표준API 규격 버전정보로써 “v”+숫자로 표기(예: v1, v34 등)
- <industry> 업권 정보
• 업권을 URI로 구분하고, 업권 간 URI 중복을 방지하기 위해
<industry>로 구분
• 지원 API는 종합포털과 마이데이터사업자/정보제공자 간 이용되
는 API이기 때문에 <industry> 불필요
  업권
<industry> 값
은행
bank
카드
card
금융투자
invest
보험
insu
전자금융
efin
할부금융
capital
보증보험 (서울보증보험)
ginsu
통신
telecom
P2P
p2p
인수채권 (한국자산관리공사, 국민행복기금, 케이알앤씨, 농협자산관리회사 등 채권인수회사)
bond
대부 (금전대부업자, 매입추심업자)
usury
                           • 업권이 복수 개인 기관(예:겸영여신업자 등)의 경우, 종합포털로 부터 업권별 기관코드가 각각 발급되며, 따라서 종합포털에 기관 등록 시 업권 수만큼 복수 개의 기관 등록 필요
* 즉, 기관 별 <industry>는 1:1 관계 (한 기관이 복수 개의 <industry>를 가질 수 없음)
- <resource> 정보제공자가 보유하고 있는 특정 자원을 요청하기 위한 식별 정보로써, 정보제공자가 제공해야 할 API를 특정
 금융보안원 www.fsec.or.kr - 7 -

   • API 명세(제5장 ~ 제7장) 내 “API 명 (URI)”는 <resource>만을 표기하였으며, 따라서 실제 URI는 <version>, <industry> 등을 포함 (상세 URI는 제4장 참조)
금융분야 마이데이터 표준API 규격
    URI 예시
    예시1) A 은행이 제공해야 하는 “업권별 정보제공 API” 목록 :
<base path> / <version> / bank / <resource>
예시2) B 전자금융업자가 제공해야 하는 “업권별 정보제공 API” 목록 :
<base path> / <version> / efin / <resource>
예시3) 은행업, 카드업 모두 수행하는 C기관이 제공해야 하는 “업권별 정보제공 API” 목록 :
<base path #1> / <version> / bank / <resource> <base path #2> / <version> / card / <resource>
* C기관(은행업)의 기관코드와 C기관(카드업)의 기관코드는 상이
* C기관의 <base path #1>과 <base path #2>는 동일해도 무관 ※ 예시1~3 모두, 인증 API 및 지원 API는 <base path> 별로 제공
   ᄋ (HTTP 헤더) 표준 API에 지원되는 HTTP 헤더를 다음과 같이 지정
- (Content-Type) 데이터(Body)의 포맷을 지정하기 위한 헤더 값으로 필요에 따라 application/x-www-form-urlencoded, 또는 application/json 으로 지정 (상세 내용은 제5장~제7장 API 명세 참조)
- (x-api-type) 정보제공 API 호출 시 정보주체의 개입 없이 마이데이터 사업자가 정기적 전송을 위해 호출한 API인지 또는 정보주체가 직접 개입*하여 비정기적 전송으로 호출한 API인지 여부를 구분하기 위한 헤더 값 (상세 내용은 3.3 참조)
* 예시 : 사용자가 마이데이터사업자 앱에 접속하여 “조회” 등 버튼 클릭 시 API 호출
 금융보안원 www.fsec.or.kr - 8 -

   금융분야 마이데이터 표준API 규격
    분류
조건
헤더 값 지정
요청 (Request)
정기적 전송 시
“x-api-type: scheduled”
  비정기적 전송 시
전송요구 직후
“x-api-type: user-consent”
로그인 또는 새로고침 시
“x-api-type: user-refresh”
특정자산 거래내역 조회 시
“x-api-type: user-search”
응답 (Response)
해당없음
-
                  - (x-api-tran-id) API를 송수신한 기관 간 거래추적이 필요(민원대응, 장 애처리 등)한 경우 거래를 식별하기 위한 거래고유번호로 HTTP 요청 및 응답 헤더에 값을 설정 (첨부14 참조)
ᄋ (응답코드 및 메시지) 기본적인 HTTP 응답코드는 정상응답 여부 및 주요한 에러를 범주로 처리하는 등 각 API 처리상황의 상세한 판단이 불가능하므로, 세부 응답코드 및 메시지를 사용함 ([첨부1] 참조)
- (OAuth 2.0 관련 API) RFC 6749 및 7009 국제표준 준용
- (그 외) 세부 응답코드와 메시지는 응답 본문 내 ‘rsp_code’, ‘rsp_msg’ 필드에
포함하여 반환
ᄋ (페이지네이션, 부분범위 조회) 정보의 목록을 반환하는 API(거래내역 조회 등)에는 부분범위 조회를 위한 ‘Cursor-Based Pagination’ 기법을 적용하여 데이터수신자의 처리 효율성과 정보제공자의 부 하를 경감함
 금융보안원 www.fsec.or.kr - 9 -

   금융분야 마이데이터 표준API 규격
         구분
이름
limit
next_page*
next_page*
타입 (길이)
N (3)
aNS (1000)
aNS (1000)
설명
   요청
응답 (JSON 응답)
[부분범위 조회 요청/응답 파라미터 규격]
기준개체 이후 반환될 개체의 개수 (최대 500까지 설정 가능)
다음 페이지 요청을 위한 기준개체 (설정 시 해당 개체를 포함한 limit 개 반환)
다음 페이지 요청을 위한 기준개체 (다음 페이지 존재하지 않는 경우 미회신)
   * 기준개체 식별자의 생성규칙은 정보제공자가 자율적으로 정함. 다만, 특수문자를 포함 하여 기준개체 식별자를 생성하는 정보제공자의 경우 URL safe한 방식(URL encoding, URL Safe BASE64 등)을 적용하여 생성 및 응답
- 다만, limit을 최대 500까지 설정하더라도 정보제공자의 환경*에 따라
페이지 당 반환되는 개체의 개수는 유동적 (0 ≤ 반환되는 개체의 개수 ≤ limit)
* 중계기관과 이용기관 간 전용선 회선용량 한계 등으로 인해 페이지 당 2~30개만 회신 가능한 경우 존재
** 반환되는 개체의 개수가 줄어드는 만큼 마이데이터사업자가 처리해야 하는 API 호출횟 수는 증가 → 추후 신용정보원이 수수료 기준 마련 시, API 호출횟수, API 처리량 등을 종합적으로 고려 예정
 금융보안원 www.fsec.or.kr - 10 -

   금융분야 마이데이터 표준API 규격
  ᄋ (부하개선을 위한 조회 Timestamp) 마이데이터사업자가 정보를 수집한 이후 동일한 정보에 대해 정기전송 요청 시 정보제공자의 정보가 수 정사항이 없을 경우 조회 Timestamp를 이용하여 전송을 최소화
- 정보제공자는 조회 Timestamp 기능을 의무적으로 구현할 필요없음 (선택사항)
  Timestamp를 이용한 부하개선 (예시)
     1. (정보수신자) 최초로 API 호출 (본 예시는 "계좌 기본정보 조회"(가칭) API를 호출 시 시나리오)
- ST(조회 timestamp)=0
2. (정보제공자) ST와 MT(가장 최근에 DB갱신된 시간)를 비교
- ST(0) < MT(40)이므로, DB를 조회하여 정보를 정보수신자에게 전달
- 이때 정보수신자가 조회한 시간(current time)을 함께 회신 (예시에서는 현재시
각을 100이라 가정)
3. (정보수신자) 전달받은 정보 및 조회기준시점(100)을 DB에 반영
4. (정보수신자) 정기적 전송을 위해 정보수신자가 동일한 API를 호출
- DB에 저장되어 있는 ST(100)을 함께 정보제공자에게 전달
5. (정보제공자) 그 사이에 정보제공자의 DB가 변경된 적이 없으므로(MT=40),
UP_TO_DATE 회신
6. (정보제공자) 계좌 기본정보가 갱신될 경우, MT도 UPDATE 시점의 시간(예시에서
    금융보안원 www.fsec.or.kr - 11 -

   금융분야 마이데이터 표준API 규격
         는 UPDATE 시점의 현재시각을 150이라 가정)도 갱신
7. (정보수신자) 정기적 전송을 위해 정보수신자가 동일한 API를 호출
- DB에 저장되어 있는 ST(100)을 함께 정보제공자에게 전달
8. (정보제공자) ST와 MT(가장 최근에 DB갱신된 시간=150)를 비교
- ST(100) < MT(150)이므로, DB를 조회하여 정보를 정보수신자에게 전달
- 이때 정보수신자가 조회한 시간(current time)을 함께 회신 (예시에서는 현재시
각을 200이라 가정)
9. (정보수신자) 전달받은 정보 및 조회기준시점(200)을 DB에 반영
  - 복수 개의 데이터(List)를 회신하는 API(예:은행-001 등)의 경우, 정보 제공자는 List 목록수, 목록내용이 변경되는 경우에도 MT값을 갱신할 필요
* 예시 : 특정 정보주체의 계좌가 3개 있었는데, 그 중 1개가 해지되는 경우 List 목록수 가 3개에서 2개로 줄어들게 되며, 해당 계좌가 해지되는 시점의 현재 시각을 MT값으 로 설정이 필요 (목록이 추가 또는 삭제되는 경우도 DB 변경으로 처리하지 않을 경우 ‘신규’ 또는 ‘해지’ 자산을 조회할 방법 부재)
- 페이지네이션과 Timestamp를 동시에 처리하는 API(예:은행-001 등)의 경우, 최초 호출한 API(첫 번째 페이지 조회) 시에만 search_timestamp를 요청 및 응답하고, 이후 호출하는 API들은 search_timestamp 요청 및 응답 불필요(next_page로 처리)
* 예시 : 최초 은행-001 API 호출 시 마이데이터사업자가 ST=100으로 요청하고 정보제 공자가 ST=200으로 회신한 경우, 다음 페이지(next_page) 요청을 위해 은행-001 API 호출 시에는 search_timestamp를 요청 및 응답에서 제외
 금융보안원 www.fsec.or.kr - 12 -
        """

    def query(self, prompt: str):
        document = self._retrieve_document()
        response = self.chain.invoke({"document": document, "query":prompt})

        return response.content