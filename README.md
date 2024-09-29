# sj-kb-md-agent

## Quick Demo

### Demo URL
https://sjoonb-kakaobank-ai.streamlit.app/

## Setup and Installation

### Requirements
- Python 3.8+

### Create and Activate Virtual Environment

1. venv를 생성합니다:
   ```bash
   python3 -m venv env
   ```

2. venv를 활성화합니다:

   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```

3. dependencies를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```
### OpenAI API Key 설정

RAG 시스템의 Generation 단계와, 시스템 성능 평가 과정에서 OpenAI의 모델을 활용합니다. 이에 따라 환경변수에 다음과 같이 키를 저장해야 합니다.

```bash
EXPORT OPENAI_API_KEY=YOUR_OPEN_AI_API_KEY
```

### Run the Application

```bash
streamlit run streamlit_app.py
```

## Evaluation

시스템의 최소한의 품질 유지와 지속적인 성능 개선을 위해선 적절한 성능 검증이 요구됩니다.
그러나, LLM 모델의 특성상 일반적인 rull-based 기반의 테스트 방법론을 적용하기 어렵습니다.

이를 해결하기 위해, [RAGAS](https://docs.ragas.io/en/stable/index.html) 프레임워크를 활용하여 테스트 환경을 구성하였습니다. RAGAS는 LLM을 에이전트로 활용하여 RAG 파이프라인의 성능을 평가하고, 지속적인 학습을 통해 성능을 개선할 수 있도록 돕는 평가 프레임워크입니다.

### Metric

시스템의 성능을 평가하기 위한 메트릭으로, [Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness.html) 를 활용합니다. 

이는, 생성된 답변과 정답의 유사성을 평가하는 메트릭으로, 0 ~ 1 사이의 점수로 표현됩니다. 높은 점수는 생성된 답변과 정답의 유사성이 높음을 의미하며, 더 나은 정확성을 나타냅니다.

### Golden Dataset

요구사항을 바탕으로 1차적으로 Claude 의 프로젝트 기능을 활용해 답변을 생성한 후, 문서를 살펴보며 적절한 답변예시를 작성하였습니다. 

자세한 데이터 내용은 [Golden Dataset](./test/golden_dataset.yaml) 를 확인해주세요.

### Run Evaluation

```bash  
python rag_evaluation.py
```

Example output:
```bash
Threshold for Answer Correctness: 0.75

Test Case 1:
Question: x-api-tran-id에 대해 알려주세요.
Expected Answer: x-api-tran-id는 API를 송수신한 기관 간 거래추적이 필요한 경우(예: 민원대응, 장애처리 등) 거래를 식별하기 위한 거래고유번호입니다. 이 값은 HTTP 요청 및 응답 헤더에 설정됩니다.

Generated Answer: "x-api-tran-id"는 API를 송수신한 기관 간 거래추적이 필요(민원대응, 장애처리 등)한 경우 거래를 식별하기 위한 거래고유번호로 HTTP 요청 및 응답 헤더에 값을 설정하는 것입니다.
Score (Answer Correctness): 1.00
Reason: 생성된 답변은 예상 답변의 핵심 정보를 모두 포함하고 있으며, 의미도 정확히 일치합니다. 또한 정보의 정확성 측면에서도 예상 답변과 동일한 내용을 전달하고 있습니다. 따라서 모든 평가 기준을 충족하므로 최고 점수를 부여할 수 있습니다.
Result: PASSED

Test Case 2:
Question: API 스펙 중 aNS는 어떤 것을 뜻하나요?
Expected Answer: aNS는 알파벳, 숫자 및 특수기호 조합을 의미합니다. 

Generated Answer: API 스펙 중 aNS는 "다음 페이지 요청을 위한 기준개체"를 의미합니다. 요청 파라미터 규격에서 "next_page*"의 타입으로 aNS (1000)으로 정의되어 있으며, 응답에서도 "next_page*"가 aNS (1000)으로 나타납니다.
Score (Answer Correctness): 0.00
Reason: 생성된 답변은 예상 답변과 전혀 다른 정보를 제공하고 있습니다. 예상 답변에서는 aNS가 "알파벳, 숫자 및 특수기호 조합"을 의미한다고 설명하고 있지만, 생성된 답변에서는 aNS가 "다음 페이지 요청을 위한 기준개체"를 의미한다고 설명하고 있습니다. 따라서 내용의 유사성, 의미의 일치성, 정보의 정확성 모두에서 일치하지 않습니다.
Result: FAILED
```
