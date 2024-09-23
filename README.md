# sj-kb-md-agent

## Quick Demo

### Demo URL
[TBU]

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

### Run the Application

```bash
streamlit run streamlit_app.py
```

## Test

시스템의 최소한의 품질 유지와 지속적인 성능 개선을 위해선 적절한 테스트가 요구됩니다.
그러나, LLM 모델의 특성상 일반적인 rull-based 기반의 테스트 방법론을 적용하기 어렵습니다.

이를 해결하기 위해, [RAGAS](https://docs.ragas.io/en/stable/index.html) 프레임워크를 활용하여 테스트 환경을 구성하였습니다. RAGAS는 LLM을 에이전트로 활용하여 RAG 파이프라인의 성능을 평가하고, 지속적인 학습을 통해 성능을 개선할 수 있도록 돕는 평가 프레임워크입니다.

### Metric

시스템의 성능을 평가하기 위한 메트릭으로, [Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness.html) 를 활용합니다. 

이는, 생성된 답변과 정답의 유사성을 평가하는 메트릭으로, 0 ~ 1 사이의 점수로 표현됩니다. 높은 점수는 생성된 답변과 정답의 유사성이 높음을 의미하며, 더 나은 정확성을 나타냅니다.

### Golden Dataset

요구사항을 바탕으로 1차적으로 Claude 의 프로젝트 기능을 활용해 답변을 생성한 후, 문서를 살펴보며 적절한 답변예시를 작성하였습니다. 

자세한 내용은 [Golden Dataset](./test/golden_dataset.yaml) 를 확인해주세요.

### Run Test

```bash  
python src/rag_test.py
```

Example output:
```bash
Threshold for Answer Correctness: 0.75

Test Case 1:
Question: API 스펙 중 aNS는 어떤 것을 뜻하나요?
Expected Answer: API 스펙에서 aNS는 다음을 의미합니다: ~~
Generated Answer: API 스펙에서 aNS는 다음을 의미합니다: ~~
Score (Answer Correctness): 0.9
Result: PASSED

Test Case 2:
Question:  x-api-tran-id에 대해 알려주세요.
Expected Answer: x-api-tran-id는 API 요청에 대한 고유한 식별자로, 요청과 응답 간의 관계를 추적하는 데 사용됩니다.
Generated Answer: 죄송합니다. 관련된 문서를 찾을 수 없습니다.
Score (Answer Correctness): 0.331
Result: FAILED
```
