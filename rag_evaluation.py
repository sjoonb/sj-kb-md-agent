import yaml
from colorama import Fore, init
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from src.rag.interfaces import IRAG
from src.rag.llm_retriever_rag_impl import LlmRetrieverRAGImpl

# Initialize colorama
init(autoreset=True)

ANSWER_SIMILARITY_THRESHOLD = 0.75

def load_test_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return [{'question': item['Q'], 'ground_truth': item['A']} for item in data]

def create_evaluation_chain():
    evaluation_prompt = PromptTemplate(
        input_variables=["ground_truth", "generated_answer"],
        template="""
        예상 답변과 생성된 답변이 주어졌습니다. 두 답변 간의 연관성을 평가해주세요.

        예상 답변: {ground_truth}
        생성된 답변: {generated_answer}

        다음 기준에 따라 평가해주세요:
        1. 내용의 유사성: 생성된 답변이 예상 답변의 핵심 정보를 포함하고 있는가?
        2. 의미의 일치성: 생성된 답변이 예상 답변과 같은 의미를 전달하는가?
        3. 정보의 정확성: 생성된 답변의 정보가 예상 답변의 정보와 일치하는가?

        0.00에서 1.00 사이의 점수를 제공하고, 그 이유를 간단히 설명해주세요. 
        점수와 설명을 다음 형식으로 반환해주세요:

        점수: [0.00에서 1.00 사이의 숫자]
        이유: [점수에 대한 간단한 설명]
        """
    )
    
    return (
        {"question": RunnablePassthrough(), "ground_truth": RunnablePassthrough(), "generated_answer": RunnablePassthrough()}
        | evaluation_prompt
        | ChatOpenAI(temperature=0, model="gpt-4o")
    )

def parse_evaluation_result(result):
    lines = result.content.split('\n')
    score = float(lines[0].split(':')[1].strip())
    reason = lines[1].split(':')[1].strip()
    return score, reason

def evaluate_results(results, threshold):
    print("\n--- Test Results ---\n")
    print(f"Threshold for Answer Semantic Similarity: " + Fore.GREEN + f"{threshold:.2f}\n")
    
    for idx, result in enumerate(results):
        print(f"Test Case {idx + 1}:")
        print(f"Question: {result['question']}")
        print(f"Expected Answer: {result['ground_truth']}")
        print(f"Generated Answer: {result['generated_answer']}")
        
        score_color = Fore.GREEN if result['score'] >= threshold else Fore.RED
        print(f"Score (Answer Semantic Similarity): {score_color}{result['score']:.2f}{Fore.RESET}")
        print(f"Reason: {result['reason']}")
        
        result_text = "PASSED" if result['score'] >= threshold else "FAILED"
        result_color = Fore.GREEN if result_text == "PASSED" else Fore.RED
        print(f"Result: {result_color}{result_text}{Fore.RESET}\n")

def test_rag(rag: IRAG):
    print("-" * 50)
    print(f"Testing {rag.__class__.__name__}...\n")
    
    evaluation_chain = create_evaluation_chain()
    data_samples = load_test_dataset("golden_dataset.yaml")
    results = []

    for sample in data_samples:
        generated_answer = rag.query(sample['question'])
        evaluation_result = evaluation_chain.invoke({
            "question": sample['question'],
            "ground_truth": sample['ground_truth'],
            "generated_answer": generated_answer
        })
        score, reason = parse_evaluation_result(evaluation_result)
        
        results.append({
            "question": sample['question'],
            "ground_truth": sample['ground_truth'],
            "generated_answer": generated_answer,
            "score": score,
            "reason": reason
        })

    evaluate_results(results, ANSWER_SIMILARITY_THRESHOLD)

if __name__ == "__main__":
    rag_implementations = [LlmRetrieverRAGImpl()]
    for rag_impl in rag_implementations:
        test_rag(rag=rag_impl)