import time
import yaml
from colorama import Fore, init  # type: ignore
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_correctness

from src.rag.interfaces import IRAG
from src.rag.llamaindex_rag_impl import LlamaIndexRAGImpl
from src.rag.llm_retriever_rag_impl import LlmRetreivalRAGImpl

# Initialize colorama
init(autoreset=True)

# Define the correctness threshold
ANSWER_CORRECTNESS_THRESHOLD = 0.75

# Function to load test dataset from a YAML file
def load_test_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    data_samples = {
        'question': [item['Q'] for item in data],
        'ground_truth': [item['A'] for item in data]
    }
    return data_samples

def evaluate_results(df, threshold):
    """
    Evaluate test results against the threshold and print the outcome with color.
    """
    print("\n--- Test Results ---\n")
    print(f"Threshold for Answer Correctness: " + Fore.GREEN + f"{threshold:.2f}\n")
    for idx, row in df.iterrows():
        print(f"Test Case {idx + 1}:")
        print(f"Question: {row['question']}")
        print(f"Expected Answer: {row['ground_truth']}")
        print(f"Generated Answer: {row['answer']}")
        
        # Color the score based on the threshold
        score_color = Fore.GREEN if row['answer_correctness'] >= threshold else Fore.RED
        print(f"Score (Answer Correctness): {score_color}{row['answer_correctness']:.3f}{Fore.RESET}")

        # Print result status
        result_text = "PASSED" if row['answer_correctness'] >= threshold else "FAILED"
        result_color = Fore.GREEN if result_text == "PASSED" else Fore.RED
        print(f"Result: {result_color}{result_text}{Fore.RESET}\n")

def test_rag(rag: IRAG):
    """
    Test the LlamaIndex-based RAG system by generating a response and evaluating it.
    """
    # Initialize LlamaIndex RAG System
    rag.initialize()

    metrics = [answer_correctness]
    critic_llm = ChatOpenAI(model="gpt-4o")

    # Load the test dataset from the YAML file
    data_samples = load_test_dataset("golden_dataset.yaml")
    data_samples['response'] = [rag.query(q) for q in data_samples['question']]

    dataset = Dataset.from_dict(data_samples)

    result = evaluate(
        metrics=metrics,
        dataset=dataset,
        llm=critic_llm,
    )

    # Convert results to a DataFrame and evaluate
    evaluate_results(result.to_pandas(), ANSWER_CORRECTNESS_THRESHOLD)

if __name__ == "__main__":
    # test_rag(rag=LlamaIndexRAGImpl())
    test_rag(rag=LlmRetreivalRAGImpl())