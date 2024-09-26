import yaml
import pandas as pd
from datasets import Dataset
from llama_index.llms.openai import OpenAI
from ragas.integrations.llama_index import evaluate
from ragas.metrics import answer_correctness
from colorama import Fore, init # type: ignore
from rag import RAGSystem

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

# Function to get prompts from the query engine
def get_prompts(query_engine):
    return query_engine.get_prompts()

# Print current prompts
def print_prompt_dict(prompts_dict):
    print("[TEST] Current prompts:")
    for k, p in prompts_dict.items():
        prompt_text = f"Prompt Key: {k}\nText:"
        print(prompt_text)
        print(p.get_template())
        print("")

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

def test_rag_system():
    """
    Test the RAG system by generating a response and evaluating it.
    """
    # Initialize RAG System
    rag_system = RAGSystem(input_dir="./data")
    rag_system.initialize()

    # Get and print prompts
    prompts = get_prompts(rag_system.query_engine)
    print_prompt_dict(prompts)

    metrics = [answer_correctness]
    evaluator_llm = OpenAI(model="gpt-4o-mini")

    # Load the test dataset from the YAML file
    data_samples = load_test_dataset("test/golden_dataset.yaml")
    
    # Create a dataset object from the loaded test data
    dataset = Dataset.from_dict(data_samples)

    # Evaluate the test data using the RAG system
    result = evaluate(
        query_engine=rag_system.query_engine,
        metrics=metrics,
        dataset=dataset,
        llm=evaluator_llm,
    )

    # Convert results to a DataFrame and evaluate
    evaluate_results(result.to_pandas(), ANSWER_CORRECTNESS_THRESHOLD)

if __name__ == "__main__":
    test_rag_system()