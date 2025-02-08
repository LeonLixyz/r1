import os
import json
import argparse
import re
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import multiprocessing
from functools import partial

#### PROMPT TEMPLATES ####
PROMPT_EXACT_ANSWER = "You will be given a question and a response format. Please output the answer to the question following the format.\n\nResponse format:\nExplanation: {{your explanation for your final answer}}\nExact Answer: {{your succinct, final answer}}\nConfidence: {{your confidence score between 0% and 100% for your answer}}\n\nQuestion:\n{question}"

PROMPT_MC = "You will be given a question and a response format. Please output the answer to the question following the format.\n\nResponse format:\nExplanation: {{your explanation for your answer choice}}\nAnswer: {{your chosen answer}}\nConfidence: {{your confidence score between 0% and 100% for your answer}}\n\nQuestion:\n{question}"

### MESSAGE FORMATTER ###
def format_message(question):
    answer_type = question['answer_type']
    prompt_template = PROMPT_EXACT_ANSWER if answer_type == 'exact_match' else PROMPT_MC
    question_text = question['question']
    input_prompt = prompt_template.format(question=question_text)
    return [{"role": "user", "content": input_prompt}]

### RESPONSE PARSER ###
def parse_response(response_text):
    explanation = ""
    answer = ""
    confidence = ""
    explanation_match = re.search(r'Explanation:\s*(.*?)(?:\n|$)', response_text)
    answer_match = re.search(r'(?:Exact Answer|Answer):\s*(.*?)(?:\n|$)', response_text)
    confidence_match = re.search(r'Confidence:\s*(\d+)%', response_text)
    if explanation_match:
        explanation = explanation_match.group(1).strip()
    if answer_match:
        answer = answer_match.group(1).strip()
    if confidence_match:
        confidence = int(confidence_match.group(1))
    return {"explanation": explanation, "answer": answer, "confidence": confidence}

### SAVE RESULT ###
def save_single_result(result):
    if result is None:
        return
    os.makedirs('results', exist_ok=True)
    output_file = f'results/{result["id"]}.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

### DATASET LOADING ###
def get_test_questions():
    """Creates toy test questions for pipeline testing."""
    return [
        {
            "id": "test_q1",
            "question": "Let $N = 36036$. Find the number of primitive Dirichlet characters of conductor $N$ and order $6$.",
            "answer_type": "exact_match",
            "image": ""
        }
    ]

def get_existing_results():
    """Get a set of question IDs that have already been processed."""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        return set()
    existing_ids = set()
    for filename in os.listdir(results_dir) :
        if filename.endswith('.json'):
            existing_ids.add(filename.replace('.json', ''))
    return existing_ids

def process_question(question, args):
    """Process a single question using the OpenAI client"""
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.http_url,
    )
    
    messages = format_message(question)
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=65536,
        )
        
        message = response.choices[0].message
        content = message.content
        reasoning = message.reasoning_content
        parsed = parse_response(content)

        result = {
            "id": question["id"],
            "question": question["question"],
            "reasoning": reasoning,
            "raw_response": content,
            "parsed": parsed
        }
        save_single_result(result)
        return result
    except Exception as e:
        print(f"Error processing question {question['id']}: {e}")
        print(f"Question content: {question}")
        print(f"Formatted messages: {messages}")
        return None

def main(args):
    if args.test_mode:
        questions = get_test_questions()
    else:
        dataset = load_dataset(args.dataset, split="test").to_dict()
        questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
        questions = [q for q in questions if not q.get('image')]
        print("Total questions in dataset:", len(questions))

    existing_ids = get_existing_results()
    questions = [q for q in questions if q['id'] not in existing_ids]
    if not questions:
        print("All questions have already been processed!")
        return
    print(f"Processing {len(questions)} new questions...")

    # Process all questions at once
    process_func = partial(process_question, args=args)
    
    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.imap_unordered(process_func, questions),
            total=len(questions),
            desc="Completed requests"
        ))

    # Filter out None results from failed requests
    results = [r for r in results if r is not None]
    return results

### ARGUMENT PARSING AND ENTRY POINT ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name (for non-test mode)")
    parser.add_argument("--model", type=str, help="vLLM model endpoint name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--test_mode", action="store_true", help="Use test questions instead of a dataset")
    parser.add_argument("--http_url", type=str, default="http://localhost:8000/v1", help="vLLM API endpoint")
    parser.add_argument("--stream", action="store_true", default=True, help="(Ignored) Use streaming API (default: True)")
    args = parser.parse_args()
    main(args)
