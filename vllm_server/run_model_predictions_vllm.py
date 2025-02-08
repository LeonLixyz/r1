import os
import json
import argparse
import asyncio
import re
import httpx
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

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

### STREAMING vLLM REQUEST ###
async def vllm_attempt_question(question, client, args):
    """
    For a given question, send a streaming POST request to the vLLM /chat/completions endpoint.
    Accumulate the final generated text and reasoning content.
    """
    messages = format_message(question)
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "stream": True
    }
    final_content = ""
    final_reasoning = ""
    try:
        async with client.stream("POST", "/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                # Remove "data:" prefix if present.
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                # Now check if the line indicates the end of the stream.
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except Exception as e:
                    print(f"[{question['id']}] Could not parse chunk: {line}\nError: {e}")
                    continue
                if "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        final_content += delta["content"]
                    if "reasoning_content" in delta:
                        final_reasoning += delta["reasoning_content"]
    except httpx.ReadError as e:
        print(f"[{question['id']}] ReadError encountered: {e}")
        return None
    parsed = parse_response(final_content)
    complete_response = {
        "id": question["id"],
        "question": question["question"],
        "reasoning": final_reasoning,
        "raw_response": final_content,
        "parsed": parsed
    }
    return complete_response

### NON-STREAMING vLLM REQUEST ###
async def vllm_attempt_question_nonstream(question, client, args):
    """
    For a given question, send a non-streaming POST request to the vLLM /chat/completions endpoint.
    """
    messages = format_message(question)
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "stream": False
    }
    try:
        response = await client.post("/chat/completions", json=payload)
        response_json = response.json()
        final_content = response_json["choices"][0]["message"]["content"]
        parsed = parse_response(final_content)
        complete_response = {
            "id": question["id"],
            "question": question["question"],
            "reasoning": "",  # No reasoning content in non-streaming mode
            "raw_response": final_content,
            "parsed": parsed
        }
        return complete_response
    except httpx.RequestError as e:
        print(f"[{question['id']}] RequestError encountered: {e}")
        return None

### SAVE RESULT ###
async def save_single_result(result):
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
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            existing_ids.add(filename.replace('.json', ''))
    return existing_ids

### ASYNC PIPELINE ###
async def attempt_all(questions, args):
    async with httpx.AsyncClient(base_url=args.http_url, timeout=300, http2=True) as client:
        semaphore = asyncio.Semaphore(args.num_workers)
        async def bound_func(question):
            async with semaphore:
                print(f"Starting question {question['id']}")
                if args.stream:
                    result = await vllm_attempt_question(question, client, args)
                else:
                    result = await vllm_attempt_question_nonstream(question, client, args)
                print(f"Finished question {question['id']}")
                await save_single_result(result)
                return result
        tasks = [bound_func(q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)
        return [r for r in results if r is not None]

### MAIN FUNCTION ###
def main(args):
    if args.test_mode:
        questions = get_test_questions()
    else:
        dataset = load_dataset(args.dataset, split="test").to_dict()
        questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
        # Optionally filter out multimodal questions.
        questions = [q for q in questions if not q['image']]
        print("Total questions:", len(questions))
    existing_ids = get_existing_results()
    questions = [q for q in questions if q['id'] not in existing_ids]
    if not questions:
        print("All questions have already been processed!")
        return
    print(f"Processing {len(questions)} new questions...")
    asyncio.run(attempt_all(questions, args))

### ARGUMENT PARSING AND ENTRY POINT ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name (for non-test mode)")
    parser.add_argument("--model", type=str, help="vLLM model endpoint name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--test_mode", action="store_true", help="Use test questions instead of a dataset")
    parser.add_argument("--http_url", type=str, default="http://localhost:8000/v1", help="vLLM API endpoint")
    parser.add_argument("--stream", action="store_true", default=True, help="Use streaming API (default: True)")
    args = parser.parse_args()
    main(args)
