import os
import json
import argparse
import asyncio
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from together import Together, AsyncTogether
import re  


#### PROMPT TEMPLATES ####

PROMPT_EXACT_ANSWER = "You will be given a question and a response format. Please output the answer to the question following the format.\n\nResponse format:\nExplanation: {{your explanation for your final answer}}\nExact Answer: {{your succinct, final answer}}\nConfidence: {{your confidence score between 0% and 100% for your answer}}\n\nQuestion:\n{question}"

PROMPT_MC = "You will be given a question and a response format. Please output the answer to the question following the format.\n\nResponse format:\nExplanation: {{your explanation for your answer choice}}\nAnswer: {{your chosen answer}}\nConfidence: {{your confidence score between 0% and 100% for your answer}}\n\nQuestion:\n{question}"

### SET UP API KEYS ####

client = Together(api_key='9cb3e0c48d9412890a81871625ce2d9e68b422932268935a3158a018dba8232e')

def extract_thinking_parts(text):
    """
    Extracts both thinking and non-thinking parts from text.
    Returns a tuple of (non_thinking_text, thinking_text).
    """
    # Extract thinking parts
    thinking_parts = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    
    # Extract non-thinking parts
    non_thinking_parts = re.split(r'<think>.*?</think>', text, flags=re.DOTALL)
    
    # Join the parts
    non_thinking_text = ''.join(non_thinking_parts).strip()
    thinking_text = '\n'.join(thinking_parts).strip()
    
    return non_thinking_text, thinking_text

def format_message(question):
    answer_type = question['answer_type']
    prompt_template = PROMPT_EXACT_ANSWER if answer_type == 'exact_match' else PROMPT_MC
    question_text = question['question']

    text_content = dict(type="text", text=question_text)
    if question['image']: # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question['image']))
        content = [text_content, image_content]
    else:
        content = [text_content]

    input_prompt = prompt_template.format(question=question_text)
    messages = [
        {"role": "user", "content": input_prompt}, 
    ]
    return messages

def parse_response(response_text):
    """Parse the response text into components"""
    explanation = ""
    answer = ""
    confidence = ""
    
    # Extract components using regex
    explanation_match = re.search(r'Explanation:\s*(.*?)(?:\n|$)', response_text)
    answer_match = re.search(r'(?:Exact Answer|Answer):\s*(.*?)(?:\n|$)', response_text)
    confidence_match = re.search(r'Confidence:\s*(\d+)%', response_text)
    
    if explanation_match:
        explanation = explanation_match.group(1).strip()
    if answer_match:
        answer = answer_match.group(1).strip()
    if confidence_match:
        confidence = int(confidence_match.group(1))
    
    return {
        "explanation": explanation,
        "answer": answer,
        "confidence": confidence
    }

async def attempt_question(question, async_client):
    messages = format_message(question)
    # Calculate original prompt tokens
    initial_response = await async_client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=0  # Just to get prompt tokens
    )
    original_prompt_tokens = initial_response.usage.prompt_tokens
    
    async def try_generation(current_messages, attempt_count=1, last_response=None, accumulated_content=""):
        try:
            if attempt_count > 5:
                print(f"[{question['id']}] Maximum retries reached. Failed all attempts.")
                # Don't save partial results for failed attempts
                return question["id"], None, None, None, None
            
            max_tokens = None if args.max_tokens == -1 else args.max_tokens
            print(f"[{question['id']}] Attempt {attempt_count}: Sending API request")

            response = await async_client.chat.completions.create(
                model=args.model,
                messages=current_messages,
                temperature=args.temperature,
                max_tokens=max_tokens,
            )
            
            # Store response for potential failure case
            last_response = response
            
            content = response.choices[0].message.content
            
            tokens = {
                "completion_tokens": response.usage.total_tokens - original_prompt_tokens,
                "prompt_tokens": original_prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Get previous assistant response if it exists
            previous_content = ""
            for msg in current_messages:
                if msg["role"] == "assistant":
                    previous_content = msg["content"]
            
            # Concatenate with previous content if it exists
            full_content = previous_content + content if previous_content else content
            accumulated_content = full_content  # Update accumulated content
            
            # Try to parse the response
            non_thinking, thinking = extract_thinking_parts(full_content)
            parsed_response = parse_response(non_thinking)
            
            # If parsing fails, retry with concatenated content
            if not all([parsed_response["explanation"], parsed_response["answer"], parsed_response["confidence"]]):
                new_messages = [msg for msg in current_messages if msg["role"] == "user"]
                new_messages.append({"role": "assistant", "content": full_content})
                return await try_generation(new_messages, attempt_count + 1, response, accumulated_content)
                        
            complete_response = {
                "id": question["id"],
                "question": question["question"],
                "raw_response": full_content,
                "parsed": parsed_response,
                "thinking": thinking,
                "usage": tokens,
                "generation_attempts": attempt_count
            }
            
            return question["id"], parsed_response, thinking, tokens, complete_response
            
        except Exception as e:
            print(f"[{question['id']}]  Error in try_generation: {e}")
            if attempt_count < 5:
                return await try_generation(current_messages, attempt_count + 1, last_response, accumulated_content)
            return None
    
    return await try_generation(messages)

async def save_single_result(result):
    """Save a single result to a JSON file"""
    if result is None:
        return
        
    question_id, parsed_response, thinking, tokens, complete_response = result
    
    # Don't save result if it's a failed attempt
    if complete_response is None:
        # Create output directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Only record in failed_attempts.json
        failed_file = 'results/failed_attempts.json'
        failed_ids = []
        if os.path.exists(failed_file):
            with open(failed_file, 'r') as f:
                failed_ids = json.load(f)
        if question_id not in failed_ids:
            failed_ids.append(question_id)
            with open(failed_file, 'w') as f:
                json.dump(failed_ids, f, indent=2)
        return
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save successful result to a JSON file named with the question ID
    output_file = f'results/{question_id}.json'
    with open(output_file, 'w') as f:
        json.dump(complete_response, f, indent=2)

async def attempt_all(questions):
    async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
    
    async def bound_func(question):
        try:
            async with semaphore:
                print(f"Starting worker for question {question['id']}")
                result = await attempt_question(question, async_client)
                print(f"Finished worker for question {question['id']}")
                if result:
                    await save_single_result(result)
                return result
        except Exception as e:
            print(f"Error processing question {question['id']}: {e}")
            return None
            
    semaphore = asyncio.Semaphore(args.num_workers)
    tasks = [bound_func(q) for q in questions]
    # Remove return_exceptions from tqdm_asyncio.gather
    results = await tqdm_asyncio.gather(*tasks)
    return [r for r in results if r is not None]

    
def get_test_questions():
    """Creates toy test questions for pipeline testing"""
    return [
        {
            "id": "test_q1",
            "question": "Let $N = 36036$. Find the number of primitive Dirichlet characters of conductor $N$ and order $6$.",
            "answer_type": "exact_match",
            "image": ""  
        }
    ]

def get_existing_results():
    """Get a set of question IDs that have already been processed"""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        return set()
    
    existing_ids = set()
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            existing_ids.add(filename.replace('.json', ''))
    return existing_ids

def main(args):
    assert args.num_workers > 1, "num_workers must be 2 or greater"
    
    if args.test_mode:
        questions = get_test_questions()
    else:
        dataset = load_dataset(args.dataset, split="test").to_dict() 
        questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
        questions = [q for q in questions if not q['image']]
        print("total questions: ", len(questions))
    

    existing_ids = get_existing_results()
    questions = [q for q in questions if q['id'] not in existing_ids]
    
    if not questions:
        print("All questions have already been processed!")
        return
    
    print(f"Processing {len(questions)} new questions...")
    asyncio.run(attempt_all(questions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="HLE HF Dataset")
    parser.add_argument("--model", type=str, help="Model Endpoint Name")
    parser.add_argument("--max_tokens", type=int, default=None, help="Limit completion tokens. Recommended to avoid model collapse.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    parser.add_argument("--num_workers", type=int, default=10, help="Async semaphore size. This depends on your rate limit.")
    parser.add_argument("--test_mode", action="store_true", help="Use test questions instead of loading dataset")
    args = parser.parse_args()
    main(args)
