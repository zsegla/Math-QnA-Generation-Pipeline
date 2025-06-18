#src/utils/gemini_api_utils.py
import json
import time
import google.generativeai as genai
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions
import re

from src.config import settings
from src.utils.gemini_api import model

# Global variable for proactive rate limiting, initialized at the module level
_last_api_call_time = 0

def call_gemini_api_with_retries(
    model_instance: genai.GenerativeModel,
    prompt_content: List[Any],
    max_retries: int = 5,
    initial_delay: int = 60,
    min_retry_delay: int = 15,
    call_description: str = "Gemini API call"
) -> Tuple[Any, int, int, float]: # Returns (response, prompt_tokens, candidate_tokens, duration)
    """
    Wrapper to call Gemini API with proactive rate limiting, exponential back-off,
    and retry for rate limits. Ensures API calls do not exceed the specified RPM.
    Also tracks time and token usage for each call.
    """
    global _last_api_call_time

    PROACTIVE_DELAY_SECONDS = 4 # Target: 15 RPM = 60 seconds / 15 requests = 4 seconds/request

    # --- Proactive Rate Limiting Logic ---
    current_time = time.time()
    time_since_last_call = current_time - _last_api_call_time

    if time_since_last_call < PROACTIVE_DELAY_SECONDS:
        wait_for = PROACTIVE_DELAY_SECONDS - time_since_last_call
        print(f"[{call_description}] Proactively waiting {wait_for:.2f} seconds to maintain {60/PROACTIVE_DELAY_SECONDS:.0f} RPM limit.")
        time.sleep(wait_for)

    # Update last call time immediately before the API call attempt.
    _last_api_call_time = time.time()
    # --- End Proactive Rate Limiting Logic ---

    retry_count = 0
    current_backoff_delay = initial_delay

    while retry_count < max_retries:
        start_time = time.perf_counter() # Start timing for the current attempt
        try:
            response = model_instance.generate_content(
                prompt_content,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                    "top_p": 1,
                    "top_k": 40
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            end_time = time.perf_counter() # End timing for the current attempt
            duration = end_time - start_time

            prompt_token_count = 0
            candidates_token_count = 0
            if response.usage_metadata:
                prompt_token_count = response.usage_metadata.prompt_token_count
                candidates_token_count = response.usage_metadata.candidates_token_count
            
            print(f"[{call_description}] API call completed in {duration:.2f} seconds. Tokens used: Prompt={prompt_token_count}, Candidates={candidates_token_count}.")
            return response, prompt_token_count, candidates_token_count, duration # Return metrics

        except exceptions.ResourceExhausted as e:
            retry_count += 1
            print(f"[{call_description}] Rate limit hit! Retrying in {current_backoff_delay} seconds (attempt {retry_count}/{max_retries}). Error: {e}")

            retry_after_api_suggestion = current_backoff_delay
            wait_time_on_error = max(retry_after_api_suggestion, min_retry_delay)
            time.sleep(wait_time_on_error)

            current_backoff_delay *= 2
            _last_api_call_time = time.time() # Update last call time after wait

        except Exception as e:
            print(f"[{call_description}] An unexpected error occurred during API call: {e}")
            # For unexpected errors, return default metrics to avoid breaking the pipeline, but re-raise the exception
            raise # Re-raise the exception as it's unexpected
            
    # If all retries are exhausted, raise an exception and return default metrics
    raise Exception(f"[{call_description}] Failed to get a response after {max_retries} retries due to rate limits.")


def configure_gemini():
    """Configures the Gemini API with the provided API key."""
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in src/config/settings.py. Please set it.")
    genai.configure(api_key=settings.GEMINI_API_KEY)
    print("[+] Gemini API configured.")

def load_image_from_path(image_path: Path) -> Image.Image:
    """Loads an image from a given path."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")
    return Image.open(image_path)

def _extract_and_clean_json(response_text: str) -> str:
    """
    Extracts JSON string from a markdown code block if present.
    Handles cases where LLM wraps JSON in ```json ... ``` using regex for robustness.
    """
    match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if match:
        extracted_json = match.group(1)
        print(f"    [!] Extracted JSON from markdown block.")
        return extracted_json.strip() # Ensure to strip whitespace around extracted JSON
    return response_text.strip() # Return original and strip if no markdown block found

def _robust_json_load(json_string: str, call_description: str = "JSON parsing") -> Optional[Dict[str, Any]]:
    """
    Attempts to load a JSON string, with a specific heuristic for common
    "Invalid \escape" errors in LLM-generated LaTeX within JSON.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"    [X] {call_description}: JSONDecodeError during initial parse: {e}")
        if "Invalid \\escape" in str(e):
            print(f"    {call_description}: Attempting common fix for unescaped LaTeX backslashes...")
            try:
                # This is a heuristic. It aggressively replaces single backslashes
                # with double backslashes. This is often necessary when LLMs output LaTeX
                # (e.g., `\lambda`) directly into JSON strings without escaping it
                # as `\\lambda` as JSON requires.
                fixed_json_string = json_string.replace('\\', '\\\\')
                
                # Check for accidental double escaping of valid JSON escapes like \n
                # This is a trade-off. For the "Invalid \escape" error, the aggressive replace
                # is usually the intended fix, as the LLM failed to escape it initially.
                
                print(f"    {call_description}: Fixed string (start): {fixed_json_string[:200]}...")
                return json.loads(fixed_json_string)
            except json.JSONDecodeError as inner_e:
                print(f"    [X] {call_description}: Aggressive backslash fix also failed: {inner_e}")
                return None
        else:
            # For other JSONDecodeErrors, log and return None or re-raise if you prefer
            print(f"    [X] {call_description}: Unhandled JSONDecodeError type: {e}")
            return None
    except Exception as e:
        print(f"    [X] {call_description}: Unexpected error in robust JSON loading: {e}")
        return None

def gemini_extract_latex_from_image(image_path: Path, model_instance: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Extracts LaTeX code from an image using the Gemini API, with retries.
    Returns a dictionary including extracted LaTeX and API metrics.
    """
    try:
        image = load_image_from_path(image_path)
        prompt_parts = [
            "Extract all distinct mathematical expressions from this image and provide them exclusively in LaTeX format. "
            "Each formula should be on a new line. Do not include any conversational text, explanations, or code blocks. "
            "Only output the raw LaTeX strings. If no math is present, output nothing.",
            image
        ]
        print(f"    [>] Sending image {image_path.name} to Gemini for LaTeX extraction...")
        response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
            model_instance=model_instance,
            prompt_content=prompt_parts,
            call_description=f"LaTeX extraction for {image_path.name}"
        )
        latex_text = None
        if response and response.candidates and response.candidates[0].content.parts:
            latex_text = response.candidates[0].content.parts[0].text
        
        return {
            "extracted_latex": latex_text,
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": candidate_tokens,
            "api_call_duration": duration
        }
    except Exception as e:
        print(f"    [X] Error extracting LaTeX from {image_path.name}: {e}")
        return {
            "extracted_latex": None,
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

def gemini_generate_qna_and_solution(image_path: Path, extracted_latex : str, model_instance : genai.GenerativeModel) -> Dict[str, Any]:
    """
    Generates a math Q&A problem, its detailed step-by-step solution, and a canonical final answer
    from an image and extracted LaTeX using the Gemini API, with retries.
    Returns a dictionary including Q&A data and API metrics.
    """
    try:
        img = load_image_from_path(image_path)
        prompt_parts = [
            "Given this image and the extracted LaTeX expressions, generate a "
            "challenging math problem, a detailed step-by-step solution, "
            "a single canonical final answer, an array of critical expressions in LaTeX, "
            "an array of key topics, and an array of critical steps. "
            "Format the output as a JSON object with the following keys: "
            "'question', 'solution', 'final_answer' (string in LaTeX if applicable), "
            "'critical_expressions' (array of LaTeX strings), 'topics' (array of strings), "
            "and 'critical_steps' (array of strings).",
            f"Extracted LaTeX:\n{extracted_latex}",
            img
        ]
        print(f"    [>] Sending image {image_path.name} to Gemini for Q&A generation...")
        
        response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
            model_instance=model_instance,
            prompt_content=prompt_parts,
            call_description=f"Q&A generation for {image_path.name}"
        )
        
        json_string = ""
        if response and response.candidates and response.candidates[0].content.parts:
            raw_response_text = response.candidates[0].content.parts[0].text
            json_string = _extract_and_clean_json(raw_response_text) # Clean the JSON string

        # Use the robust JSON loader here
        qna_data = _robust_json_load(json_string, call_description=f"Q&A JSON parse for {image_path.name}")
        
        generated_question = "Error: Failed to parse Q&A from image."
        solution = f"Error: {json_string}"
        final_answer = "N/A"
        critical_expressions = []
        topics = []
        critical_steps = []

        if qna_data: # Check if parsing was successful
            # Basic validation based on the requested output format in the prompt
            required_keys = ["question", "solution", "final_answer", "critical_expressions", "topics", "critical_steps"]
            if all(key in qna_data for key in required_keys):
                generated_question = qna_data["question"]
                solution = qna_data["solution"]
                final_answer = qna_data["final_answer"]
                critical_expressions = qna_data["critical_expressions"]
                topics = qna_data["topics"]
                critical_steps = qna_data["critical_steps"]
            else:
                print(f"    [X] Missing required keys in Gemini's Q&A JSON response for {image_path.name}.")
        
        return {
            "generated_question": generated_question,
            "solution": solution,
            "final_answer": final_answer,
            "critical_expressions": critical_expressions,
            "topics": topics,
            "critical_steps": critical_steps,
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": candidate_tokens,
            "api_call_duration": duration
        }
    except Exception as e:
        print(f"    [X] Error generating Q&A for {image_path.name}: {e}")
        return {
            "generated_question": f"Error during Q&A generation: {e}",
            "solution": "N/A",
            "final_answer": "N/A",
            "critical_expressions": [],
            "topics": [],
            "critical_steps": [],
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

def gemini_generate_solution_for_question(question: str, model_instance: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Generates a step-by-step solution for a given mathematical question using the Gemini API.
    This is intended to produce the LLM2 proposed solution.
    Returns a dictionary including the solution and API metrics.
    """
    try:
        prompt_parts = [
            f"Solve the following mathematical question step-by-step. Provide a detailed solution.\nQuestion: {question}",
        ]
        print(f"    [>] Sending question to Gemini for LLM2 solution generation...")
        response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
            model_instance=model_instance,
            prompt_content=prompt_parts,
            call_description="LLM2 Solution generation"
        )
        solution_text = "N/A: No solution generated by LLM2."
        if response and response.candidates and response.candidates[0].content.parts:
            solution_text = response.candidates[0].content.parts[0].text.strip()
        
        return {
            "solution_text": solution_text,
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": candidate_tokens,
            "api_call_duration": duration
        }
    except Exception as e:
        print(f"    [X] Error generating LLM2 solution for question: {e}")
        return {
            "solution_text": f"Error generating LLM2 solution: {e}",
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

def gemini_extract_final_answer_from_solution(solution_text: str, model_instance: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Extracts the concise final numerical or mathematical answer from a detailed solution text.
    Returns a dictionary including the extracted answer and API metrics.
    """
    if not solution_text or solution_text.lower() in ["n/a", "error", "n/a: no solution generated by llm2."]:
        return {
            "extracted_answer": "N/A",
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

    try:
        prompt_parts = [
            f"From the following mathematical solution, extract ONLY the final concise answer. "
            f"If the answer is numerical, provide only the number. If it's a mathematical expression, provide it in LaTeX. "
            f"Do not include any other text or explanation. If no clear final answer is present, output 'N/A'.\nSolution: {solution_text}",
        ]
        print(f"    [>] Extracting final answer from solution...")
        response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
            model_instance=model_instance,
            prompt_content=prompt_parts,
            call_description="Final answer extraction"
        )
        extracted_answer = "N/A"
        if response and response.candidates and response.candidates[0].content.parts:
            extracted_answer = response.candidates[0].content.parts[0].text.strip()
            # Basic cleanup: sometimes it might wrap in markdown even if not requested
            if extracted_answer.startswith("```") and extracted_answer.endswith("```"):
                extracted_answer = extracted_answer.strip("` \n")
            if extracted_answer.lower() == "n/a":
                extracted_answer = "N/A" # Ensure consistent "N/A" capitalization
        
        return {
            "extracted_answer": extracted_answer,
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": candidate_tokens,
            "api_call_duration": duration
        }
    except Exception as e:
        print(f"    [X] Error extracting final answer: {e}")
        return {
            "extracted_answer": "N/A",
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

def gemini_evaluate_solution_as_judge(question: str, llm1_solution: str, llm2_solution: str) -> Dict[str, Any]:
    """
    Evaluates LLM2's solution against LLM1's using Gemini as a judge.
    Returns a dictionary including the evaluation result and API metrics.
    """
    prompt_parts = [
        f"You are an expert mathematical judge. Here is a question:\nQuestion: {question}\n\n"
        f"Here is a correct ground truth solution (LLM1):\nLLM1 Solution: {llm1_solution}\n\n"
        f"Here is a proposed solution (LLM2):\nLLM2 Solution: {llm2_solution}\n\n"
        "Compare LLM2's solution to LLM1's. Is LLM2's solution correct and equivalent to LLM1's? "
        "Respond only with 'Correct' or 'Incorrect'. No explanations.",
    ]
    print(f"    [>] Evaluating solution with Gemini judge...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="Solution evaluation"
    )
    evaluation_result = "Error: No response from judge."
    if response and response.candidates and response.candidates[0].content.parts:
        evaluation_result = response.candidates[0].content.parts[0].text.strip()
    
    return {
        "evaluation_result": evaluation_result,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }

def gemini_determine_category_and_subcategory(question: str, solution: str, extracted_math: str) -> Dict[str, Any]:
    """
    Determines the category and subcategory of the math problem using Gemini.
    Returns a dictionary including category data and API metrics.
    """
    prompt_parts = [
        f"Based on the following mathematical question, solution, and extracted math, categorize the problem.\n"
        f"Question: {question}\nSolution: {solution}\nExtracted Math: {extracted_math}\n\n"
        "Provide a single, concise category and subcategory (e.g., 'Algebra', 'Linear Equations'; 'Calculus', 'Derivatives'). "
        "Format your response as a JSON object with keys `category` and `subcategory`."
    ]
    print(f"    [>] Determining category with Gemini...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="Category determination"
    )
    json_string = ""
    if response and response.candidates and response.candidates[0].content.parts:
        raw_response_text = response.candidates[0].content.parts[0].text
        json_string = _extract_and_clean_json(raw_response_text) # Clean the JSON string

    # Use the robust JSON loader here
    category_data = _robust_json_load(json_string, call_description="Category JSON parse")
    
    category = "Uncategorized"
    subcategory = "General"
    if category_data:
        category = category_data.get("category", "Uncategorized")
        subcategory = category_data.get("subcategory", "General")
    else:
        print(f"    [X] Gemini returned malformed JSON for category, or parsing failed.")
    
    return {
        "category": category,
        "subcategory": subcategory,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }

def gemini_generate_topics(question: str, solution: str, extracted_math: str) -> Dict[str, Any]:
    """
    Generates a list of topics related to the math problem using Gemini.
    Returns a dictionary including the list of topics and API metrics.
    """
    prompt_parts = [
        f"Identify key mathematical topics covered in this problem:\n"
        f"Question: {question}\nSolution: {solution}\nExtracted Math: {extracted_math}\n\n"
        "Provide a comma-separated list of topics (e.g., 'Differentiation', 'Integration', 'Trigonometry')."
    ]
    print(f"    [>] Generating topics with Gemini...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="Topic generation"
    )
    response_text = ""
    if response and response.candidates and response.candidates[0].content.parts:
        response_text = response.candidates[0].content.parts[0].text
    topics_list = [topic.strip() for topic in response_text.split(',') if topic.strip()]
    
    return {
        "topics_list": topics_list,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }

def gemini_determine_api_answer_status(llm2_proposed_solution: str) -> Dict[str, Any]:
    """
    Determines if LLM2 provided a valid answer.
    Returns a dictionary including the status and API metrics.
    """
    prompt_parts = [
        f"Review the following solution: '{llm2_proposed_solution}'. Does it appear to be a legitimate attempt at solving a math problem, or is it an empty/error/irrelevant response? Respond with 'Yes' if it's a valid attempt, or 'No' otherwise.",
    ]
    print(f"    [>] Determining API answer status with Gemini...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="API answer status determination"
    )
    status = "0" # Default to "0" as previously
    if response and response.candidates and response.candidates[0].content.parts:
        status = response.candidates[0].content.parts[0].text.strip().lower()
    
    return {
        "status": status,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }

def gemini_generate_alternate_answers(canonical_answer: str, model_instance: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Generates alternate correct answers if possible, using Gemini.
    Returns a dictionary including the list of alternate answers and API metrics.
    """
    if not canonical_answer or canonical_answer.lower() in ["n/a", "error"]:
        return {
            "alternate_answers": [],
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }
        
    prompt_parts = [
        f"Given the canonical final answer: '{canonical_answer}'. "
        "Provide 1-2 mathematically equivalent alternate representations or forms of this answer. "
        "If no meaningful alternate forms exist (e.g., for simple numbers), state 'None'. "
        "Format as a comma-separated list of LaTeX strings.",
    ]
    print(f"    [>] Generating alternate answers with Gemini...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model_instance, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="Alternate answers generation"
    )
    response_text = ""
    alternate_answers_list = []
    if response and response.candidates and response.candidates[0].content.parts:
        response_text = response.candidates[0].content.parts[0].text
        if "none" not in response_text.lower() and response_text.strip():
            alternate_answers_list = [ans.strip() for ans in response_text.split(',') if ans.strip()]
    
    return {
        "alternate_answers": alternate_answers_list,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }

def gemini_generate_critical_expressions_and_steps(question: str, llm1_solution: str, model_instance: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Generates critical mathematical expressions and solution steps using Gemini.
    Returns a dictionary including critical expressions/steps and API metrics.
    """
    prompt_parts = [
        f"From the following question and its solution, identify:\n"
        f"1. Key mathematical expressions that define the problem or its core parts.\n"
        f"2. The most critical steps or transformations in the solution process.\n"
        f"Question: {question}\nSolution: {llm1_solution}\n\n"
        "Format your response as a JSON object with two keys: `critical_expressions` (a list of LaTeX strings) and `critical_steps` (a list of concise English descriptions of steps)."
    ]
    print(f"    [>] Generating critical expressions and steps with Gemini...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model_instance, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="Critical expressions and steps generation"
    )
    json_string = ""
    critical_expressions = []
    critical_steps = []

    if response and response.candidates and response.candidates[0].content.parts:
        raw_response_text = response.candidates[0].content.parts[0].text
        json_string = _extract_and_clean_json(raw_response_text) # Clean the JSON string
    
    # Use the robust JSON loader here
    critical_data = _robust_json_load(json_string, call_description="Critical data JSON parse")
    
    if critical_data:
        critical_expressions = critical_data.get("critical_expressions", [])
        critical_steps = critical_data.get("critical_steps", [])
    else:
        print(f"    [X] Gemini returned malformed JSON for critical data, or parsing failed.")
    
    return {
        "critical_expressions": critical_expressions,
        "critical_steps": critical_steps,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }