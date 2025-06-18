#src/generator/qna_generator.py
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from src.utils import gemini_api_utils
import google.generativeai as genai

class QnAGenerator:
    def __init__(self, model: genai.GenerativeModel, output_dir: Path):
        self.model = model
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.qna_data: List[Dict[str, Any]] = []

    def process_image_for_qna(self, image_path: Path) -> Dict[str, Any]:
        """
        Processes a single image to generate a Q&A pair, solution, and related metadata.
        Aggregates token usage and time spent across all API calls for this question.
        """
        print(f"  [>>] Processing image {image_path.name} for Q&A generation...")

        # Initialize overall metrics for this question
        total_prompt_tokens_per_question = 0
        total_candidate_tokens_per_question = 0
        total_api_time_per_question = 0.0

        try:
            # 1. Extract LaTeX expressions
            latex_extraction_result = gemini_api_utils.gemini_extract_latex_from_image(image_path, self.model)
            extracted_latex = latex_extraction_result["extracted_latex"]
            total_prompt_tokens_per_question += latex_extraction_result["prompt_token_count"]
            total_candidate_tokens_per_question += latex_extraction_result["candidates_token_count"]
            total_api_time_per_question += latex_extraction_result["api_call_duration"]

            # 2. Generate Q&A, solution, final answer, etc. (LLM1)
            qna_gen_result = gemini_api_utils.gemini_generate_qna_and_solution(image_path, extracted_latex if extracted_latex else "", self.model)
            generated_question_llm1 = qna_gen_result["generated_question"]
            llm1_solution = qna_gen_result["solution"]
            llm1_final_answer = qna_gen_result["final_answer"]
            llm1_critical_expressions = qna_gen_result["critical_expressions"]
            topics = qna_gen_result["topics"]
            llm1_critical_steps = qna_gen_result["critical_steps"]
            total_prompt_tokens_per_question += qna_gen_result["prompt_token_count"]
            total_candidate_tokens_per_question += qna_gen_result["candidates_token_count"]
            total_api_time_per_question += qna_gen_result["api_call_duration"]

            # 3. Generate LLM2 proposed solution
            llm2_solution_result = gemini_api_utils.gemini_generate_solution_for_question(generated_question_llm1, self.model)
            llm2_proposed_solution = llm2_solution_result["solution_text"]
            total_prompt_tokens_per_question += llm2_solution_result["prompt_token_count"]
            total_candidate_tokens_per_question += llm2_solution_result["candidates_token_count"]
            total_api_time_per_question += llm2_solution_result["api_call_duration"]

            # 4. Extract final answer from LLM2 solution
            # IMPORTANT FIX: Pass only the solution text from the dictionary
            llm2_final_answer_result = gemini_api_utils.gemini_extract_final_answer_from_solution(llm2_proposed_solution, self.model)
            llm2_final_answer_extracted = llm2_final_answer_result["extracted_answer"]
            total_prompt_tokens_per_question += llm2_final_answer_result["prompt_token_count"]
            total_candidate_tokens_per_question += llm2_final_answer_result["candidates_token_count"]
            total_api_time_per_question += llm2_final_answer_result["api_call_duration"]
            
            # 5. Evaluate LLM2's solution against LLM1's (Gemini as judge)
            evaluation_result = gemini_api_utils.gemini_evaluate_solution_as_judge(generated_question_llm1, llm1_solution, llm2_proposed_solution)
            judge_evaluation = evaluation_result["evaluation_result"]
            total_prompt_tokens_per_question += evaluation_result["prompt_token_count"]
            total_candidate_tokens_per_question += evaluation_result["candidates_token_count"]
            total_api_time_per_question += evaluation_result["api_call_duration"]

            # 6. Determine API answer status
            api_answer_status_result = gemini_api_utils.gemini_determine_api_answer_status(llm2_proposed_solution)
            api_answer_status = api_answer_status_result["status"]
            total_prompt_tokens_per_question += api_answer_status_result["prompt_token_count"]
            total_candidate_tokens_per_question += api_answer_status_result["candidates_token_count"]
            total_api_time_per_question += api_answer_status_result["api_call_duration"]

            # 7. Generate alternate answers
            alternate_answers_result = gemini_api_utils.gemini_generate_alternate_answers(llm1_final_answer, self.model)
            alternate_answers = alternate_answers_result["alternate_answers"]
            total_prompt_tokens_per_question += alternate_answers_result["prompt_token_count"]
            total_candidate_tokens_per_question += alternate_answers_result["candidates_token_count"]
            total_api_time_per_question += alternate_answers_result["api_call_duration"]

            # 8. Determine Category and Subcategory
            category_result = gemini_api_utils.gemini_determine_category_and_subcategory(generated_question_llm1, llm1_solution, extracted_latex if extracted_latex else "")
            category = category_result["category"]
            subcategory = category_result["subcategory"]
            total_prompt_tokens_per_question += category_result["prompt_token_count"]
            total_candidate_tokens_per_question += category_result["candidates_token_count"]
            total_api_time_per_question += category_result["api_call_duration"]

            return {
                "image_filename": image_path.name,
                "extracted_latex": extracted_latex,
                "generated_question_llm1": generated_question_llm1,
                "llm1_solution": llm1_solution,
                "llm1_final_answer": llm1_final_answer,
                "llm2_proposed_solution": llm2_proposed_solution,
                "llm2_final_answer_extracted": llm2_final_answer_extracted,
                "judge_evaluation": judge_evaluation,
                "api_answer_status": api_answer_status,
                "llm1_critical_expressions": llm1_critical_expressions,
                "topics": topics,
                "llm1_critical_steps": llm1_critical_steps,
                "alternate_answers": alternate_answers,
                "category": category,
                "subcategory": subcategory,
                "total_prompt_tokens_per_question": total_prompt_tokens_per_question,
                "total_candidate_tokens_per_question": total_candidate_tokens_per_question,
                "total_api_time_per_question": total_api_time_per_question
            }

        except Exception as e:
            print(f"  [X] Failed to process {image_path.name}: {e}")
            return {
                "image_filename": image_path.name,
                "extracted_latex": "Error",
                "generated_question_llm1": f"Error processing image: {e}",
                "llm1_solution": "Error",
                "llm1_final_answer": "Error",
                "llm2_proposed_solution": "Error",
                "llm2_final_answer_extracted": "Error",
                "judge_evaluation": "Error",
                "api_answer_status": "No",
                "llm1_critical_expressions": [],
                "topics": [],
                "llm1_critical_steps": [],
                "alternate_answers": [],
                "category": "Error",
                "subcategory": "Error",
                "total_prompt_tokens_per_question": total_prompt_tokens_per_question, # Still include totals even on error
                "total_candidate_tokens_per_question": total_candidate_tokens_per_question,
                "total_api_time_per_question": total_api_time_per_question
            }

    def save_qna_data(self):
        if not self.qna_data:
            print("[!] No Q&A data to save.")
            return

        df = pd.DataFrame(self.qna_data)
        output_filepath = self.output_dir / "final_qna_dataset.csv"
        df.to_csv(output_filepath, index=False)
        print(f"[✓] Q&A data saved to {output_filepath}")