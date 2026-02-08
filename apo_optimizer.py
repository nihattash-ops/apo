from evaluator import Evaluator
from prompt_generator import PromptGenerator
import json
import os
import random

class APOOptimizer:
    def __init__(self, dataset_path, llm_model_name="gpt-3.5-turbo", api_key=None, base_url=None):
        if not api_key:
            raise ValueError("API key is required. Provide it from main.py.")
        self.evaluator = Evaluator(model_name=llm_model_name, api_key=api_key, base_url=base_url)
        self.generator = PromptGenerator(model_name=llm_model_name, api_key=api_key, base_url=base_url)
        self.dataset = self._load_dataset(dataset_path)
        self.best_prompt = None
        self.best_score = -1.0
        self.history = []

    def _load_dataset(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_feedback(
        self,
        current_score,
        current_prompt,
        batch=None,
        model_outputs=None,
        max_examples=5,
    ):
        examples = batch if batch is not None else self.dataset
        outputs = []
        for item in examples:
            if isinstance(item, dict) and "output" in item:
                outputs.append(str(item["output"]))
            if len(outputs) >= max_examples:
                break
        outputs_text = "; ".join(outputs) if outputs else "N/A"
        model_outputs = model_outputs or []
        model_outputs_text = "; ".join(model_outputs[:max_examples]) if model_outputs else "N/A"
        return (
            f"The current prompt achieved a score of {current_score:.4f}. "
            f"Current prompt: {current_prompt} "
            "Try to improve it further. Focus on clarity and precision. "
            f"Correct answers examples: {outputs_text} "
            f"Model answers examples: {model_outputs_text}"
        )

    def _print_outputs_and_scores(self, outputs, scores):
        if not outputs:
            return
        print("Model outputs and scores:")
        for idx, output in enumerate(outputs, start=1):
            score = None
            if scores and idx - 1 < len(scores):
                score = scores[idx - 1]
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            print(f"  {idx}. score={score_text} output={output}")

    def optimize(
        self,
        initial_prompt,
        num_iterations=5,
        num_candidates=3,
        batch_size=4,
        shuffle=True,
        update_per_batch=True,
        max_workers=8,
    ):
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        current_prompt = initial_prompt
        print(f"Starting optimization with initial prompt: '{current_prompt}'")

        if not update_per_batch:
            for i in range(num_iterations):
                print(f"\n--- Iteration {i+1}/{num_iterations} ---")

                current_score, model_outputs, _, model_scores = self.evaluator.evaluate_prompt_with_details(
                    current_prompt,
                    self.dataset,
                    batch_size=batch_size,
                    max_workers=max_workers,
                    shuffle=shuffle,
                    max_examples=len(self.dataset),
                )
                self.history.append({"iteration": i+1, "prompt": current_prompt, "score": current_score})
                self._print_outputs_and_scores(model_outputs, model_scores)

                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_prompt = current_prompt
                    print(f"New best prompt found (score: {self.best_score:.4f}): '{self.best_prompt}'")

                feedback = self._build_feedback(
                    current_score,
                    current_prompt,
                    model_outputs=model_outputs,
                )
                candidate_prompts = self.generator.generate_candidates(
                    current_prompt,
                    num_candidates,
                    feedback=feedback,
                )
                print(f"Generated {len(candidate_prompts)} candidate prompts.")

                best_candidate_score = current_score
                best_candidate_prompt = current_prompt

                for j, candidate in enumerate(candidate_prompts):
                    print(f"  Evaluating candidate {j+1}: '{candidate[:70]}...'")
                    candidate_score = self.evaluator.evaluate_prompt(
                        candidate,
                        self.dataset,
                        batch_size=batch_size,
                        max_workers=max_workers,
                        shuffle=shuffle,
                    )
                    self.history.append({
                        "iteration": i+1,
                        "candidate_id": j+1,
                        "prompt": candidate,
                        "score": candidate_score,
                    })

                    if candidate_score > best_candidate_score:
                        best_candidate_score = candidate_score
                        best_candidate_prompt = candidate
                        print(f"    Candidate {j+1} is better (score: {candidate_score:.4f}).")

                if best_candidate_score > self.best_score:
                    self.best_score = best_candidate_score
                    self.best_prompt = best_candidate_prompt
                    print(
                        f"Iteration {i+1} found a new overall best prompt (score: {self.best_score:.4f})."
                    )

                if best_candidate_score <= current_score and i > 0:
                    print("No significant improvement in this iteration. Stopping optimization.")
                    break

                current_prompt = best_candidate_prompt

            print("\n--- Optimization Finished ---")
            print(f"Final Best Prompt: '{self.best_prompt}'")
            print(f"Final Best Score: {self.best_score:.4f}")
            return self.best_prompt, self.best_score, self.history

        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")
            improved_in_iteration = False

            data = list(self.dataset)
            if shuffle:
                random.shuffle(data)

            batch_count = 0
            for start in range(0, len(data), batch_size):
                batch_count += 1
                batch = data[start:start + batch_size]

                current_score, model_outputs, _, model_scores = self.evaluator.evaluate_prompt_batch_with_details(
                    current_prompt,
                    batch,
                    max_workers=max_workers,
                )
                self.history.append({
                    "iteration": i+1,
                    "batch_id": batch_count,
                    "prompt": current_prompt,
                    "score": current_score,
                })
                self._print_outputs_and_scores(model_outputs, model_scores)

                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_prompt = current_prompt
                    improved_in_iteration = True
                    print(
                        f"New best prompt found (score: {self.best_score:.4f}): '{self.best_prompt}'"
                    )

                feedback = self._build_feedback(
                    current_score,
                    current_prompt,
                    batch=batch,
                    model_outputs=model_outputs,
                )
                candidate_prompts = self.generator.generate_candidates(
                    current_prompt,
                    num_candidates,
                    feedback=feedback,
                )
                print(
                    f"Batch {batch_count}: generated {len(candidate_prompts)} candidate prompts."
                )

                best_candidate_score = current_score
                best_candidate_prompt = current_prompt

                for j, candidate in enumerate(candidate_prompts):
                    print(f"  Evaluating candidate {j+1}: '{candidate[:70]}...'")
                    candidate_score = self.evaluator.evaluate_prompt_batch(
                        candidate,
                        batch,
                        max_workers=max_workers,
                    )
                    self.history.append({
                        "iteration": i+1,
                        "batch_id": batch_count,
                        "candidate_id": j+1,
                        "prompt": candidate,
                        "score": candidate_score,
                    })

                    if candidate_score > best_candidate_score:
                        best_candidate_score = candidate_score
                        best_candidate_prompt = candidate
                        print(
                            f"    Candidate {j+1} is better (score: {candidate_score:.4f})."
                        )

                if best_candidate_score > self.best_score:
                    self.best_score = best_candidate_score
                    self.best_prompt = best_candidate_prompt
                    improved_in_iteration = True
                    print(
                        f"Batch {batch_count} found a new overall best prompt (score: {self.best_score:.4f})."
                    )

                current_prompt = best_candidate_prompt

            if not improved_in_iteration and i > 0:
                print("No significant improvement in this iteration. Stopping optimization.")
                break

        print("\n--- Optimization Finished ---")
        print(f"Final Best Prompt: '{self.best_prompt}'")
        print(f"Final Best Score: {self.best_score:.4f}")
        return self.best_prompt, self.best_score, self.history

if __name__ == "__main__":
    from main import API_KEY, LLM_MODEL, BASE_URL

    dummy_dataset = [
        {"input": "What is the capital of France?", "output": "Paris"},
        {"input": "What is 2 + 2?", "output": "4"},
        {"input": "Who painted the Mona Lisa?", "output": "Leonardo da Vinci"},
        {"input": "What is the largest ocean?", "output": "Pacific Ocean"}
    ]
    os.makedirs("data", exist_ok=True)
    with open("data/sample_data.json", "w", encoding="utf-8") as f:
        json.dump(dummy_dataset, f, ensure_ascii=False, indent=4)

    optimizer = APOOptimizer(
        dataset_path="data/sample_data.json",
        llm_model_name=LLM_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    initial_prompt = "Answer the question directly and concisely."
    best_prompt, best_score, history = optimizer.optimize(
        initial_prompt,
        num_iterations=3,
        num_candidates=2,
    )

    print("\nOptimization History:")
    for entry in history:
        print(entry)
