"""
APO Optimizer for Agents: Optimizes agent system prompts using the APO algorithm.
数据格式：list[dict]，每条必须含 "input" 和 "output"。Agent 只处理 input，算法用 output 评估。
"""
from evaluator_agent import AgentEvaluator
from prompt_generator import PromptGenerator
from dataset import load_dataset_from_json
from agent_protocol import as_rollout
import random
from typing import List, Dict, Any


class APOOptimizerAgent:
    """
    Optimizes agent system prompts using the APO (Automatic Prompt Optimization) algorithm.
    """

    def __init__(self, agent, dataset: List[Dict[str, Any]], llm_model_name="gpt-3.5-turbo",
                 api_key=None, base_url=None, reward_prompt_path="reward.txt"):
        """
        Initialize the APO Optimizer for agents.

        Args:
            agent: 可调用 (input_data, system_prompt) -> output
            dataset: List of dicts, each with "input" and "output" keys
            llm_model_name: Name of the LLM model for prompt generation and evaluation
            api_key: API key for the LLM service
            base_url: Base URL for the LLM API
            reward_prompt_path: Path to the reward evaluation prompt template
        """
        self._rollout = as_rollout(agent)
        self.evaluator = AgentEvaluator(
            self._rollout,
            llm_model_name=llm_model_name,
            api_key=api_key,
            base_url=base_url,
            reward_prompt_path=reward_prompt_path
        )
        self.generator = PromptGenerator(model_name=llm_model_name, api_key=api_key, base_url=base_url)
        self.dataset = dataset
        self.best_system_prompt = ""
        self.best_score = -1.0
        self.history = []
    
    @classmethod
    def from_dataset_path(cls, agent, dataset_path, llm_model_name="gpt-3.5-turbo",
                         api_key=None, base_url=None, reward_prompt_path="reward.txt"):
        """
        Create optimizer by loading dataset from a JSON file (list of dicts with "input" and "output").
        """
        dataset = load_dataset_from_json(dataset_path)
        return cls(agent, dataset, llm_model_name, api_key, base_url, reward_prompt_path)
    
    def _build_feedback(
        self,
        current_score,
        current_prompt,
        batch: List[Dict[str, Any]] = None,
        model_outputs=None,
        max_examples=5,
    ):
        """
        Build feedback message for prompt generation.
        dataset 每条为 dict，含 "input" 和 "output"；这里只取 output 作为正确答案样例。
        """
        examples = batch if batch is not None else self.dataset
        outputs = []
        for item in examples:
            if isinstance(item, dict) and "output" in item:
                outputs.append(str(item["output"]))
            if len(outputs) >= max_examples:
                break
        
        outputs_text = "; ".join(outputs) if outputs else "N/A"
        model_outputs = model_outputs or []
        safe_outputs = []
        for x in model_outputs[:max_examples]:
            try:
                safe_outputs.append("N/A" if x is None else str(x))
            except Exception:
                safe_outputs.append("N/A")
        model_outputs_text = "; ".join(safe_outputs) if safe_outputs else "N/A"
        
        return (
            f"The current system prompt achieved a score of {current_score:.4f}. "
            f"Current system prompt: {current_prompt} "
            "Try to improve it further. Focus on clarity and precision. "
            f"Correct answers examples: {outputs_text} "
            f"Model answers examples: {model_outputs_text}"
        )
    
    def _print_outputs_and_scores(self, outputs, scores):
        """Print model outputs and their scores."""
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
        """
        Optimize the agent's system prompt.
        
        Args:
            initial_prompt: Initial system prompt to start with
            num_iterations: Number of optimization iterations
            num_candidates: Number of candidate prompts to generate each iteration
            batch_size: Size of each batch (for mini-batch optimization)
            shuffle: Whether to shuffle the dataset
            update_per_batch: Whether to update after each batch (mini-batch mode)
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Tuple of (best_prompt, best_score, history)
        """
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        current_prompt = initial_prompt
        self.best_system_prompt = initial_prompt
        print(f"Starting optimization with initial system prompt: '{current_prompt}'")
        
        if not update_per_batch:
            # Full dataset mode
            return self._optimize_full_dataset(
                current_prompt,
                num_iterations,
                num_candidates,
                batch_size,
                shuffle,
                max_workers,
            )
        else:
            # Mini-batch mode
            return self._optimize_mini_batch(
                current_prompt,
                num_iterations,
                num_candidates,
                batch_size,
                shuffle,
                max_workers,
            )
    
    def _optimize_full_dataset(
        self,
        current_prompt,
        num_iterations,
        num_candidates,
        batch_size,
        shuffle,
        max_workers,
    ):
        """Optimize using full dataset evaluation."""
        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")
            
            current_score, model_outputs, model_scores = self.evaluator.evaluate_agent_with_details(
                self.dataset,
                current_prompt,
                batch_size=batch_size,
                max_workers=max_workers,
                shuffle=shuffle,
                max_examples=len(self.dataset),
            )
            
            self.history.append({"iteration": i+1, "prompt": current_prompt, "score": current_score})
            self._print_outputs_and_scores(model_outputs, model_scores)
            
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_system_prompt = current_prompt
                print(f"New best prompt found (score: {self.best_score:.4f}): '{self.best_system_prompt}'")
            
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
            print(f"Generated {len(candidate_prompts)} candidate system prompts.")
            
            best_candidate_score = current_score
            best_candidate_prompt = current_prompt
            
            for j, candidate in enumerate(candidate_prompts):
                print(f"  Evaluating candidate {j+1}: '{candidate[:70]}...'")
                candidate_score = self.evaluator.evaluate_agent(
                    self.dataset,
                    candidate,
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
                self.best_system_prompt = best_candidate_prompt
                print(f"Iteration {i+1} found a new overall best prompt (score: {self.best_score:.4f}).")
            
            if best_candidate_score <= current_score and i > 0:
                print("No significant improvement in this iteration. Stopping optimization.")
                break

            current_prompt = best_candidate_prompt

        print("\n--- Optimization Finished ---")
        print(f"Final Best System Prompt: '{self.best_system_prompt}'")
        print(f"Final Best Score: {self.best_score:.4f}")
        return self.best_system_prompt, self.best_score, self.history
    
    def _optimize_mini_batch(
        self,
        current_prompt,
        num_iterations,
        num_candidates,
        batch_size,
        shuffle,
        max_workers,
    ):
        """Optimize using mini-batch evaluation."""
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
                
                current_score, model_outputs, model_scores = self.evaluator.evaluate_agent_batch(batch, current_prompt, max_workers=max_workers)
                
                self.history.append({
                    "iteration": i+1,
                    "batch_id": batch_count,
                    "prompt": current_prompt,
                    "score": current_score,
                })
                self._print_outputs_and_scores(model_outputs, model_scores)
                
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_system_prompt = current_prompt
                    improved_in_iteration = True
                    print(f"New best prompt found (score: {self.best_score:.4f}): '{self.best_system_prompt}'")
                
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
                print(f"Batch {batch_count}: generated {len(candidate_prompts)} candidate system prompts.")
                
                best_candidate_score = current_score
                best_candidate_prompt = current_prompt
                
                for j, candidate in enumerate(candidate_prompts):
                    print(f"  Evaluating candidate {j+1}: '{candidate[:70]}...'")
                    candidate_score, _, _ = self.evaluator.evaluate_agent_batch(batch, candidate, max_workers=max_workers)
                    
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
                        print(f"    Candidate {j+1} is better (score: {candidate_score:.4f}).")
                
                if best_candidate_score > self.best_score:
                    self.best_score = best_candidate_score
                    self.best_system_prompt = best_candidate_prompt
                    improved_in_iteration = True
                    print(f"Batch {batch_count} found a new overall best prompt (score: {self.best_score:.4f}).")
                
                current_prompt = best_candidate_prompt

            if not improved_in_iteration and i > 0:
                print("No significant improvement in this iteration. Stopping optimization.")
                break
        
        print("\n--- Optimization Finished ---")
        print(f"Final Best System Prompt: '{self.best_system_prompt}'")
        print(f"Final Best Score: {self.best_score:.4f}")
        return self.best_system_prompt, self.best_score, self.history