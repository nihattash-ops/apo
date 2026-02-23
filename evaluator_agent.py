"""
Agent Evaluator: Evaluates agent performance on a dataset.
Dataset: list of dicts with "input" and "output" keys. Agent 只接收 input，算法用 output 评估。
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
from openai import OpenAI
from typing import Any, List, Dict


class AgentEvaluator:
    """
    Evaluates an agent's performance on a dataset.
    Uses LLM-based evaluation to score outputs against expected output.
    """

    def __init__(self, agent, llm_model_name="gpt-3.5-turbo", api_key=None, base_url=None,
                 reward_prompt_path="reward.txt"):
        """
        Args:
            agent: 可调用 (input_data, system_prompt) -> output
            llm_model_name: Name of the LLM model for evaluation
            api_key: API key for the LLM service
            base_url: Base URL for the LLM API
            reward_prompt_path: Path to the reward evaluation prompt template
        """
        self.agent = agent
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = llm_model_name
        self.reward_prompt_template = self._load_reward_prompt(reward_prompt_path)

    def _load_reward_prompt(self, path):
        """Load reward prompt template from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Reward prompt file not found at {path}, using default template.")
            return (
                "You are a strict evaluator.\n"
                "Assign a score between 0 and 1 based on how well Output matches Expected.\n"
                "The score must reflect similarity: identical answers = 1, completely different answers = 0.\n"
                "if Output is partially correct, longest the answer is, the lower the score should be.\n"
                "Return only JSON in the form: {{\"score\": <number_between_0_and_1>}}.\n\n"
                "Output: {prediction}\nExpected: {true_label}"
            )

    def _evaluate_output(self, output: Any, expected_output: Any):
        """
        Evaluate the agent's output against the expected output using LLM.
        Returns a score between 0 and 1.
        """
        if output is None:
            return 0.0
        pred_str = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        expect_str = expected_output if isinstance(expected_output, str) else json.dumps(expected_output, ensure_ascii=False)
        evaluation_prompt = self.reward_prompt_template.format(
            prediction=pred_str,
            true_label=expect_str
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a strict evaluator."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content.strip())
            score = result.get("score", 0.0)

            if isinstance(score, (int, float)) and 0.0 <= score <= 1.0:
                return float(score)
            return 0.0

        except Exception as e:
            print(f"Error evaluating output: {e}")
            return 0.0

    def _process_and_evaluate_one(self, item, current_prompt: str):
        """
        Rollout：用 current_prompt 调用 agent(input_data, current_prompt)，再评估。
        Returns (output, score).
        """
        if isinstance(item, dict):
            input_data = item.get("input")
            expected_output = item.get("output")
        else:
            input_data = getattr(item, "input", None)
            expected_output = getattr(item, "output", None)
        output = self.agent(input_data, current_prompt)
        score = self._evaluate_output(output, expected_output)
        return output, score

    def evaluate_agent_batch(self, dataset_batch: List[Dict[str, Any]], current_prompt: str, max_workers=8):
        """
        Evaluate the agent on a batch. 每次调用传入 current_prompt（rollout 风格）。

        Returns:
            Tuple of (average_score, outputs, scores)
        """
        if not dataset_batch:
            return 0.0, [], []

        if max_workers is None or max_workers <= 0:
            max_workers = 1

        worker_count = min(max_workers, len(dataset_batch))
        results = [None] * len(dataset_batch)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {}
            for idx, item in enumerate(dataset_batch):
                future = executor.submit(self._process_and_evaluate_one, item, current_prompt)
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    results[idx] = (None, 0.0)

        outputs = [r[0] for r in results if r is not None]
        scores = [r[1] for r in results if r is not None]

        valid_scores = [s for s in scores if isinstance(s, (int, float))]
        if not valid_scores:
            print("Warning: No valid scores in batch.")
            return 0.0, outputs, scores

        batch_score = sum(valid_scores) / len(valid_scores)
        return batch_score, outputs, scores

    def evaluate_agent(self, dataset: List[Dict[str, Any]], current_prompt: str, batch_size=4, max_workers=8, shuffle=False):
        """Evaluate the agent on the full dataset. Returns average score."""
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if not dataset:
            return 0.0

        data = list(dataset)
        if shuffle:
            random.shuffle(data)

        total_score = 0.0
        total_items = 0

        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            batch_score, _, _ = self.evaluate_agent_batch(batch, current_prompt, max_workers=max_workers)
            total_score += batch_score * len(batch)
            total_items += len(batch)

        average_score = total_score / total_items if total_items > 0 else 0.0
        print(f"Evaluated agent with score: {average_score:.4f}")
        return average_score

    def evaluate_agent_with_details(self, dataset: List[Dict[str, Any]], current_prompt: str, batch_size=4, max_workers=8,
                                    shuffle=False, max_examples=5):
        """
        Evaluate and return (average_score, sample_outputs, sample_scores).
        """
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if not dataset:
            return 0.0, [], []

        data = list(dataset)
        if shuffle:
            random.shuffle(data)

        total_score = 0.0
        total_items = 0
        sample_outputs = []
        sample_scores = []

        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            batch_score, outputs, scores = self.evaluate_agent_batch(batch, current_prompt, max_workers=max_workers)
            total_score += batch_score * len(batch)
            total_items += len(batch)

            if len(sample_outputs) < max_examples:
                remaining = max_examples - len(sample_outputs)
                sample_outputs.extend(outputs[:remaining])
                sample_scores.extend(scores[:remaining])

        average_score = total_score / total_items if total_items > 0 else 0.0
        print(f"Evaluated agent with score: {average_score:.4f}")
        return average_score, sample_outputs, sample_scores
