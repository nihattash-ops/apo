"""
APO Optimizer for Agents: Optimizes agent system prompts using the APO algorithm.
支持多种数据形态（可配置 input/output 键名）、可选自定义 reward、rollout 透传 kwargs。
"""

from evaluator_agent import AgentEvaluator
from prompt_generator import PromptGenerator
from dataset import load_dataset_from_json
from apo_protocols import get_input_from_item, get_output_from_item
import random
from typing import List, Dict, Any, Callable, Optional

from apo_protocols import Rollout


def _extract_display_output(x: Any) -> str:
    """若 rollout 返回 dict 且含 'output'，仅用 output 展示；否则整条转 str。"""
    if x is None:
        return "N/A"
    if isinstance(x, dict) and "output" in x:
        return str(x["output"])
    return str(x)


class APOOptimizerAgent:
    """
    Optimizes agent system prompts using the APO (Automatic Prompt Optimization) algorithm.
    只接受 rollout 函数，不关心其内部实现；可配置数据键名、自定义 reward、rollout 额外参数。
    """

    def __init__(self, rollout: Rollout, dataset: List[Dict[str, Any]], llm_model_name="gpt-3.5-turbo",
                 api_key=None, base_url=None, reward_prompt_path="reward.txt",
                 input_key: str = "input",
                 output_key: str = "output",
                 reward_fn: Optional[Callable[[Any, Any, Any], float]] = None,
                 rollout_kwargs: Optional[Dict[str, Any]] = None,
                 feedback_template_path: Optional[str] = None,
                 generate_prompt_path: Optional[str] = None):
        """
        Initialize the APO Optimizer.

        Args:
            rollout: 可调用 (input_data, system_prompt, **kwargs) -> output（可为 dict 含 "output" 键，仅用该键打分）
            dataset: List of dicts（或具 input_key/output_key 结构的对象列表）
            llm_model_name: Name of the LLM model for prompt generation and evaluation
            api_key: API key for the LLM service
            base_url: Base URL for the LLM API
            reward_prompt_path: Path to the reward evaluation prompt template（reward_fn 为 None 时使用）
            input_key: 数据条中「输入」的键名
            output_key: 数据条中「期望输出」的键名
            reward_fn: 可选。自定义打分函数 (output, expected_output, item) -> float in [0,1]
            rollout_kwargs: 可选。调用 rollout 时透传的额外参数
            feedback_template_path: 可选。feedback 模板文件，占位符 {score} {current_prompt} {samples_block}
            generate_prompt_path: 可选。候选提示词生成指令模板文件，占位符 {num_candidates} {base_prompt} {feedback}
        """
        self._rollout = rollout
        self.input_key = input_key
        self.output_key = output_key
        self.feedback_template_path = feedback_template_path
        self.evaluator = AgentEvaluator(
            rollout,
            llm_model_name=llm_model_name,
            api_key=api_key,
            base_url=base_url,
            reward_prompt_path=reward_prompt_path,
            input_key=input_key,
            output_key=output_key,
            reward_fn=reward_fn,
            rollout_kwargs=rollout_kwargs,
        )
        self.generator = PromptGenerator(
            model_name=llm_model_name, api_key=api_key, base_url=base_url,
            generate_prompt_path=generate_prompt_path,
        )
        self.dataset = dataset
        self.best_system_prompt = ""
        self.best_score = -1.0
        self.history = []

    @classmethod
    def from_dataset_path(cls, rollout: Rollout, dataset_path, llm_model_name="gpt-3.5-turbo",
                         api_key=None, base_url=None, reward_prompt_path="reward.txt",
                         input_key: str = "input", output_key: str = "output",
                         reward_fn=None, rollout_kwargs=None,
                         feedback_template_path=None, generate_prompt_path=None):
        dataset = load_dataset_from_json(dataset_path)
        return cls(rollout, dataset, llm_model_name, api_key, base_url, reward_prompt_path,
                   input_key=input_key, output_key=output_key, reward_fn=reward_fn, rollout_kwargs=rollout_kwargs,
                   feedback_template_path=feedback_template_path, generate_prompt_path=generate_prompt_path)

    def _build_feedback(
        self,
        current_score,
        current_prompt,
        batch: List[Dict[str, Any]] = None,
        model_outputs=None,
        max_examples=5,
    ):
        """
        按样本对齐构造 feedback：每条样本的 Input / Correct output / Model output 一一对应（无 memory）。
        若配置了 feedback_template_path 则从文件加载模板，占位符：score, current_prompt, samples_block。
        """
        examples = batch if batch is not None else self.dataset
        input_list = []
        correct_list = []
        for item in examples:
            inp = get_input_from_item(item, self.input_key)
            out = get_output_from_item(item, self.output_key)
            if out is not None:
                input_list.append(str(inp) if inp is not None else "N/A")
                correct_list.append(str(out))
            if len(correct_list) >= max_examples:
                break
        model_outputs = model_outputs or []
        model_list = [_extract_display_output(x) for x in model_outputs[:max_examples]]
        n = max(len(input_list), len(correct_list), len(model_list))
        if n == 0:
            samples_block = "（无样例）"
        else:
            lines = []
            for i in range(n):
                input_i = input_list[i] if i < len(input_list) else "N/A"
                correct_i = correct_list[i] if i < len(correct_list) else "N/A"
                model_i = model_list[i] if i < len(model_list) else "N/A"
                lines.append(
                    f"--- 样本 {i+1} ---\n"
                    f"  Input: {input_i}\n"
                    f"  Correct output: {correct_i}\n"
                    f"  Model output:  {model_i}"
                )
            samples_block = "\n\n".join(lines)
        if self.feedback_template_path:
            try:
                with open(self.feedback_template_path, "r", encoding="utf-8") as f:
                    tpl = f.read().strip()
                return tpl.format(
                    score=f"{current_score:.4f}",
                    current_prompt=current_prompt,
                    samples_block=samples_block,
                )
            except FileNotFoundError:
                pass
            except KeyError:
                pass
        return (
            f"Score: {current_score:.4f}（满分 1.0）\n"
            f"Current system prompt: {current_prompt}\n\n"
            "请根据以下「逐条对齐」的样本改进提示词：每条样本的 Input 为输入，Correct output 为期望输出，Model output 为当前模型输出。\n\n"
            f"{samples_block}"
        )

    def _print_outputs_and_scores(self, outputs, scores):
        if not outputs:
            return
        scores_str = ",".join(f"{s:.2f}" if isinstance(s, (int, float)) else "?" for s in (scores or [])[:20])
        print(f"  scores: [{scores_str}]")
        for idx, output in enumerate(outputs[:5], start=1):
            disp = _extract_display_output(output)
            if len(disp) > 200:
                disp = disp[:200] + "..."
            print(f"  out[{idx}]: {disp}")
        if len(outputs) > 5:
            print(f"  ... +{len(outputs) - 5} more")

    def _print_feedback(self, feedback: str, max_len: int = 400):
        print("  feedback:", feedback[:max_len] + "..." if len(feedback) > max_len else feedback)

    def _print_candidates(self, candidate_prompts: List[str]):
        for j, c in enumerate(candidate_prompts, 1):
            print(f"  candidate[{j}]: {c[:80]}..." if len(c) > 80 else f"  candidate[{j}]: {c}")

    def _write_prompt_to_log(self, log_file, label: str, prompt: str):
        if log_file is None:
            return
        try:
            log_file.write(f"\n{'='*60}\n{label}\n{'='*60}\n")
            log_file.write(prompt)
            log_file.write("\n")
            log_file.flush()
        except Exception as e:
            print(f"Warning: write prompt log failed: {e}")
    
    def optimize(
        self,
        initial_prompt,
        num_iterations=5,
        num_candidates=3,
        batch_size=4,
        shuffle=True,
        update_per_batch=True,
        max_workers=8,
        prompt_log_path: Optional[str] = None,
        feedback_max_examples: int = 5,
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
            prompt_log_path: 可选。将每轮当前提示词、feedback、候选写入该文件（UTF-8）
            feedback_max_examples: 构造 feedback 时取几条样本做「逐条对齐」，默认 5

        Returns:
            Tuple of (best_prompt, best_score, history)
        """
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if feedback_max_examples is None or feedback_max_examples < 1:
            feedback_max_examples = 1

        current_prompt = initial_prompt
        self.best_system_prompt = initial_prompt
        print(f"Starting optimization with initial system prompt (len={len(initial_prompt)})")

        log_file = None
        if prompt_log_path:
            try:
                log_file = open(prompt_log_path, "w", encoding="utf-8")
                log_file.write("# APO 训练过程全部提示词（UTF-8 编码）\n")
                self._write_prompt_to_log(log_file, "初始提示词 initial_prompt", initial_prompt)
            except Exception as e:
                print(f"Warning: cannot open prompt log {prompt_log_path}: {e}")
                log_file = None

        try:
            if not update_per_batch:
                return self._optimize_full_dataset(
                    current_prompt, num_iterations, num_candidates, batch_size, shuffle, max_workers,
                    log_file=log_file, feedback_max_examples=feedback_max_examples)
            return self._optimize_mini_batch(
                current_prompt, num_iterations, num_candidates, batch_size, shuffle, max_workers,
                log_file=log_file, feedback_max_examples=feedback_max_examples)
        finally:
            if log_file:
                try:
                    self._write_prompt_to_log(log_file, "最终最优提示词 final_best_prompt", self.best_system_prompt)
                    log_file.close()
                except Exception:
                    pass

    def _optimize_full_dataset(
        self,
        current_prompt,
        num_iterations,
        num_candidates,
        batch_size,
        shuffle,
        max_workers,
        log_file=None,
        feedback_max_examples=5,
    ):
        """Optimize using full dataset evaluation."""
        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")
            current_score, model_outputs, model_scores = self.evaluator.evaluate_agent_with_details(
                self.dataset, current_prompt, batch_size=batch_size, max_workers=max_workers,
                shuffle=shuffle, max_examples=len(self.dataset))
            self.history.append({"iteration": i+1, "prompt": current_prompt, "score": current_score})
            if log_file:
                self._write_prompt_to_log(log_file, f"迭代 {i+1}/{num_iterations} 当前提示词 (score={current_score:.4f})", current_prompt)
            print(f"[iter {i+1}/{num_iterations}] score={current_score:.4f}")
            self._print_outputs_and_scores(model_outputs, model_scores)
            if current_score > self.best_score:
                self.best_score, self.best_system_prompt = current_score, current_prompt
                print(f"  -> new best score={self.best_score:.4f}")
            feedback = self._build_feedback(
                current_score, current_prompt, model_outputs=model_outputs, max_examples=feedback_max_examples)
            if log_file:
                self._write_prompt_to_log(log_file, f"迭代 {i+1}/{num_iterations} feedback", feedback)
            self._print_feedback(feedback)
            candidate_prompts = self.generator.generate_candidates(current_prompt, num_candidates, feedback=feedback)
            if log_file:
                for j, c in enumerate(candidate_prompts, 1):
                    self._write_prompt_to_log(log_file, f"迭代 {i+1} 候选 candidate[{j}]", c)
            self._print_candidates(candidate_prompts)
            best_candidate_score, best_candidate_prompt = current_score, current_prompt
            for j, candidate in enumerate(candidate_prompts):
                candidate_score = self.evaluator.evaluate_agent(
                    self.dataset, candidate, batch_size=batch_size, max_workers=max_workers, shuffle=shuffle)
                self.history.append({"iteration": i+1, "candidate_id": j+1, "prompt": candidate, "score": candidate_score})
                if candidate_score > best_candidate_score:
                    best_candidate_score, best_candidate_prompt = candidate_score, candidate
            if best_candidate_score > self.best_score:
                self.best_score, self.best_system_prompt = best_candidate_score, best_candidate_prompt
                print(f"  -> new best score={self.best_score:.4f}")
            if best_candidate_score <= current_score and i > 0:
                print("  -> no improvement, stop.")
                break
            current_prompt = best_candidate_prompt
        print("\n--- Optimization Finished ---")
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
        log_file=None,
        feedback_max_examples=5,
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
                current_score, model_outputs, model_scores = self.evaluator.evaluate_agent_batch(
                    batch, current_prompt, max_workers=max_workers)
                self.history.append({"iteration": i+1, "batch_id": batch_count, "prompt": current_prompt, "score": current_score})
                if log_file:
                    self._write_prompt_to_log(log_file, f"迭代 {i+1} batch {batch_count} 当前提示词 (score={current_score:.4f})", current_prompt)
                print(f"[iter {i+1} batch {batch_count}] score={current_score:.4f}")
                self._print_outputs_and_scores(model_outputs, model_scores)
                if current_score > self.best_score:
                    self.best_score, self.best_system_prompt = current_score, current_prompt
                    improved_in_iteration = True
                    print(f"  -> new best score={self.best_score:.4f}")
                feedback = self._build_feedback(
                    current_score, current_prompt, batch=batch, model_outputs=model_outputs, max_examples=feedback_max_examples)
                if log_file:
                    self._write_prompt_to_log(log_file, f"迭代 {i+1} batch {batch_count} feedback", feedback)
                self._print_feedback(feedback)
                candidate_prompts = self.generator.generate_candidates(current_prompt, num_candidates, feedback=feedback)
                if log_file:
                    for j, c in enumerate(candidate_prompts, 1):
                        self._write_prompt_to_log(log_file, f"迭代 {i+1} batch {batch_count} 候选 candidate[{j}]", c)
                self._print_candidates(candidate_prompts)
                best_candidate_score, best_candidate_prompt = current_score, current_prompt
                for j, candidate in enumerate(candidate_prompts):
                    candidate_score, _, _ = self.evaluator.evaluate_agent_batch(batch, candidate, max_workers=max_workers)
                    self.history.append({"iteration": i+1, "batch_id": batch_count, "candidate_id": j+1, "prompt": candidate, "score": candidate_score})
                    if candidate_score > best_candidate_score:
                        best_candidate_score, best_candidate_prompt = candidate_score, candidate
                if best_candidate_score > self.best_score:
                    self.best_score, self.best_system_prompt = best_candidate_score, best_candidate_prompt
                    improved_in_iteration = True
                    print(f"  -> new best score={self.best_score:.4f}")
                current_prompt = best_candidate_prompt
            if not improved_in_iteration and i > 0:
                print("  -> no improvement, stop.")
                break
        print("\n--- Optimization Finished ---")
        print(f"Final Best Score: {self.best_score:.4f}")
        return self.best_system_prompt, self.best_score, self.history