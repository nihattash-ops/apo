import os
from openai import OpenAI
from typing import Optional



class PromptGenerator:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None, base_url=None,
                 generate_prompt_path: Optional[str] = None):
        if not api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.generate_prompt_path = generate_prompt_path

    def _load_instruction(self, base_prompt: str, num_candidates: int, feedback: Optional[str]) -> str:
        """若配置了 generate_prompt_path 则从文件加载模板，占位符：num_candidates, base_prompt, feedback。"""
        if self.generate_prompt_path:
            try:
                with open(self.generate_prompt_path, "r", encoding="utf-8") as f:
                    tpl = f.read().strip()
                return tpl.format(
                    num_candidates=num_candidates,
                    base_prompt=base_prompt,
                    feedback=feedback or "（无）",
                )
            except FileNotFoundError:
                pass
            except KeyError:
                pass
        instruction = (
            f"You are an expert prompt engineer. Your goal is to generate {num_candidates} variations of the following prompt to improve its performance for a given task. "
            "The variations should be distinct and explore different phrasing, emphasis, or additional instructions. "
            "Each variation should be on a new line, prefixed with 'PROMPT:'.\n\nOriginal Prompt: {base_prompt}"
        ).format(base_prompt=base_prompt)
        if feedback:
            instruction += f"\n\nConsider the following feedback: {feedback}"
        return instruction

    def generate_candidates(self, base_prompt, num_candidates=3, feedback=None):
        """
        根据基础提示词和可选的反馈，生成多个候选提示词。
        若配置了 generate_prompt_path 则从该文件加载指令模板。
        支持按 PROMPT: 分块解析（多行候选）。
        """
        candidate_prompts = []
        instruction = self._load_instruction(base_prompt, num_candidates, feedback)

        try:
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "system", "content": instruction}],
                "temperature": 0.7,
                "n": 1,
            }
            if os.environ.get("LLM_NAME") == "qwen":
                kwargs["extra_body"] = {"result_format": "message"}
            response = self.client.chat.completions.create(**kwargs)
            generated_text = response.choices[0].message.content.strip()

            # 支持多行候选：按 PROMPT: 分割，每块为一条候选
            for block in generated_text.split("PROMPT:"):
                block = block.strip()
                if block:
                    candidate_prompts.append(block)
            # 若无 PROMPT: 则回退到按行解析
            if not candidate_prompts:
                for line in generated_text.split("\n"):
                    if line.strip().startswith("PROMPT:"):
                        candidate_prompts.append(line.strip()[len("PROMPT:"):].strip())

            while len(candidate_prompts) < num_candidates:
                candidate_prompts.append(f"Rephrased: {base_prompt}")

            return candidate_prompts[:num_candidates]

        except Exception as e:
            print(f"Error generating candidate prompts: {e}")
            return [base_prompt] * num_candidates
