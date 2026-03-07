"""
Cargo Agent：货运信息抽取。
算法只接受 (input_data, system_prompt) -> output 的函数。调用方定义 rollout，内设 agent.system_prompt 并调用 agent._process(input_data)。
"""
from openai import OpenAI
import json
import os

API_KEY = os.getenv("APO_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"


class CargoAgent:
    """货运信息抽取 Agent。提供 system_prompt 与 _process(input_data)，由调用方在 rollout 中组合使用。"""

    def __init__(self, system_prompt="", model_name="gpt-3.5-turbo", api_key=None, base_url=None):
        if not api_key:
            raise ValueError("API key is required.")
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        path = "data/shipper_memories.json"
        with open(path, 'r', encoding='utf-8') as f:
            memories = json.load(f)
        self.shipper_dict = {item["shipper_id"]: item["memory"] for item in memories}


    def _run(self, input_data):
        """处理 input_data（如含 shipper_id、user_messages 的 dict），使用当前 self.system_prompt。"""
        inp = input_data if isinstance(input_data, dict) else {}
        shipper_id = inp.get("shipper_id", "")
        memory = self.shipper_dict.get(shipper_id, "")
        user_messages = inp.get("user_messages", "")
        input_text = "用户输入的消息是：" + str(user_messages) + "\n" + "记忆是：" + str(memory)
        return self._call_llm(input_text)

    def _call_llm(self, input_text):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error processing input: {e}")
            return None
