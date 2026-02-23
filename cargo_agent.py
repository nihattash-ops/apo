"""
Cargo Agent: 只需实现 set_system_prompt(prompt) 与 process(input_data)。
框架通过 as_rollout(agent) 包装；使用时可用 agent(input_data) 单参调用。
"""
from openai import OpenAI
import json
import os

API_KEY = os.getenv("APO_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"


class CargoAgent:
    """货运信息抽取 Agent：实现 set_system_prompt + process(input_data)，可 agent(input_data) 当作函数用。"""

    def __init__(self, system_prompt, model_name="gpt-3.5-turbo", api_key=None, base_url=None):
        if not api_key:
            raise ValueError("API key is required.")
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        path = "data/shipper_memories.json"
        with open(path, 'r', encoding='utf-8') as f:
            memories = json.load(f)
        self.shipper_dict = {item["shipper_id"]: item["memory"] for item in memories}

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def process(self, input_data):
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

    def __call__(self, input_data):
        """单参调用：使用当前已设置的 system_prompt，agent(input_data) 即 process(input_data)。"""
        return self.process(input_data)


if __name__ == "__main__":
    with open("data/sample_send_cargo_data.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)
    item = tasks[0]
    input_data = item["input"]
    system_prompt = """根据用户的输入及记忆，抽取装货地点、卸货地点、货物名称。并以JSON格式输出。
    输出格式
    {
        "start": "装货地",
        "end": "卸货地",
        "cargo_name": "货物名称"
    }
    """
    agent = CargoAgent(system_prompt=system_prompt, model_name=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL)
    agent.set_system_prompt(system_prompt)
    result = agent(input_data)
    print(result)
