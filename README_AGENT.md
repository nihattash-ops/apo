# APO Agent 优化框架

这是一个通用的 APO (Automatic Prompt Optimization) 框架，能够适配各种类型的 Agent，自动优化 Agent 的系统提示词。

## 核心特性

### 1. **Agent 接口（两种写法，使用方式统一）**
- 将算法与具体的任务逻辑完全分离
- **推荐**：Agent 只实现 `set_system_prompt(prompt)` 与 `process(input_data)`，**无须**实现 `__call__(input_data, system_prompt)`；框架通过 `as_rollout(agent)` 自动包装成两参调用。使用时先 `agent.set_system_prompt(prompt)`，再 **`agent(input_data)` 单参当作函数用**，无需感知内部两参调用。
- **可选**：直接提供可调用 `(input_data, system_prompt) -> output`（如纯函数或类实现两参 `__call__`），框架直接使用。

### 2. **数据集格式**
- 数据为 **list[dict]**，每条必须包含 `"input"` 和 `"output"`
- Agent 只处理 `input`；优化器用 `output` 作为标准答案做评估
- 通过 `dataset.load_dataset_from_json(path)` 加载

### 3. **通用优化算法**
- APO 算法独立于具体的 Agent 实现
- 支持全数据集和 Mini-batch 两种优化模式
- 并发评估提升效率

### 4. **灵活的评估机制**
- 每个Agent 自定义评估逻辑
- 支持任务特定的评分标准
- 可通过 `reward.txt`（或任务专属如 `cargo_reward.txt`）自定义评估提示词

## 当前算法实现概要

1. **数据**：`dataset = load_dataset_from_json(path)` → `list[dict]`，每条 `{"input": ..., "output": ...}`。Agent 只看到 `input`，评估时用 `output` 作为标准答案。
2. **Agent 接口**：优化器通过 `as_rollout(agent)` 得到「可调用 (input_data, system_prompt) -> output」。若 agent 实现的是 `set_system_prompt` + `process(input_data)`，则由框架包装；否则若 agent 本身可两参调用，则直接使用。
3. **评估**：对每条样本调用 `agent(item["input"], current_prompt)` 得到模型输出，再与 `item["output"]` 一起填入 `reward.txt` 模板，用 LLM 打 0~1 分，批内取平均。
4. **优化循环**：每轮（或每个 mini-batch）用当前 prompt 评估 → 用当前分数与样例构建 feedback → `PromptGenerator` 生成若干候选 prompt → 对每个候选评估 → 取分数最高的作为新的 current_prompt，并更新全局 best；迭代多轮直到结束或无明显提升。

## 项目结构

```
apo_agent_project/
├── agent_protocol.py          # as_rollout：将 set_system_prompt+process 或两参可调用 转为 (input_data, system_prompt) -> output
├── dataset.py                 # 数据集加载 load_dataset_from_json
├── qa_agent.py                # 问答 Agent 实现
├── cargo_agent.py             # 货运信息抽取 Agent 实现
├── evaluator_agent.py         # Agent 评估器（Rollout + reward 评估）
├── apo_optimizer_agent.py     # APO 优化器（传入 agent + dataset）
├── prompt_generator.py        # 提示词生成器
├── reward.txt                 # 通用评估提示词模板
├── cargo_reward.txt           # 货运任务评估提示词（可选）
├── main_agent.py              # 问答优化示例
├── train.py                   # 货运 Agent 优化入口
├── data/
│   ├── sample_data.json       # 问答示例数据
│   ├── sample_send_cargo_data.json  # 货运示例数据
│   └── shipper_memories.json  # 货运场景记忆
└── README_AGENT.md            # 本文档
```

## 核心模块说明

### 1. 数据集格式与加载

数据为 **list[dict]**，每条必须包含 `"input"` 和 `"output"`。Agent 只处理 `input`，优化器用 `output` 与模型输出对比做 LLM 评估。

```python
from dataset import load_dataset_from_json

# 从 JSON 加载（根节点为数组）
dataset = load_dataset_from_json("data/sample_data.json")
# 每条: {"input": ..., "output": ...}
# 例如 QA: {"input": "What is 2+2?", "output": "4"}
# 例如货运: {"input": {"shipper_id": "...", "user_messages": "..."}, "output": {"start": "...", "end": "...", "cargo_name": "..."}}
```

`input` / `output` 可以是任意类型（字符串、字典、列表等），只要与 Agent 和评估逻辑一致即可。

### 2. Agent 接口（二选一，推荐方式一）

**方式一（推荐）**：Agent **无须**实现 `__call__(input_data, system_prompt)`，只实现 `set_system_prompt(prompt)` 与 `process(input_data)`。框架通过 `as_rollout(agent)` 自动包装。使用时先设置 prompt，再 **`agent(input_data)` 单参当作函数用**。

```python
class MyAgent:
    def __init__(self, system_prompt, ...):
        self.system_prompt = system_prompt

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def process(self, input_data):
        # 使用当前 self.system_prompt 处理 input_data
        return ...  # 模型输出

    def __call__(self, input_data):
        """单参调用：agent(input_data) 即 process(input_data)。"""
        return self.process(input_data)

# 优化时直接传入实例，框架内部会包装
optimizer = APOOptimizerAgent(agent=MyAgent(...), dataset=dataset, ...)
# 使用：agent.set_system_prompt(best_prompt); result = agent(input_data)
```

**方式二**：直接提供可调用 `(input_data, system_prompt) -> output`（如纯函数或类实现两参 `__call__`），框架直接使用。

评估由框架通过 `reward.txt` 与 LLM 完成。

### 3. QAAgent / CargoAgent

- **QAAgent**：问答任务，实现 `set_system_prompt` + `process(input_data)`，可 `agent(input_data)` 单参调用。
- **CargoAgent**：货运信息抽取，同样实现 `set_system_prompt` + `process(input_data)`，可 `agent(input_data)` 单参调用。
- 评估由框架的 `reward.txt` / `cargo_reward.txt` 完成。

### 4. AgentEvaluator（评估器）

评估 Agent 在任务列表上的性能：
- 并发处理多个任务
- 批处理支持
- 返回详细评估结果

### 5. APOOptimizerAgent（优化器）

APO 优化器，接收 **agent**（纯函数或带协议的对象）和 **dataset**（list[dict] 含 input/output）：
- 内部将 agent 转为 Rollout 可调用 `(input_data, system_prompt) -> output`
- 迭代优化：评估当前 prompt → 生成候选 prompt → 评估候选 → 选最优
- 支持全数据集模式与 Mini-batch 模式

## 快速开始

### 基本使用（问答 Agent）

```python
from qa_agent import QAAgent
from dataset import load_dataset_from_json
from apo_optimizer_agent import APOOptimizerAgent
import os

# 1. 创建 Agent
agent = QAAgent(
    system_prompt="Answer the question directly and concisely.",
    model_name="deepseek-chat",
    api_key=os.getenv("APO_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 2. 加载数据集（list[dict]，每条含 "input" 和 "output"）
dataset = load_dataset_from_json("data/sample_data.json")

# 3. 创建优化器（方式 1: 直接传 dataset）
optimizer = APOOptimizerAgent(
    agent=agent,
    dataset=dataset,
    llm_model_name="deepseek-chat",
    api_key=os.getenv("APO_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 或者方式 2: 从路径加载
optimizer = APOOptimizerAgent.from_dataset_path(
    agent=agent,
    dataset_path="data/sample_data.json",
    llm_model_name="deepseek-chat",
    api_key=os.getenv("APO_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 4. 运行优化
best_prompt, best_score, history = optimizer.optimize(
    initial_prompt="Answer the question directly and concisely.",
    num_iterations=5,
    num_candidates=3,
    batch_size=4,
    shuffle=True,
    update_per_batch=True,
    max_workers=8
)

# 5. 使用优化后的提示词（设好 prompt 后 agent(input_data) 单参当作函数用）
agent.set_system_prompt(best_prompt)
answer = agent("What is the capital of France?")
```

### 创建自定义 Agent

推荐只实现 `set_system_prompt` + `process(input_data)`，无须感知两参调用；设好 prompt 后用 `agent(input_data)`：

```python
from openai import OpenAI
from dataset import load_dataset_from_json
from apo_optimizer_agent import APOOptimizerAgent

class SummarizationAgent:
    def __init__(self, system_prompt, model_name="gpt-3.5-turbo", api_key=None, base_url=None):
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def process(self, input_data):
        input_text = input_data if isinstance(input_data, str) else str(input_data)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Summarize: {input_text}"}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    def __call__(self, input_data):
        return self.process(input_data)

dataset = load_dataset_from_json("data/summarization_data.json")
agent = SummarizationAgent("Summarize in one sentence.", ...)
optimizer = APOOptimizerAgent(agent, dataset=dataset, ...)
best_prompt, best_score, history = optimizer.optimize(...)
# 使用：agent.set_system_prompt(best_prompt); agent(长文本)
```

### 支持任意 input/output 类型

数据集的 `input` / `output` 可以是任意类型（字符串、字典、列表等），只要与 Agent 可调用 `(input_data, system_prompt) -> output` 和评估提示词一致即可：

```python
# 例如：input 为字典（如货运场景）
dataset = [
    {"input": {"shipper_id": "260216", "user_messages": "发一票到新疆的货"}, "output": {"start": "上海浦东区", "end": "新疆伊宁市", "cargo_name": "塑料颗粒"}},
]
# 调用时 agent(item["input"], system_prompt)，即 input_data 为该 dict

# 例如：input 为字符串（问答）
dataset = [{"input": "What is 2+2?", "output": "4"}]
# 调用时 agent(item["input"], system_prompt)
```

只要 Agent 实现 `set_system_prompt` + `process(input_data)`（或直接两参可调用），并在 `reward.txt` 中约定如何根据 Output 与 Expected 打分即可。

## 优化模式

### 1. 全数据集模式（update_per_batch=False）

每次迭代使用完整任务列表评估，适合：
- 小规模任务集
- 需要精确评估
- 计算资源充足

### 2. Mini-batch 模式（update_per_batch=True）

每次迭代分批处理任务，每批后更新，适合：
- 大规模任务集
- 需要快速迭代
- 计算资源有限

## 配置参数

### Agent 参数（以 QAAgent/CargoAgent 为例）
- `system_prompt`: 初始系统提示词
- `model_name`: LLM 模型名称
- `api_key`: API 密钥
- `base_url`: API 基础 URL  

优化器单独指定评估模板：`reward_prompt_path`（如 `reward.txt` 或 `cargo_reward.txt`）。

### 优化器参数
- `dataset`: 数据集（list[dict]，每条含 `"input"` 和 `"output"`）；也可用 `from_dataset_path(..., dataset_path=...)` 从文件加载
- `initial_prompt`: 初始提示词
- `num_iterations`: 迭代次数（默认 5）
- `num_candidates`: 每次迭代生成的候选数（默认 3）
- `batch_size`: 批大小（默认 4）
- `shuffle`: 是否打乱数据（默认 True）
- `update_per_batch`: 是否每批更新（默认 True）
- `max_workers`: 最大并发数（默认 8）

## 自定义评估提示词

编辑 `reward.txt` 文件来自定义评估逻辑：

```
You are a strict evaluator.
Assign a score between 0 and 1 based on how well Output matches Expected.
The score must reflect similarity: identical answers = 1, completely different answers = 0.
if Output is partially correct, longest the answer is, the lower the score should be.
Return only JSON in the form: {"score": <number_between_0_and_1>}.

Output: {prediction}
Expected: {true_label}
```

## 数据集格式

数据集应为 JSON 数组，每个元素包含 `input` 和 `output` 字段：

```json
[
    {
        "input": "What is the capital of France?",
        "output": "Paris"
    },
    {
        "input": "What is 2 + 2?",
        "output": "4"
    }
]
```

可选字段：
- `task_type`: 任务类型标识
- `metadata`: 任意额外的元数据

```json
[
    {
        "input": "Hello",
        "output": "Hola",
        "task_type": "translation",
        "metadata": {
            "source_lang": "en",
            "target_lang": "es"
        }
    }
]
```

## 运行示例

```bash
# 设置 API 密钥
export APO_API_KEY="your-api-key"

# 问答 Agent 优化（使用 sample_data.json）
python main_agent.py

# 货运信息抽取 Agent 优化（使用 sample_send_cargo_data.json + cargo_reward.txt）
python train.py
```

## 架构优势

### 1. **完全解耦**
- APO 算法与具体任务分离
- 可以轻松切换不同的 Agent
- 算法复用性高

### 2. **高度可扩展**
- 支持任意类型的 Agent
- 支持任意数据类型的任务
- 自定义评估逻辑
- 灵活的提示词模板

### 3. **统一的数据格式**
- 数据集为 list[dict]，统一包含 input / output
- Agent 只消费 input；评估用 output 与模型输出对比
- input/output 可为任意类型（字符串、字典等）

### 4. **易于维护**
- 清晰的模块划分
- 统一的接口设计
- 完善的错误处理

### 5. **性能优化**
- 并发处理
- 批处理支持
- 可配置的资源使用

## 应用场景

1. **问答系统优化**
   - 优化问答 Agent 的提示词
   - 提升准确率

2. **文本摘要**
   - 优化摘要质量和长度
   - 平衡信息量和简洁性

3. **代码生成**
   - 优化代码风格和质量
   - 提升代码可读性

4. **翻译任务**
   - 优化翻译准确性
   - 保持语言风格

5. **对话系统**
   - 优化对话流畅度
   - 提升用户体验

6. **图像分类**
   - 优化图像识别提示词
   - 提升分类准确率

7. **音频处理**
   - 优化语音识别或合成
   - 提升音频质量

## 注意事项

1. 需要配置 LLM API 密钥
2. 大规模任务集建议使用 Mini-batch 模式
3. 评估提示词需要根据任务特点调整
4. 并发数应根据 API 限流设置调整
5. 数据集的 input 类型需与 Agent 的 `process(input_data)` 的 input_data 匹配，output 需与评估提示词（如 reward.txt）约定一致

## 扩展性

### 添加新的 Agent 类型

推荐：实现 `set_system_prompt(prompt)` 与 `process(input_data)`（可选 `__call__(self, input_data)` 便于单参使用），传给 `APOOptimizerAgent(agent=..., dataset=...)`。或直接提供两参可调用 `(input_data, system_prompt) -> output`。

### 自定义评估策略

通过 `reward.txt` 文件自定义 LLM 评估提示词即可；评估在框架层统一完成，Agent 无需实现评估逻辑。

### 支持新的任务类型

只需准备符合格式的数据集（list[dict] 含 input/output），并实现对应的 `process(input_data)` 与评估提示词（reward.txt）即可。可选在 JSON 中增加 `task_type`、`metadata` 等字段供 Agent 或评估逻辑使用。

## 便捷方法

设好 prompt 后统一用 **`agent(input_data)`** 单参调用；QAAgent 另有 `process_input(input_text)` 使用当前 `system_prompt`。

```python
# 使用优化得到的 prompt
agent.set_system_prompt(best_prompt)
answer = agent("What is the capital of France?")
# 或
answer = agent.process_input("What is the capital of France?")
```

## 许可证

本项目遵循 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！