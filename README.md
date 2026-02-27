# APO 优化框架

基于 APO (Automatic Prompt Optimization) 的系统提示词优化：算法接受一个 **`(input_data, system_prompt) -> output`** 的函数与数据集，迭代搜索更优的 `system_prompt`。支持多种数据形态、自定义 reward、以及通过 `make_rollout` 快速包装任意 Agent。

---

## 算法接口

- **Rollout**：可调用对象，签名为 `(input_data, system_prompt) -> output`（或带 `**kwargs` 的变体）。  
  算法内部对每条样本调用 `rollout(item 的 input, current_prompt, **rollout_kwargs)` 得到输出，再与标准答案一起评估。不关心 rollout 内部实现。  
  - 可用 **`make_rollout(agent, system_prompt_attr="system_prompt", run_method="_run")`** 将任意「可设置 system_prompt + 单次 run 方法」的 Agent 包成 rollout。
- **数据集**：`list[dict]`（或具相同语义的对象列表），默认每条包含 `"input"` 和 `"output"`。  
  - 可通过 **`input_key` / `output_key`** 配置为其他键名或属性名，以适配不同数据格式。  
  - `input`：传给 rollout 的输入。  
  - `output`：标准答案，用于与 rollout 输出一起做评估（或交给自定义 `reward_fn`）。
- **评估**：默认使用 reward 模板 + LLM 打分；也可传入 **`reward_fn(output, expected_output, item) -> float`** 做规则/精确匹配等自定义打分。

---

## 算法流程

1. **评估**  
   对给定 `current_prompt`，在（全量或当前批）数据上对每条调用 `rollout(item["input"], current_prompt)`，得到模型输出；将「模型输出」与「item["output"]」填入 **reward 模板**（如 `reward.txt`），用 LLM 打 0~1 分；批内取平均作为当前 prompt 的分数。

2. **反馈与候选生成**  
   用当前分数、当前 prompt、数据中的正确答案样例、模型输出样例拼成 **feedback**，交给 **PromptGenerator**；由 LLM 根据当前 prompt + feedback 生成若干条 **候选 prompt**（数量由 `num_candidates` 控制）。

3. **候选评估与更新**  
   对每个候选 prompt 在（全量或当前批）数据上同样通过 rollout 评估得分；取分数最高的候选作为下一轮的 `current_prompt`，并据此更新全局最优 prompt 与分数。

4. **迭代与模式**  
   - **全数据集模式**（`update_per_batch=False`）：每轮在整份数据集上评估当前 prompt、生成候选、再在整份数据上评估所有候选，取最优进入下一轮。  
   - **Mini-batch 模式**（`update_per_batch=True`）：每轮按 `batch_size` 切批；每批内对当前 prompt 评估 → 生成候选 → 仅在该批上评估候选 → 用该批最优更新 current_prompt，并更新全局最优。  
   可配置 `num_iterations`、`shuffle`、`max_workers`（并发数）等；若某轮（全量或批内）无提升且非首轮，可提前结束。

---

## 模块与数据

- **APOOptimizerAgent**：入口。构造时传入 `rollout`、`dataset`、LLM 配置、`reward_prompt_path`；可选 `input_key`、`output_key`、`reward_fn`、`rollout_kwargs`。`optimize(...)` 返回 `(best_prompt, best_score, history)`。
- **AgentEvaluator**：封装「rollout 调用 + 默认 LLM 打分或自定义 `reward_fn`」，支持可配置 input/output 键、`rollout_kwargs`、按批与并发。
- **PromptGenerator**：根据当前 prompt 与 feedback 调用 LLM 生成多条候选提示词（解析 `PROMPT:` 前缀）。
- **apo_protocols**：`Rollout` 类型、`get_input_from_item` / `get_output_from_item`、`make_rollout(agent, ...)`，便于适配各种 Agent 与数据格式。
- **数据集**：`load_dataset_from_json(path)` 得到 `list[dict]`，默认每条 `{"input": ..., "output": ...}`；键名可通过 `input_key`/`output_key` 与算法一致。

Reward 模板中占位符 `{prediction}`、`{true_label}` 对应「模型输出」与「item["output"]」；LLM 返回的 JSON 需包含 `score`（0~1）。

---

## 使用方式

调用方提供 **rollout** 或使用 **make_rollout** 包装 Agent，无需固定形态。例如：

```python
# 方式一：make_rollout（推荐，适合多数 Agent）
from apo_protocols import make_rollout
from apo_optimizer_agent import APOOptimizerAgent
from dataset import load_dataset_from_json
agent = YourAgent(system_prompt="...", ...)
rollout = make_rollout(agent, system_prompt_attr="system_prompt", run_method="_run")

# 方式二：手写 rollout
def rollout(input_data, system_prompt):
    agent.system_prompt = system_prompt
    return agent._run(input_data)

# 可选：自定义 reward、不同数据键名、rollout 透传参数
optimizer = APOOptimizerAgent(
    rollout=rollout, dataset=dataset, llm_model_name=..., api_key=..., base_url=...,
    reward_prompt_path="reward.txt",
    input_key="input", output_key="output",  # 数据条键名
    reward_fn=None,   # 或 (output, expected, item) -> 0~1 的自定义打分
    rollout_kwargs={} # 调用 rollout 时透传的额外参数
)
best_prompt, best_score, history = optimizer.optimize(initial_prompt=..., num_iterations=5, num_candidates=3, ...)
```

项目中的 `train.py`、 为示例：如何用 `make_rollout` 或手写 rollout，以及如何调用 `optimize`。

---

## 项目结构

```
├── apo_optimizer_agent.py   # 优化器：rollout + dataset
├── apo_protocols.py        # Rollout 类型、make_rollout、get_input/output_from_item
├── evaluator_agent.py       # 评估器
├── prompt_generator.py     # 候选提示词生成
├── dataset.py              # load_dataset_from_json
├── reward.txt / cargo_reward.txt
├── train.py                
├── train_send_cargo.py     
├── cargo_agent.py
├── send_cargo_extract_rollout.py
└── data/
```

---

## 运行示例

```bash
export APO_API_KEY="your-api-key"
python train.py             
```

算法侧不依赖具体 Agent；只要传入的 rollout 满足 `(input_data, system_prompt) -> output` 且数据集与 reward 模板格式一致即可。
