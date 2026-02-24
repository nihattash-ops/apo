"""
货运 Agent APO 训练。
算法只接受 (input_data, system_prompt) -> output 的 rollout；可用 make_rollout(agent) 或手写函数包装任意 Agent。
"""
import os
from dataset import load_dataset_from_json
from apo_optimizer_agent import APOOptimizerAgent
from apo_protocols import make_rollout
from cargo_agent import CargoAgent

# Configuration
API_KEY = os.getenv("APO_API_KEY")
LLM_MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
DATASET_PATH = "data/sample_send_cargo_data.json"
OPTIMIZED_PROMPT_PATH = "optimized_prompt.txt"

system_prompt = """抽取装货地点、卸货地点、货物名称。"""
agent = CargoAgent(system_prompt=system_prompt, model_name=LLM_MODEL, api_key=API_KEY, base_url=BASE_URL)

# 方式一：使用 make_rollout，适合任何具有 system_prompt 属性 + _run(input_data) 的 Agent
rollout = make_rollout(agent, system_prompt_attr="system_prompt", run_method="_run")

# 方式二：手写 rollout（算法同样接受）
# def rollout(input_data, system_prompt):
#     agent.system_prompt = system_prompt
#     return agent._run(input_data)


print("=" * 60)
print("Rollout: (input_data, system_prompt) -> output，算法只接受此函数")
print("=" * 60)
print(f"Initial System Prompt: '{system_prompt[:60]}...'")

# Load dataset
print("\n" + "=" * 60)
print("Loading Dataset")
print("=" * 60)
dataset = load_dataset_from_json(DATASET_PATH)
print(f"Loaded {len(dataset)} items from {DATASET_PATH}")
print("\nExample items (input/output keys):")
for i, item in enumerate(dataset[:3], 1):
    print(f"  {i}. input keys: {list(item.get('input', {}).keys()) if isinstance(item.get('input'), dict) else '...'}, output: {str(item.get('output', ''))[:50]}...")

# APO 优化器只接受 (input_data, system_prompt) -> output 的函数
print("\n" + "=" * 60)
print("Initializing APO Optimizer")
print("=" * 60)
optimizer = APOOptimizerAgent(
    rollout=rollout,
    dataset=dataset,
    llm_model_name=LLM_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    reward_prompt_path="cargo_reward.txt",
)

num_iterations = 2
num_candidates = 3
batch_size = 8
shuffle = True
update_per_batch = True
max_workers = 4

print(f"LLM Model: {LLM_MODEL}")
print(f"Dataset: {DATASET_PATH}")
print(f"Number of items: {len(dataset)}")
print(f"Iterations: {num_iterations}, Candidates: {num_candidates}, Batch size: {batch_size}")
print(f"Update per batch: {update_per_batch}, Max workers: {max_workers}")

# Run optimization
print("\n" + "=" * 60)
print("Starting APO Optimization")
print("=" * 60)

try:
    best_prompt, best_score, history = optimizer.optimize(
        initial_prompt=system_prompt,
        num_iterations=num_iterations,
        num_candidates=num_candidates,
        batch_size=batch_size,
        shuffle=shuffle,
        update_per_batch=update_per_batch,
        max_workers=max_workers,
    )

    # Summary & save
    print("\n" + "=" * 60)
    print("APO Optimization Summary")
    print("=" * 60)
    print(f"Final Best System Prompt: '{best_prompt}'")
    print(f"Final Best Score: {best_score:.4f}")
    with open(OPTIMIZED_PROMPT_PATH, "w", encoding="utf-8") as f:
        f.write(best_prompt)
    print(f"\nOptimized prompt saved to: {OPTIMIZED_PROMPT_PATH}")

    print("\nOptimization History (best prompt per iteration):")
    for entry in history:
        if "candidate_id" not in entry:
            iteration = entry.get("iteration")
            score = entry.get("score")
            prompt = entry.get("prompt", "")
            print(f"  Iteration {iteration}: Score={score:.4f}, Prompt='{prompt[:70]}...'")

    # 使用 函数(input_data, best_prompt) 测试
    print("\n" + "=" * 60)
    print("Testing Optimized Prompt")
    print("=" * 60)
    print(f"Optimized System Prompt: '{best_prompt}'")
    print("\nTest Results (rollout(input_data, best_prompt)):")
    for i, item in enumerate(dataset[:3], 1):
        output = rollout(item["input"], best_prompt)
        print(f"\n  Item {i}:")
        print(f"    Input: {item['input']}")
        print(f"    Expected: {item['output']}")
        print(f"    Output: {output}")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
