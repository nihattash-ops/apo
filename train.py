"""
货运 Agent APO 训练。
Agent 只需实现 set_system_prompt(prompt) + process(input_data)，由 as_rollout 自动包装；
使用时 agent.set_system_prompt(prompt) 后 agent(input_data) 单参当作函数用。
"""
import os
from dataset import load_dataset_from_json
from apo_optimizer_agent import APOOptimizerAgent
from cargo_agent import CargoAgent

# Configuration
API_KEY = os.getenv("APO_API_KEY")
LLM_MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
DATASET_PATH = "data/sample_send_cargo_data.json"
OPTIMIZED_PROMPT_PATH = "optimized_prompt.txt"

initial_system_prompt = """抽取装货地点、卸货地点、货物名称。"""
cargo_agent = CargoAgent(system_prompt=initial_system_prompt, model_name=LLM_MODEL, api_key=API_KEY, base_url=BASE_URL)

print("=" * 60)
print("Creating Agent (set_system_prompt + process，使用时 agent(input_data) 单参)")
print("=" * 60)
print(f"Initial System Prompt: '{initial_system_prompt[:60]}...'")

# Load dataset
print("\n" + "=" * 60)
print("Loading Dataset")
print("=" * 60)
dataset = load_dataset_from_json(DATASET_PATH)
print(f"Loaded {len(dataset)} items from {DATASET_PATH}")
print("\nExample items (input/output keys):")
for i, item in enumerate(dataset[:3], 1):
    print(f"  {i}. input keys: {list(item.get('input', {}).keys()) if isinstance(item.get('input'), dict) else '...'}, output: {str(item.get('output', ''))[:50]}...")

# APO optimizer（传入 agent，内部 as_rollout 将 set_system_prompt+process 包装为两参调用）
print("\n" + "=" * 60)
print("Initializing APO Optimizer")
print("=" * 60)
optimizer = APOOptimizerAgent(
    agent=cargo_agent,
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
        initial_prompt=initial_system_prompt,
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

    # 使用：set_system_prompt 后 agent(input_data) 单参当作函数用
    print("\n" + "=" * 60)
    print("Testing Optimized Agent")
    print("=" * 60)
    cargo_agent.set_system_prompt(best_prompt)
    print(f"Optimized System Prompt: '{best_prompt}'")
    print("\nTest Results on sample items (agent(input_data)):")
    for i, item in enumerate(dataset[:3], 1):
        output = cargo_agent(item["input"])
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
