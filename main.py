from apo_optimizer import APOOptimizer
import os

API_KEY = os.getenv("APO_API_KEY")
LLM_MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

def main():
    dataset_path = "data/sample_data.json"
    initial_prompt = "Answer the question directly and concisely."
    num_iterations = 5
    num_candidates = 3
    batch_size = 4
    shuffle = True
    update_per_batch = True
    max_workers = 8

    print("Starting APO Optimization Process...")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Dataset: {dataset_path}")
    print(f"Initial Prompt: '{initial_prompt}'")
    print(
        "Iterations: {0}, Candidates per iteration: {1}, Batch size: {2}, "
        "Shuffle: {3}, Update per batch: {4}, Max workers: {5}"
        .format(num_iterations, num_candidates, batch_size, shuffle, update_per_batch, max_workers)
    )

    try:
        if not API_KEY:
            raise ValueError("APO_API_KEY is not set in environment.")
        optimizer = APOOptimizer(
            dataset_path=dataset_path,
            llm_model_name=LLM_MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        best_prompt, best_score, history = optimizer.optimize(
            initial_prompt=initial_prompt,
            num_iterations=num_iterations,
            num_candidates=num_candidates,
            batch_size=batch_size,
            shuffle=shuffle,
            update_per_batch=update_per_batch,
            max_workers=max_workers,
        )

        print("\n====================================")
        print("APO Optimization Summary")
        print("====================================")
        print(f"Final Best Prompt: '{best_prompt}'")
        print(f"Final Best Score: {best_score:.4f}")
        print("\nOptimization History (showing only best prompt per iteration):")
        for entry in history:
            if "candidate_id" not in entry:
                iteration = entry.get("iteration")
                score = entry.get("score")
                prompt = entry.get("prompt", "")
                print(
                    "  Iteration {0}: Score={1:.4f}, Prompt='{2}...'"
                    .format(iteration, score, prompt[:70])
                )

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
