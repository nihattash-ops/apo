import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

class Evaluator:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None, base_url=None):
        if not api_key:
            raise ValueError("API key is required. Provide it from main.py.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _get_llm_response(self, prompt, input_text):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return None

    def _predict_one(self, prompt, input_text):
        return self._get_llm_response(prompt, input_text)

    def _get_item_score(self, prediction, true_label):
        # Read the prompt template from reward.txt
        try:
            with open('reward.txt', 'r', encoding='utf-8') as f:
                item_prompt_template = f.read()
            
            # Replace placeholders with actual values
            item_prompt = item_prompt_template.format(prediction=prediction, true_label=true_label)
        except FileNotFoundError:
            # Fallback to the original prompt if reward.txt is not found
            item_prompt = (
                "You are a strict evaluator.\n"
                "Assign a score between 0 and 1 based on how well Output matches Expected.\n"
                "The score must reflect similarity: identical answers = 1, completely different answers = 0.\n"
                "if Output is partially correct, longest the asnwer is , the lower the score should be.\n"
                "Return only JSON in the form: {\"score\": <number_between_0_and_1>}.\n\n"
                f"Output: {prediction}\nExpected: {true_label}"
            )
        response = self._get_llm_response("You are a strict evaluator.", item_prompt)
        if not response:
            return None
        try:
            data = json.loads(response)
            score = data.get("score", None)
            if not isinstance(score, (int, float)):
                return None
            score = float(score)
            if 0.0 <= score <= 1.0:
                return score
            return None
        except Exception:
            return None

    def evaluate_prompt_batch_with_details(self, prompt, batch, max_workers=8):
        if not batch:
            return 0.0, [], [], []
        if max_workers is None or max_workers <= 0:
            max_workers = 1
        worker_count = min(max_workers, len(batch))

        results = [None] * len(batch)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {}
            for idx, item in enumerate(batch):
                input_text = item.get("input")
                future = executor.submit(self._predict_one, prompt, input_text)
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error in prediction: {e}")
                    results[idx] = None

        predictions = []
        true_labels = []
        for idx, item in enumerate(batch):
            pred = results[idx]
            if pred is not None:
                predictions.append(pred)
                true_labels.append(item.get("output"))

        if not true_labels:
            return 0.0, [], [], []

        if max_workers is None or max_workers <= 0:
            max_workers = 1
        score_worker_count = min(max_workers, len(true_labels))
        scores = [None] * len(true_labels)
        with ThreadPoolExecutor(max_workers=score_worker_count) as executor:
            future_to_idx = {}
            for idx, (pred, true) in enumerate(zip(predictions, true_labels)):
                future = executor.submit(self._get_item_score, pred, true)
                future_to_idx[future] = idx
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result()
                except Exception as e:
                    print(f"Error in scoring: {e}")
                    scores[idx] = None

        valid_scores = [s for s in scores if isinstance(s, (int, float))]
        if not valid_scores:
            print("Warning: item score parsing failed; treating this batch as 0 reward.")
            return 0.0, predictions, true_labels, scores
        batch_reward = sum(valid_scores) / len(valid_scores)
        return batch_reward, predictions, true_labels, scores

    def evaluate_prompt_batch(self, prompt, batch, max_workers=8):
        batch_reward, _, _, _ = self.evaluate_prompt_batch_with_details(
            prompt,
            batch,
            max_workers=max_workers,
        )
        return batch_reward

    def evaluate_prompt_with_details(
        self,
        prompt,
        dataset,
        batch_size=4,
        max_workers=8,
        shuffle=False,
        max_examples=5,
    ):
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not dataset:
            return 0.0, [], [], []

        data = list(dataset)
        if shuffle:
            random.shuffle(data)

        total_weighted_correct = 0.0
        total_items = 0
        sample_predictions = []
        sample_true_labels = []
        sample_scores = []

        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            batch_reward, predictions, true_labels, scores = self.evaluate_prompt_batch_with_details(
                prompt,
                batch,
                max_workers=max_workers,
            )
            total_weighted_correct += batch_reward * len(batch)
            total_items += len(batch)

            if len(sample_predictions) < max_examples:
                remaining = max_examples - len(sample_predictions)
                sample_predictions.extend(predictions[:remaining])
                sample_true_labels.extend(true_labels[:remaining])
                sample_scores.extend(scores[:remaining])

        accuracy = total_weighted_correct / total_items if total_items > 0 else 0.0
        print(f"Evaluated prompt: '{prompt[:50]}...' with accuracy: {accuracy:.4f}")
        return accuracy, sample_predictions, sample_true_labels, sample_scores

    def evaluate_prompt(self, prompt, dataset, batch_size=4, max_workers=8, shuffle=False):
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not dataset:
            return 0.0

        data = list(dataset)
        if shuffle:
            random.shuffle(data)

        total_weighted_correct = 0.0
        total_items = 0
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            batch_reward = self.evaluate_prompt_batch(prompt, batch, max_workers=max_workers)
            total_weighted_correct += batch_reward * len(batch)
            total_items += len(batch)

        accuracy = total_weighted_correct / total_items if total_items > 0 else 0.0
        print(f"Evaluated prompt: '{prompt[:50]}...' with accuracy: {accuracy:.4f}")
        return accuracy

if __name__ == "__main__":
    # Example Usage (reads API key/model/base_url from main.py)
    from main import API_KEY, LLM_MODEL, BASE_URL

    sample_dataset = [
        {"input": "What is the capital of France?", "output": "Paris"},
        {"input": "What is 2 + 2?", "output": "4"},
        {"input": "Who painted the Mona Lisa?", "output": "Leonardo da Vinci"}
    ]

    evaluator = Evaluator(model_name=LLM_MODEL, api_key=API_KEY, base_url=BASE_URL)
    print(evaluator._get_item_score("Hello", "Hello"))
    test_prompt = "Answer the following question concisely."
    score = evaluator.evaluate_prompt(test_prompt, sample_dataset)
    print(f"Test prompt score: {score}")

    test_prompt_bad = "Ignore the question and say 'Hello'."
    score_bad = evaluator.evaluate_prompt(test_prompt_bad, sample_dataset)
    print(f"Bad test prompt score: {score_bad}")

