import json
import os
import tempfile
from unittest.mock import patch

from apo_optimizer import APOOptimizer
from evaluator import Evaluator


def _fake_batch_scores(predictions, true_labels):
    scores = []
    for pred, true in zip(predictions, true_labels):
        scores.append(1.0 if pred == true else 0.0)
    return scores


def test_evaluate_prompt_batch_accuracy():
    dataset = [
        {"input": "a", "output": "A"},
        {"input": "b", "output": "B"},
        {"input": "c", "output": "C"},
    ]
    evaluator = Evaluator(model_name="test", api_key="test")

    with patch.object(Evaluator, "_predict_one", side_effect=["A", "B", "C"]), \
         patch.object(Evaluator, "_get_batch_scores", side_effect=_fake_batch_scores):
        score = evaluator.evaluate_prompt("prompt", dataset, batch_size=2, max_workers=4, shuffle=False)

    assert score == 1.0


def test_batch_size_edge_case():
    dataset = [
        {"input": "x", "output": "X"},
    ]
    evaluator = Evaluator(model_name="test", api_key="test")

    with patch.object(Evaluator, "_predict_one", return_value="X"), \
         patch.object(Evaluator, "_get_batch_scores", side_effect=_fake_batch_scores):
        score = evaluator.evaluate_prompt("prompt", dataset, batch_size=4, max_workers=2, shuffle=False)

    assert score == 1.0


def test_optimizer_update_per_batch():
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, "data.json")
        dummy_dataset = [
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2"},
        ]
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dummy_dataset, f)

        optimizer = APOOptimizer(
            dataset_path=dataset_path,
            llm_model_name="test",
            api_key="test",
        )

        def fake_score_with_details(prompt, batch, max_workers=8):
            score = 1.0 if prompt == "better" else 0.0
            return score, ["a1", "a2"], ["a1", "a2"], [score, score]

        optimizer.evaluator.evaluate_prompt_batch_with_details = fake_score_with_details
        optimizer.generator.generate_candidates = lambda prompt, n, feedback=None: ["better"]

        best_prompt, best_score, history = optimizer.optimize(
            initial_prompt="base",
            num_iterations=1,
            num_candidates=1,
            batch_size=1,
            shuffle=False,
            update_per_batch=True,
            max_workers=2,
        )

        assert best_prompt == "better"
        assert best_score == 1.0
        assert any("batch_id" in entry for entry in history)


def test_concurrency_smoke():
    dataset = [
        {"input": "x", "output": "x"},
        {"input": "y", "output": "y"},
    ]
    evaluator = Evaluator(model_name="test", api_key="test")

    with patch.object(Evaluator, "_predict_one", side_effect=["x", "y"]), \
         patch.object(Evaluator, "_get_batch_scores", side_effect=_fake_batch_scores):
        score = evaluator.evaluate_prompt_batch("prompt", dataset, max_workers=16)

    assert score == 1.0


if __name__ == "__main__":
    test_evaluate_prompt_batch_accuracy()
    test_batch_size_edge_case()
    test_optimizer_update_per_batch()
    test_concurrency_smoke()
    print("All tests passed.")
