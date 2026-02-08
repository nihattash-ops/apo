# APO (Automatic Prompt Optimization) 项目

本项目旨在实现一个基于自动提示词优化（APO）核心原理的框架。通过迭代优化提示词，以提高大型语言模型（LLM）在特定任务上的性能。

## 项目结构

```
apo_project/
├── README.md
├── requirements.txt
├── main.py
├── apo_optimizer.py
├── evaluator.py
├── prompt_generator.py
└── data/
    └── sample_data.json
```

## 核心模块

1.  **`apo_optimizer.py`**: 负责协调整个优化流程，包括初始化、迭代优化循环、调用提示词生成器和评估器。
2.  **`evaluator.py`**: 负责评估给定提示词在特定数据集上的性能。它将使用 LLM 来执行任务并计算性能指标（例如，准确率、F1 分数）。
3.  **`prompt_generator.py`**: 负责根据当前最佳提示词和评估反馈，生成新的候选提示词。这可能涉及文本梯度、变异或启发式方法。
4.  **`main.py`**: 项目的入口点，用于配置优化任务、加载数据并启动优化过程。
5.  **`data/sample_data.json`**: 示例数据集，包含用于评估提示词的输入和期望输出。

## 算法流程概述

1.  **初始化**：选择一个初始提示词（种子提示词）和评估数据集。
2.  **评估**：使用 `evaluator` 模块评估当前提示词在数据集上的性能。
3.  **生成**：使用 `prompt_generator` 模块根据评估结果生成一组新的候选提示词。
4.  **筛选**：从候选提示词中选择表现最佳的提示词。
5.  **迭代**：重复步骤 2-4，直到达到预设的优化次数或性能指标。
6.  **输出**：返回最佳提示词及其性能。

## 安装与运行

1.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```
2.  **配置 API 密钥**：
    在 `main.py` 或环境变量中配置您的 LLM API 密钥。
3.  **运行优化**：
    ```bash
    python main.py
    ```

## 注意事项

-   本项目需要访问一个 LLM API（例如 OpenAI GPT 系列、Gemini 等）。
-   为了简化示例，本项目可能不会实现所有复杂的 APO 变体，但会涵盖核心思想。
-   实际应用中，`evaluator` 和 `prompt_generator` 的实现会更复杂，可能涉及更精细的评估指标和更智能的提示词生成策略。

## Batch Training Mode (Mini-Batch)

This project supports mini-batch optimization with concurrent evaluation. Key options:
- batch_size: number of samples per batch (default 4)
- shuffle: shuffle dataset each iteration (default True)
- update_per_batch: update prompt after each batch (default True)
- max_workers: thread pool size for concurrent LLM calls (default 8)

Example (main.py):
- batch_size = 4
- shuffle = True
- update_per_batch = True
- max_workers = 8
