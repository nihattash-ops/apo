"""
APO 协议与类型定义：使算法能适配各种形式的 Agent 或函数。
"""
from typing import Any, Callable, Protocol, Optional

# 标准 Rollout：算法只依赖 (input_data, system_prompt) -> output
Rollout = Callable[[Any, str], Any]


class RolloutWithKwargs(Protocol):
    """支持额外参数的 Rollout：部分 Agent 需要 context/kwargs。"""

    def __call__(self, input_data: Any, system_prompt: str, **kwargs: Any) -> Any:
        ...


def get_input_from_item(item: Any, input_key: str = "input") -> Any:
    """从数据条中取出 input，兼容 dict 或对象。"""
    if isinstance(item, dict):
        return item.get(input_key)
    return getattr(item, input_key, None)


def get_output_from_item(item: Any, output_key: str = "output") -> Any:
    """从数据条中取出 output（期望输出），兼容 dict 或对象。"""
    if isinstance(item, dict):
        return item.get(output_key)
    return getattr(item, output_key, None)


# 自定义 reward：(output, expected_output, item) -> score in [0, 1]
RewardFn = Callable[[Any, Any, Any], float]


def make_rollout(
    agent: Any,
    system_prompt_attr: str = "system_prompt",
    run_method: str = "_run",
    pass_extra_kwargs: bool = False,
) -> Callable[..., Any]:
    """
    将任意具有「可设置 system_prompt + 单输入 run 方法」的 Agent 包成 Rollout。
    适用于大多数单一 system_prompt + 一次调用的 Agent。

    Args:
        agent: 任意对象，需有 run_method 对应的方法。
        system_prompt_attr: 用于设置 system prompt 的属性名。
        run_method: 执行单次推理的方法名，签名为 (input_data) 或 (input_data, **kwargs)。
        pass_extra_kwargs: 若为 True，rollout 被调用时的 **kwargs 会传给 run 方法。

    Example:
        agent = CargoAgent(...)
        rollout = make_rollout(agent)
        optimizer = APOOptimizerAgent(rollout=rollout, dataset=dataset, ...)
    """
    run = getattr(agent, run_method, None)
    if run is None or not callable(run):
        raise ValueError(f"Agent must have a callable '{run_method}' method")

    def _rollout(input_data: Any, system_prompt: str, **kwargs: Any) -> Any:
        setattr(agent, system_prompt_attr, system_prompt)
        if pass_extra_kwargs and kwargs:
            return run(input_data, **kwargs)
        return run(input_data)

    return _rollout
