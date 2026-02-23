"""
Agent protocol for APO optimization.
框架层统一使用 Rollout 可调用 (input_data, system_prompt) -> output。
Agent 可二选一：
  1) 实现 set_system_prompt(prompt) + process(input_data)，由 as_rollout 自动包装，无需感知两参调用；
  2) 或直接实现可调用 (input_data, system_prompt) -> output（如纯函数）。
使用时可先 set_system_prompt(prompt)，再 agent(input_data) 单参当作函数用。
"""
from typing import Any, Callable


# Rollout 类型：每次调用时接收 (input_data, system_prompt)，返回 output
Rollout = Callable[[Any, str], Any]


def _has_process_style(agent) -> bool:
    """是否为实现 set_system_prompt + process 的对象（无须 __call__(input_data, system_prompt)）。"""
    return (
        hasattr(agent, "set_system_prompt") and callable(getattr(agent, "set_system_prompt"))
        and hasattr(agent, "process") and callable(getattr(agent, "process"))
    )


def as_rollout(agent) -> Rollout:
    """
    将 agent 转为 (input_data, system_prompt) -> output。
    - 若 agent 有 set_system_prompt 与 process：包装为「先 set 再 process」，Agent 无须实现两参 __call__；
    - 否则若 agent 可调用：视为已是 (input_data, system_prompt) -> output，直接返回。
    """
    if _has_process_style(agent):
        def rollout(input_data: Any, system_prompt: str) -> Any:
            agent.set_system_prompt(system_prompt)
            return agent.process(input_data)
        return rollout
    if callable(agent):
        return agent
    raise TypeError(
        "Agent must be either: (1) callable (input_data, system_prompt) -> output, "
        "or (2) an object with set_system_prompt(prompt) and process(input_data)"
    )
