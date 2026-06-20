from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Dict, List, Mapping


class Agent(ABC):
    """
    Minimal abstract workflow agent.

    The agent receives a static graph and executes it node by node.
    Each graph node maps to a method on the concrete agent.
    """

    def invoke(self, graph: List[Mapping[str, str]], state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state = state or {}

        for node in graph:
            tool_name = node["tool"]
            tool: Callable[[Dict[str, Any]], Dict[str, Any]] | None = getattr(self, tool_name, None)

            if tool is None or not callable(tool):
                raise AttributeError(f"Graph tool '{tool_name}' is not implemented on {self.__class__.__name__}")

            state["current_node"] = node.get("name", tool_name)
            state = tool(state)

            if state.get("stop"):
                break

        return state
