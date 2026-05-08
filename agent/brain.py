"""
agent/brain.py
Sentient Edge Node — Cluster 01
LangGraph-based decision agent. Takes FrameResult detections and
decides what actions to dispatch via MQTT/ZeroMQ actuators.
"""

import json
import logging
from typing import TypedDict, Annotated
from dataclasses import asdict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import Ollama

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    detections: list[dict]
    actions: list[dict]
    cycle: int


def build_detection_prompt(detections: list[dict]) -> str:
    if not detections:
        return "No objects detected in the current frame. Scene is clear."
    lines = [f"- {d['label']} (confidence: {d['confidence']:.0%}, track_id: {d.get('track_id', 'N/A')})"
             for d in detections]
    return "Current frame detections:\n" + "\n".join(lines)


def perception_node(state: AgentState) -> AgentState:
    """Format detections into a message for the reasoning node."""
    prompt = build_detection_prompt(state["detections"])
    return {
        **state,
        "messages": state["messages"] + [HumanMessage(content=prompt)],
    }


def reasoning_node(state: AgentState, llm: Ollama) -> AgentState:
    """Ask the local LLM to reason about what actions to take."""
    system = (
        "You are an edge AI agent controlling a physical system. "
        "Given a list of detected objects, decide what actions to dispatch. "
        "Respond with a JSON array of actions. Each action has: "
        '{"type": "alert|move|log|ignore", "target": "...", "reason": "..."}. '
        "Be concise. Only respond with the JSON array, no explanation."
    )
    messages = [{"role": "system", "content": system}]
    for m in state["messages"]:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        messages.append({"role": role, "content": m.content})

    try:
        response = llm.invoke(messages[-1]["content"])
        # Parse JSON from LLM response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            actions = json.loads(response[start:end])
        else:
            actions = [{"type": "log", "target": "scene", "reason": response}]
    except Exception as e:
        log.warning(f"LLM parse error: {e}")
        actions = [{"type": "log", "target": "error", "reason": str(e)}]

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=json.dumps(actions))],
        "actions": actions,
        "cycle": state.get("cycle", 0) + 1,
    }


def should_act(state: AgentState) -> str:
    """Route to actuator if there are non-ignore actions."""
    actionable = [a for a in state.get("actions", []) if a.get("type") != "ignore"]
    return "dispatch" if actionable else END


def dispatch_node(state: AgentState) -> AgentState:
    """Log actions — replace with MQTT/ZMQ publish in production."""
    for action in state["actions"]:
        if action.get("type") != "ignore":
            log.info(f"ACTION: {json.dumps(action)}")
    return state


def build_agent(model: str = "llama3") -> StateGraph:
    llm = Ollama(model=model)

    graph = StateGraph(AgentState)
    graph.add_node("perception", perception_node)
    graph.add_node("reasoning", lambda s: reasoning_node(s, llm))
    graph.add_node("dispatch", dispatch_node)

    graph.set_entry_point("perception")
    graph.add_edge("perception", "reasoning")
    graph.add_conditional_edges("reasoning", should_act, {"dispatch": "dispatch", END: END})
    graph.add_edge("dispatch", END)

    return graph.compile()


if __name__ == "__main__":
    import sys
    from dataclasses import dataclass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [brain] %(message)s")

    # Test with fake detections
    test_detections = [
        {"label": "person", "confidence": 0.91, "bbox": [0.1, 0.1, 0.5, 0.9], "track_id": 1},
        {"label": "backpack", "confidence": 0.72, "bbox": [0.15, 0.5, 0.4, 0.85], "track_id": None},
    ]

    agent = build_agent(model="llama3")
    result = agent.invoke({
        "messages": [],
        "detections": test_detections,
        "actions": [],
        "cycle": 0,
    })
    print("Actions:", json.dumps(result["actions"], indent=2))
