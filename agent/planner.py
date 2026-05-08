"""
agent/planner.py — LangGraph decision graph (Cluster 01: Sentient Edge Node)

Stateful LangGraph agent that takes a perception context dict and
decides what action the node should take next.

SDKs: LangGraph, Ollama (via langchain-community)
"""
import json
import logging
import time
from typing import TypedDict, Annotated, Optional
import operator

from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an edge AI node — a local perception-decision-action agent.
You receive a structured JSON snapshot of what the node currently perceives.
Your job is to decide the single best action to take right now.

Available actions:
- alert(message, severity): Publish an alert over MQTT. severity: low|medium|high|critical
- track(track_id, label): Begin focused tracking of a specific object
- ignore(): No action needed this cycle
- log(note): Write an observation to memory without triggering actuators
- speak(text): Send a TTS message over the speaker actuator

Rules:
- Respond ONLY with valid JSON: {"action": "<name>", "payload": {...}, "reasoning": "<one sentence>"}
- Never hallucinate sensor data. Only reason about what is in the context.
- Be decisive. One action per cycle.
"""


class AgentState(TypedDict):
    context: dict
    messages: Annotated[list, operator.add]
    action: Optional[str]
    payload: Optional[dict]
    reasoning: Optional[str]
    error: Optional[str]


class EdgePlanner:
    """
    LangGraph-based decision agent for the Sentient Edge Node.
    Single-turn: context in → action out. Stateless between calls.
    """

    def __init__(self, config: dict):
        self._cfg = config.get("ollama", {})
        self._model = None
        self._graph = None

    def load(self) -> None:
        base_url = self._cfg.get("base_url", "http://localhost:11434")
        model = self._cfg.get("model", "llama3")

        self._model = ChatOllama(
            base_url=base_url,
            model=model,
            temperature=0.1,
            format="json",
        )
        self._graph = self._build_graph()
        logger.info("EdgePlanner loaded: %s @ %s", model, base_url)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("perceive", self._perceive_node)
        graph.add_node("decide", self._decide_node)
        graph.add_node("validate", self._validate_node)
        graph.set_entry_point("perceive")
        graph.add_edge("perceive", "decide")
        graph.add_edge("decide", "validate")
        graph.add_edge("validate", END)
        return graph.compile()

    def _perceive_node(self, state: AgentState) -> AgentState:
        """Format context into a human message for the LLM."""
        context_str = json.dumps(state["context"], indent=2)
        msg = HumanMessage(content=f"Current perception context:
{context_str}

What action should I take?")
        return {**state, "messages": [SystemMessage(content=SYSTEM_PROMPT), msg]}

    def _decide_node(self, state: AgentState) -> AgentState:
        """Call Ollama and get a decision."""
        try:
            response = self._model.invoke(state["messages"])
            return {**state, "messages": [AIMessage(content=response.content)], "error": None}
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return {**state, "error": str(e)}

    def _validate_node(self, state: AgentState) -> AgentState:
        """Parse and validate the LLM JSON output."""
        if state.get("error"):
            return {**state, "action": "ignore", "payload": {}, "reasoning": f"LLM error: {state['error']}"}

        last_msg = state["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        try:
            parsed = json.loads(content)
            action = parsed.get("action", "ignore")
            payload = parsed.get("payload", {})
            reasoning = parsed.get("reasoning", "")
            # validate action name
            valid_actions = {"alert", "track", "ignore", "log", "speak"}
            if action not in valid_actions:
                logger.warning("Unknown action '%s', defaulting to ignore", action)
                action = "ignore"
                payload = {}
            return {**state, "action": action, "payload": payload, "reasoning": reasoning}
        except json.JSONDecodeError as e:
            logger.warning("Could not parse LLM response as JSON: %s | raw: %s", e, content[:200])
            return {**state, "action": "ignore", "payload": {}, "reasoning": "parse error"}

    def decide(self, context: dict) -> dict:
        """
        Main entry point. Takes a perception context dict,
        returns {"action": str, "payload": dict, "reasoning": str}.
        """
        if self._graph is None:
            raise RuntimeError("EdgePlanner not loaded. Call load() first.")

        initial_state: AgentState = {
            "context": context,
            "messages": [],
            "action": None,
            "payload": None,
            "reasoning": None,
            "error": None,
        }

        t0 = time.perf_counter()
        result = self._graph.invoke(initial_state)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "action": result.get("action", "ignore"),
            "payload": result.get("payload", {}),
            "reasoning": result.get("reasoning", ""),
            "decision_ms": round(elapsed_ms, 1),
        }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    planner = EdgePlanner(cfg)
    planner.load()

    test_context = {
        "timestamp": time.time(),
        "active_track_ids": [1, 2],
        "classes_seen": {"person": 2},
        "recent_detections": [
            {"track_id": 1, "class_name": "person", "confidence": 0.92, "depth_m": 1.5},
            {"track_id": 2, "class_name": "person", "confidence": 0.88, "depth_m": 3.2},
        ],
        "recent_speech": [],
        "recent_decisions": [],
        "long_term": {"zone": "restricted", "alert_cooldown_s": 10},
    }

    result = planner.decide(test_context)
    print(json.dumps(result, indent=2))
