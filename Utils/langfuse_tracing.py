import os
import time
import json
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)
from Utils.badminton_config import INDEX2EN_SHOT

_global_tracer = None
_tracer_initialized = False


def _get_langfuse_client():
    global _global_tracer, _tracer_initialized
    if _tracer_initialized:
        return _global_tracer

    _tracer_initialized = True
    host = os.environ.get("LANGFUSE_HOST")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not all([host, public_key, secret_key]):
        print("[Langfuse] Not configured (missing LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY). Tracing disabled.")
        _global_tracer = None
        return None

    try:
        from langfuse import Langfuse
        _global_tracer = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
        )
        print(f"[Langfuse] Connected to {host}")
    except Exception as e:
        print(f"[Langfuse] Failed to connect: {e}. Tracing disabled.")
        _global_tracer = None

    return _global_tracer


def simplify_naive_rag_experiences(experiences):
    """NaiveRAG: Show complete state snapshot for each retrieved case.

    Each case is a single historical point: where the player stood, what the
    opponent hit, where the ball landed, and how the player responded.
    """
    simplified = []
    for exp in experiences:
        player_type_val = exp.get("player_type")
        player_type_name = INDEX2EN_SHOT.get(int(player_type_val), "?") if player_type_val is not None else "?"
        ps = exp.get("player_score")
        os_ = exp.get("opponent_score")

        state = {
            "score": f"{int(ps)}-{int(os_)}" if ps is not None and os_ is not None else "?-?",
            "round": int(exp.get("ball_round", 0)),
            "player_at": exp.get("player_location_area", "?"),
            "opponent_at": exp.get("opponent_location_area", "?"),
            "opponent_hit": exp.get("opponent_type_name", "?"),
            "ball_landing_at": exp.get("hit_position_area", "?"),
            "player_responded": player_type_name,
            "player_hit_to": exp.get("landing_area", "?"),
            "player_moved_to": exp.get("movement_area", "?"),
        }
        simplified.append({
            "similarity": round(exp.get("similarity", 0), 4),
            "state": state,
        })
    return simplified


def simplify_timerag_windows(windows):
    """TimeRAG: Show time-window evolution (t-window_size .. t) for each retrieved window.

    Each window captures a sequence of consecutive states leading to an action.
    Padding steps (before the rally started) are marked explicitly.
    """
    simplified = []
    for win in windows:
        window_steps = []
        for i, ws in enumerate(win.get("window_states", [])):
            if ws.get("padding"):
                window_steps.append({"step": i, "padding": True})
                continue
            ps = ws.get("player_score")
            os_ = ws.get("opponent_score")
            window_steps.append({
                "step": i,
                "ball_round": ws.get("ball_round"),
                "hitter": ws.get("player", "?"),
                "turn_role": ws.get("turn_role", "target" if ws.get("is_target_turn") else "opponent"),
                "opponent_hit": ws.get("opponent_type_name", "?"),
                "ball_landing_at": ws.get("hit_position_area", "?"),
                "player_at": ws.get("player_location_area", "?"),
                "player_responded": ws.get("player_type_name", "?"),
                "player_hit_to": ws.get("landing_area", "?"),
                "player_moved_to": ws.get("movement_area", "?"),
                "score": f"{int(ps)}-{int(os_)}" if ps is not None and os_ is not None else "?-?",
            })

        action = win.get("action", {})
        final_action = {
            "shot": action.get("player_type_name", "?"),
            "hit_to": action.get("landing_area", "?"),
            "moved_to": action.get("movement_area", "?"),
        }

        simplified.append({
            "similarity": round(win.get("similarity", 0), 4),
            "distance": round(win.get("distance", 0), 4),
            "window": window_steps,
            "final_action": final_action,
        })
    return simplified


def simplify_twin_segments(segments):
    """Twin (I-MSDP): Show multi-scale trajectory segments with step-by-step chain.

    Each segment is a contiguous suffix of length `scale` steps from a historical
    rally, showing the full decision chain: what the opponent hit, where the ball
    landed, and how the player responded at each step.
    """
    simplified = []
    for seg in segments:
        trajectory = []
        for i, step in enumerate(seg.get("steps", [])):
            ps = step.get("player_score")
            os_ = step.get("opponent_score")
            trajectory.append({
                "step": i + 1,
                "opponent_hit": step.get("opponent_type_name", "?"),
                "ball_landing_at": step.get("hit_position_area", "?"),
                "player_at": step.get("player_location_area", "?"),
                "player_responded": step.get("player_type_name", "?"),
                "player_hit_to": step.get("landing_area", "?"),
                "player_moved_to": step.get("movement_area", "?"),
                "score": f"{int(ps)}-{int(os_)}" if ps is not None and os_ is not None else "?-?",
            })

        simplified.append({
            "scale": seg.get("scale", 0),
            "similarity": round(seg.get("similarity", 0), 4),
            "distance": round(seg.get("total_distance", 0), 4),
            "trajectory": trajectory,
        })
    return simplified


class BadmintonTracer:
    def __init__(self, agent_name, player_name, enable_tracing=True):
        self.agent_name = agent_name
        self.player_name = player_name
        self.enable_tracing = enable_tracing
        self.client = None
        if enable_tracing:
            self.client = _get_langfuse_client()
        self._active_trace = None

    @property
    def enabled(self):
        return self.enable_tracing and self.client is not None

    def start_trace(self, state_dict, info, launch):
        if not self.enabled:
            return None

        try:
            rally = info.get("rally", 0)
            ball_round = info.get("round", [1])[-1] if isinstance(info.get("round"), list) else info.get("round", 1)
            player_score = info.get("score", [0, 0])[0] if info.get("score") else 0
            opponent_score = info.get("score", [0, 0])[1] if info.get("score") else 0

            trace_input = {
                "shot_type": state_dict.get("opponent_type", "?"),
                "player_area": state_dict.get("player_location_area", "?"),
                "opponent_area": state_dict.get("opponent_location_area", "?"),
                "hit_area": state_dict.get("hit_position_area", "?"),
                "launch": launch,
            }

            trace_id = self.client.create_trace_id()
            self._active_trace = self.client.start_observation(
                trace_context={"trace_id": trace_id},
                name=f"{self.agent_name}-predict",
                input=trace_input,
                metadata={
                    "agent": self.agent_name,
                    "user_id": self.player_name,
                    "session_id": f"rally-{rally}",
                    "rally": rally,
                    "round": ball_round,
                    "score": f"{player_score}-{opponent_score}",
                },
            )
            return self._active_trace
        except Exception as e:
            print(f"[Langfuse] start_trace error: {e}")
            return None

    def log_retrieval(self, query, results, result_type, latency_ms=0):
        if not self.enabled or self._active_trace is None:
            return

        try:
            query_simple = {
                "player_area": query.get("player_location_area", "?"),
                "opponent_area": query.get("opponent_location_area", "?"),
                "hit_area": query.get("hit_position_area", "?"),
                "opponent_shot": int(query.get("opponent_type", 0)),
            }

            if result_type == "naive_rag":
                simplified = simplify_naive_rag_experiences(results)
            elif result_type == "timerag":
                simplified = simplify_timerag_windows(results)
            elif result_type == "twin":
                simplified = simplify_twin_segments(results)
            else:
                simplified = results

            span = self._active_trace.start_observation(
                name="retrieval",
                input={"query": query_simple},
                output={"count": len(simplified), "cases": simplified},
                metadata={"latency_ms": latency_ms, "type": result_type},
            )
            span.end()
        except Exception as e:
            print(f"[Langfuse] log_retrieval error: {e}")

    def log_generation(self, system_prompt, user_prompt, response, latency_ms=0, raw_llm_output="", usage=None):
        if not self.enabled or self._active_trace is None:
            return

        try:
            metadata = {"latency_ms": latency_ms}
            if usage:
                metadata["usage"] = usage

            gen = self._active_trace.start_observation(
                name="llm-generation",
                as_type="generation",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                output={
                    "parsed": response,
                    "raw_output": raw_llm_output or "",
                },
                metadata=metadata,
            )
            gen.end()
        except Exception as e:
            print(f"[Langfuse] log_generation error: {e}")

    def end_trace(self, action_dict, reasoning):
        if not self.enabled or self._active_trace is None:
            return

        try:
            self._active_trace.update(
                output={
                    "action": action_dict,
                    "reasoning": reasoning,
                }
            )
            self._active_trace.end()
        except Exception as e:
            print(f"[Langfuse] end_trace error: {e}")
        finally:
            self._active_trace = None
