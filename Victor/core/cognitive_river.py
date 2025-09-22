import math
import time
import threading
from collections import deque
from typing import Any, Dict, Optional, Callable

# === COGNITIVE RIVER CORE ===

def _softmax(xs):
    m = max(xs) if xs else 0.0
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

def _ema(prev, new, alpha=0.2):
    if prev is None: return new
    if isinstance(new, (int, float)) and isinstance(prev, (int, float)):
        return (1-alpha)*prev + alpha*new
    return new

class RingBuffer:
    def __init__(self, n=256):
        self.n = n
        self.q = deque(maxlen=n)
    def add(self, x): self.q.append(x)
    def to_list(self): return list(self.q)

def _emo_boost(d):
    if not d: return 0.0
    a = float(d.get("arousal", 0.0))
    return min(0.4, 0.1 + 0.5*a)

def _mem_boost(d):
    if not d: return 0.0
    sal = float(d.get("salience", 0.0))
    return min(0.35, 0.05 + 0.6*sal)

def _sys_boost(d):
    if not d: return 0.0
    active = int(d.get("active_tasks", 0))
    return min(0.3, 0.05 + 0.02*active)

def _sens_boost(d):
    if not d: return 0.0
    novelty = float(d.get("novelty", 0.0))
    return min(0.35, 0.05 + 0.5*novelty)

def _rw_boost(d):
    if not d: return 0.0
    urgency = float(d.get("urgency", 0.0))
    return min(0.35, 0.05 + 0.5*urgency)

class CognitiveRiver8:
    STREAMS = ["status","emotion","memory","awareness","systems","user","sensory","realworld"]

    def __init__(self, loop=True, step_hz=5):
        self.loop = loop
        self.dt = 1.0/float(step_hz)

        self.state: Dict[str, Any] = {k: None for k in self.STREAMS}
        self.priority_logits: Dict[str, float] = {k: 0.0 for k in self.STREAMS}
        self.energy = 0.5
        self.stability = 0.8
        self.energy_baseline = 0.5
        self.stability_baseline = 0.8
        self.last_merge: Optional[Dict[str, Any]] = None
        self.event_log = RingBuffer(n=1024)
        self.merge_log = RingBuffer(n=512)
        self.on_merge: Optional[Callable[[Dict[str,Any]], None]] = None

    def set_status(self, d: Dict[str, Any]):    self._set("status", d, boost=0.1)
    def set_emotion(self, d: Dict[str, Any]):   self._set("emotion", d, boost=_emo_boost(d))
    def set_memory(self, d: Dict[str, Any]):    self._set("memory", d, boost=_mem_boost(d))
    def set_awareness(self, d: Dict[str, Any]): self._set("awareness", d, boost=0.15)
    def set_systems(self, d: Dict[str, Any]):   self._set("systems", d, boost=_sys_boost(d))
    def set_user(self, d: Dict[str, Any]):      self._set("user", d, boost=0.25)
    def set_sensory(self, d: Dict[str, Any]):   self._set("sensory", d, boost=_sens_boost(d))
    def set_realworld(self, d: Dict[str, Any]): self._set("realworld", d, boost=_rw_boost(d))

    def _set(self, key, payload, boost=0.0):
        self.state[key] = payload
        self.priority_logits[key] = _ema(self.priority_logits[key], self.priority_logits[key] + boost, 0.5)
        self.event_log.add({"t": time.time(), "event": "update", "key": key, "data": payload})

    def _auto_priorities(self) -> Dict[str, float]:
        logits = dict(self.priority_logits)
        aw = self._scalar_get(self.state["awareness"], "clarity", default=0.6)
        arousal = self._scalar_get(self.state["emotion"], "arousal", default=0.4)

        logits["awareness"] += 0.5 * aw
        logits["status"]    += 0.3 * aw
        logits["user"]      += 0.3 * aw
        logits["emotion"]   += 0.4 * (self.energy + arousal)
        logits["sensory"]   += 0.3 * (self.energy + max(0.0, arousal))
        logits["memory"]    += 0.4 * (1.0 - self.stability)
        logits["systems"]   += 0.3 * (1.0 - self.stability)
        logits["realworld"] += 0.25 * (aw + self.energy)

        for k in self.STREAMS:
            if self.state[k] is None:
                logits[k] -= 0.5

        ordered = [logits[k] for k in self.STREAMS]
        w = _softmax(ordered)
        return {k:w[i] for i,k in enumerate(self.STREAMS)}

    def _scalar_get(self, obj, key, default=0.0):
        try:
            if obj is None: return default
            v = obj.get(key, default)
            return float(v) if v is not None else default
        except Exception:
            return default

    def step_merge(self) -> Dict[str, Any]:
        weights = self._auto_priorities()
        signal = {k: self.state[k] for k in self.STREAMS}
        top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]

        merged = {
            "t": time.time(),
            "weights": weights,
            "signal": signal,
            "summary": {
                "energy": self.energy,
                "stability": self.stability,
                "top_streams": [item[0] for item in top3]
            },
            "intent": self._draft_intent(weights, signal)
        }

        self.last_merge = merged
        self.merge_log.add(merged)

        if self.on_merge:
            try: self.on_merge(merged)
            except Exception as e:
                self.event_log.add({"t": time.time(), "event":"on_merge_error", "err": str(e)})

        return merged

    def _draft_intent(self, w: Dict[str,float], s: Dict[str,Any]) -> Dict[str, Any]:
        sorted_w = sorted(w.items(), key=lambda x: x[1], reverse=True)
        leader = sorted_w[0][0] if sorted_w else "awareness"

        if leader in ("user","emotion"):
            goal = "respond"
        elif leader in ("systems","memory"):
            goal = "plan"
        elif leader in ("realworld","sensory"):
            goal = "observe"
        else:
            goal = "reflect"

        return {"mode": goal, "leader": leader}

    def auto_homeostasis(self):
        # Gently regress energy and stability to their baselines
        self.energy = _ema(self.energy, self.energy_baseline, alpha=0.001)
        self.stability = _ema(self.stability, self.stability_baseline, alpha=0.001)

    def run_forever(self):
        while self.loop:
            self.step_merge()
            self.auto_homeostasis()
            time.sleep(self.dt)

    def start_thread(self):
        self.loop = True
        t = threading.Thread(target=self.run_forever, daemon=True)
        t.start()
        return t

    def set_energy(self, x: float):
        self.energy = float(min(max(x,0.0),1.0))

    def set_stability(self, x: float):
        self.stability = float(min(max(x,0.0),1.0))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "t": time.time(),
            "last_merge": self.last_merge,
            "energy": self.energy,
            "stability": self.stability,
            "priority_logits": dict(self.priority_logits),
            "merge_log_tail": self.merge_log.to_list()[-5:],
        }

    def feedback_adjustment(self, key: str, success: bool):
        delta = 0.05 if success else -0.05
        self.priority_logits[key] = _ema(self.priority_logits.get(key, 0.0), self.priority_logits.get(key, 0.0) + delta, 0.3)
