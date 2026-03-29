from typing import Dict, Any, Tuple
from core.types import Observation, Action, Reward
from core.tasks import TASKS, TaskInstance

class SupportEnv:
    def __init__(self):
        self.current_task_id = None
        self.task = None
        self.instance: TaskInstance = None
        self.history = []
        self.done = True
        self.score = 0.0

    def reset(self, task_id: str) -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
            
        self.current_task_id = task_id
        self.task = TASKS[task_id]
        self.instance = self.task.sample_instance()
        self.history = []
        self.done = False
        self.score = 0.0
        
        return self._get_observation()

    def _get_observation(self) -> Observation:
        return Observation(
            ticket_text=self.instance.text,
            history=self.history.copy(),
            metadata={"ticket_id": self.instance.ticket_id}
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is already done. Please reset.")
            
        action_dict = {"action_type": action.action_type, "value": action.value}
        
        # Specific handling for intermediate tool actions (like lookup_policy)
        if action.action_type == "lookup_policy":
            topic = action.value.get("topic", "")
            # Inject policy lookup into action history so agent sees the rule
            if topic in getattr(self.task, "policies", {}):
                action_dict["result"] = self.task.policies[topic]
            else:
                action_dict["result"] = f"No policy found for '{topic}'"
                
        self.history.append(action_dict)
        
        score_update, reason, done = self.task.evaluate(self.instance, self.history)
        
        self.score += score_update
        # Bound score logically between 0.0 and 1.0 overall if we want
        current_score = max(0.0, min(1.0, self.score))
        
        reward = Reward(
            score=current_score,
            reason=reason,
            done=done
        )
        self.done = done
        
        return self._get_observation(), reward, self.done, {"raw_score": self.score}

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.current_task_id,
            "ticket_id": self.instance.ticket_id if self.instance else None,
            "done": self.done,
            "score": self.score,
            "history_length": len(self.history)
        }
