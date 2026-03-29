from typing import List, Dict, Any, Tuple
import random
from pydantic import BaseModel
from abc import ABC, abstractmethod

class TaskInstance(BaseModel):
    ticket_id: str
    text: str
    metadata: Dict[str, Any]

class SupportTask(ABC):
    task_id: str
    description: str

    @abstractmethod
    def sample_instance(self) -> TaskInstance:
        pass

    @abstractmethod
    def evaluate(self, instance: TaskInstance, history: List[Dict[str, Any]]) -> Tuple[float, str, bool]:
        """Returns (score, reason, done)"""
        pass

# Easy Task: Categorize ticket into "Billing", "Tech Support", or "Refund"
class TaskCategorize(SupportTask):
    task_id = "task_categorize_easy"
    description = "Categorize the ticket into one of: 'Billing', 'Tech Support', 'Refund'."
    instances = [
        TaskInstance(ticket_id="T001", text="I was charged twice for my subscription this month, please fix.", metadata={"topic": "Billing"}),
        TaskInstance(ticket_id="T002", text="The app keeps crashing when I open the dashboard.", metadata={"topic": "Tech Support"}),
        TaskInstance(ticket_id="T003", text="I don't like this product. Give me my money back.", metadata={"topic": "Refund"})
    ]

    def sample_instance(self) -> TaskInstance:
        return random.choice(self.instances)

    def evaluate(self, instance: TaskInstance, history: List[Dict[str, Any]]) -> Tuple[float, str, bool]:
        if not history:
            return 0.0, "No actions taken", False
        
        last_action = history[-1]
        
        # We only evaluate 'submit' type actions for final scoring
        if last_action.get("action_type") == "submit":
            prediction = last_action.get("value", {}).get("category", "")
            if prediction.lower() == instance.metadata["topic"].lower():
                return 1.0, "Correctly categorized", True
            else:
                return 0.0, f"Incorrect category. Expected {instance.metadata['topic']}", True
                
        # Partial rewards during exploration
        if last_action.get("action_type") == "think":
            return 0.1, "Thinking is good", False
            
        return 0.0, "Action not recognized", False


# Medium Task: Categorize AND Extract Order ID
class TaskExtract(SupportTask):
    task_id = "task_extract_medium"
    description = "Categorize the ticket (Billing/Tech Support/Refund) and extract the Order ID."
    instances = [
        TaskInstance(ticket_id="T101", text="Order #12345 hasn't arrived. I want a refund.", metadata={"topic": "Refund", "order_id": "12345"}),
        TaskInstance(ticket_id="T102", text="Can I change payment method? It's for order 99ABC.", metadata={"topic": "Billing", "order_id": "99ABC"}),
        TaskInstance(ticket_id="T103", text="Order O772. The screen is cracked.", metadata={"topic": "Tech Support", "order_id": "O772"})
    ]

    def sample_instance(self) -> TaskInstance:
        return random.choice(self.instances)

    def evaluate(self, instance: TaskInstance, history: List[Dict[str, Any]]) -> Tuple[float, str, bool]:
        if not history:
            return 0.0, "No actions", False
            
        last_action = history[-1]
        if last_action.get("action_type") == "submit":
            pred_cat = last_action.get("value", {}).get("category", "")
            pred_id = last_action.get("value", {}).get("order_id", "")
            
            score = 0.0
            reasons = []
            if pred_cat.lower() == instance.metadata["topic"].lower():
                score += 0.5
                reasons.append("Correct category")
            else:
                reasons.append("Incorrect category")
                
            if pred_id.lower() == instance.metadata["order_id"].lower():
                score += 0.5
                reasons.append("Correct Order ID")
            else:
                reasons.append("Incorrect Order ID")
                
            return score, " | ".join(reasons), True
            
        return 0.0, "Continue", False


# Hard Task: Multi-step drafting
class TaskDraft(SupportTask):
    task_id = "task_draft_hard"
    description = "Read ticket, use action_type='lookup_policy' to get rule on the topic, then action_type='submit' with a response draft."
    instances = [
        TaskInstance(ticket_id="T201", text="I am returning order 55X because it broke after 40 days.", metadata={"topic": "Refund", "order_id": "55X"}),
        TaskInstance(ticket_id="T202", text="Why did you charge me $10 extra on order 12Y?", metadata={"topic": "Billing", "order_id": "12Y"})
    ]
    policies = {
        "Refund": "No refunds after 30 days.",
        "Billing": "An extra $10 applies for expedited shipping."
    }

    def sample_instance(self) -> TaskInstance:
        return random.choice(self.instances)

    def evaluate(self, instance: TaskInstance, history: List[Dict[str, Any]]) -> Tuple[float, str, bool]:
        if not history:
            return 0.0, "No actions", False
            
        last_action = history[-1]
        
        # The agent can take 'lookup_policy'. We give a tiny partial reward for doing so correctly.
        # But wait, step() needs to return observation if it's lookup_policy. The grader just scores.
        # So evaluate just gives the reward. 
        if last_action.get("action_type") == "lookup_policy":
            topic = last_action.get("value", {}).get("topic", "")
            if topic in self.policies:
                # Good action, but not done
                return 0.1, "Valid policy search", False
            else:
                return -0.1, "Invalid policy topic", False
                
        if last_action.get("action_type") == "submit":
            draft = last_action.get("value", {}).get("draft", "")
            topic = instance.metadata["topic"]
            
            # Check if they looked up policy beforehand
            looked_up = any(a.get("action_type") == "lookup_policy" for a in history)
            
            score = 0.0
            reasons = []
            
            if looked_up:
                score += 0.2
                reasons.append("Looked up policy")
            else:
                reasons.append("Did not look up policy")
                
            # Naive text check for policy compliance
            if topic == "Refund" and "30 days" in draft.lower():
                score += 0.8
                reasons.append("Policy rule included")
            elif topic == "Billing" and "$10" in draft and "shipping" in draft.lower():
                score += 0.8
                reasons.append("Policy rule included")
            else:
                reasons.append("Policy rule missing in draft")
                
            return score, " | ".join(reasons), True
            
        return 0.0, "Continue", False

TASKS = {
    TaskCategorize.task_id: TaskCategorize(),
    TaskExtract.task_id: TaskExtract(),
    TaskDraft.task_id: TaskDraft()
}
