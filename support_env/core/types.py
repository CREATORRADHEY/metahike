from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class Observation(BaseModel):
    ticket_text: str = Field(description="The text of the customer support ticket.")
    history: list[Dict[str, Any]] = Field(default_factory=list, description="History of previous actions in the current episode.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Any additional context about the ticket or task.")

class Action(BaseModel):
    action_type: str = Field(description="The type of action to take (e.g., 'categorize', 'extract', 'draft_response').")
    value: Dict[str, Any] = Field(default_factory=dict, description="The payload for the action.")

class Reward(BaseModel):
    score: float = Field(description="The reward score between 0.0 and 1.0.")
    reason: str = Field(description="Reason for this score.")
    done: bool = Field(description="Whether the task is complete after this action.")
