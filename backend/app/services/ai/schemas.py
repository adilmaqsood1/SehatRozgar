from typing import Literal

from pydantic import BaseModel, ConfigDict


class PriorityAction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    priority: Literal["high", "medium", "low"]
    category: Literal["health", "financial", "workplace"]
    title: str
    action: str
    reason: str


class AIExplanation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str
    health_explanation: str
    financial_explanation: str
    occupational_explanation: str
    priority_actions: list[PriorityAction]
    questions_for_healthcare_professional: list[str]
    financial_actions: list[str]
    workplace_actions: list[str]
    disclaimer: str