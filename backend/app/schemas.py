from typing import Literal

from pydantic import BaseModel, Field


class HealthIndicators(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(ge=50, le=260)
    chol: float = Field(ge=80, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=50, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)


class AssessmentRequest(BaseModel):
    worker_id: str | None = None
    name: str = "Demo worker"
    age: int = Field(ge=18, le=100)
    occupation: str
    health: HealthIndicators
    daily_income: float = Field(ge=0)
    work_days_per_month: int = Field(ge=0, le=31)
    monthly_household_expenses: float = Field(ge=0)
    current_savings: float = Field(ge=0)
    health_savings: float = Field(ge=0)
    dependents: int = Field(ge=0, le=30)
    estimated_medical_cost: float = Field(ge=0)
    days_unable_to_work: int = Field(default=14, ge=0, le=365)


class SimulationRequest(BaseModel):
    daily_income: float = Field(ge=0)
    work_days_per_month: int = Field(ge=0, le=31)
    monthly_household_expenses: float = Field(ge=0)
    current_savings: float = Field(ge=0)
    health_savings: float = Field(ge=0)
    dependents: int = Field(ge=0, le=30)
    estimated_medical_cost: float = Field(ge=0)
    days_unable_to_work: int = Field(ge=0, le=365)


class ExplainRequest(BaseModel):
    assessment: dict
    mode: Literal["english", "simple", "roman_urdu"] = "english"