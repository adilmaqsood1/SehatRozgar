import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.ml.predictor import HealthPredictor
from app.ml.train import METADATA_PATH
from app.schemas import AssessmentRequest, ExplainRequest, SimulationRequest
from app.services.ai.resilience_agent import generate_explanation
from app.services.fallback import explain_assessment
from app.services.financial import FinancialProfile, analyze_finances, simulate_scenarios
from app.services.occupational import assess_occupation


ROOT = Path(__file__).resolve().parent
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(PROJECT_DIR / ".env", override=False)
load_dotenv(BACKEND_DIR / ".env", override=False)
WORKERS_PATH = ROOT / "datasets" / "demo_workers.json"
app = FastAPI(title="SehatRozgar API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
predictor = HealthPredictor()


def load_workers() -> list[dict]:
    return json.loads(WORKERS_PATH.read_text(encoding="utf-8"))


def financial_profile(data) -> FinancialProfile:
    return FinancialProfile(
        daily_income=data.daily_income,
        work_days_per_month=data.work_days_per_month,
        monthly_household_expenses=data.monthly_household_expenses,
        current_savings=data.current_savings,
        health_savings=data.health_savings,
        dependents=data.dependents,
        estimated_medical_cost=data.estimated_medical_cost,
    )


@app.get("/api/health")
def health_check():
    return {"status": "ok", "model_ready": predictor.available}


@app.get("/api/demo/workers")
def demo_workers():
    return [{key: value for key, value in worker.items() if key != "health"} for worker in load_workers()]


@app.get("/api/demo/workers/{worker_id}")
def get_demo_worker(worker_id: str):
    worker = next((item for item in load_workers() if item["id"] == worker_id), None)
    if worker is None:
        raise HTTPException(status_code=404, detail="Demo worker not found")
    return {**worker, "profile_type": "Synthetic Demo Profile"}


@app.post("/api/assessment")
def assess(request: AssessmentRequest):
    if not predictor.available:
        raise HTTPException(status_code=503, detail="Empirical model is not trained. Run the documented dataset download and training commands.")
    finance = analyze_finances(financial_profile(request), request.days_unable_to_work)
    financial_result = {
        **finance,
        "daily_income": request.daily_income,
        "days_unable_to_work": request.days_unable_to_work,
        "estimated_medical_cost": request.estimated_medical_cost,
    }
    result = {
        "worker": {"id": request.worker_id, "name": request.name, "age": request.age, "occupation": request.occupation},
        "health": predictor.predict(request.health.model_dump()),
        "financial": financial_result,
        "occupational": assess_occupation(request.occupation),
        "scenarios": simulate_scenarios(financial_profile(request), request.estimated_medical_cost),
        "days_unable_to_work": request.days_unable_to_work,
    }
    result["ai_explanation"] = explain_assessment(result)
    return result


@app.post("/api/financial/simulate")
def simulate(request: SimulationRequest):
    profile = financial_profile(request)
    return {
        "selected_scenario": analyze_finances(profile, request.days_unable_to_work),
        "scenarios": simulate_scenarios(profile, request.estimated_medical_cost),
        "selected_days": request.days_unable_to_work,
    }


@app.post("/api/ai/explain")
async def explain(request: ExplainRequest):
    return await generate_explanation(request.assessment, request.mode)


@app.get("/api/model")
@app.get("/api/model/metrics")
def model_metadata():
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Model metrics are not available until the empirical model is trained.")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@app.get("/api/methodology")
def methodology():
    return {
        "dataset": "UCI Heart Disease, processed Cleveland subset",
        "source": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "observations": 303,
        "target": "num: 0 vs 1-4 binarized to absence/presence in the source cohort",
        "split": "Stratified 75/25 holdout; random_state=42",
        "explanation": "Feature contributions use a one-feature-at-a-time baseline perturbation and describe model behavior, not causality.",
        "financial_index": "Weighted deterministic index: runway 25%, expense-to-income 20%, emergency gap 20%, dependents 10%, event coverage 15%, interruption exposure 10%. Health probability is excluded.",
        "safety": "Research prototype only; not medical advice, diagnosis, lending, employment, insurance pricing, or eligibility guidance."
    }


@app.get("/api/workforce/analytics")
def workforce_analytics():
    profiles = load_workers()
    levels = {"resilient": 0, "moderate": 0, "high": 0, "critical": 0}
    occupations = {}
    hazards = {}
    under_14_day_runway = 0
    preparedness = []
    for worker in profiles:
        profile = FinancialProfile(
            daily_income=worker["daily_income"],
            work_days_per_month=worker["work_days_per_month"],
            monthly_household_expenses=worker["monthly_household_expenses"],
            current_savings=worker["current_savings"],
            health_savings=worker["health_savings"],
            dependents=worker["dependents"],
            estimated_medical_cost=worker["estimated_medical_cost"],
        )
        financial = analyze_finances(profile, 14)
        levels[financial["vulnerability_level"]] += 1
        if financial["savings_runway_days"] is not None and financial["savings_runway_days"] < 14:
            under_14_day_runway += 1
        preparedness.append(100 - financial["vulnerability_score"])
        occupation = worker["occupation"]
        occupations[occupation] = occupations.get(occupation, 0) + 1
        for hazard in assess_occupation(occupation)["hazards"]:
            hazards[hazard] = hazards.get(hazard, 0) + 1
    return {
        "workers": len(profiles),
        "under_14_day_runway": under_14_day_runway,
        "average_emergency_preparedness": round(sum(preparedness) / len(preparedness)) if preparedness else 0,
        "vulnerability_distribution": [{"level": key, "count": value} for key, value in levels.items()],
        "occupation_distribution": [{"occupation": key, "count": value} for key, value in occupations.items()],
        "occupational_hazard_distribution": [{"hazard": key, "count": value} for key, value in sorted(hazards.items(), key=lambda item: item[1], reverse=True)],
        "privacy": "Aggregated synthetic profiles only; individual health inputs are excluded.",
    }