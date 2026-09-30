import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ai import resilience_agent
from app.services.ai.resilience_agent import generate_explanation
from app.services.ai.featherless_client import sanitize_for_llm
from app.services.financial import FinancialProfile, analyze_finances, simulate_scenarios
from app.services.occupational import assess_occupation

client = TestClient(app)


@pytest.fixture
def profile():
    return FinancialProfile(1800, 24, 32000, 18000, 3000, 4, 8000)


def test_financial_formulas(profile):
    result = analyze_finances(profile, 14)
    assert result["monthly_income"] == 43200
    assert result["income_loss"] == 25200
    assert result["gross_event_exposure"] == 33200
    assert result["health_savings_used"] == 3000
    assert result["net_event_exposure"] == 30200
    assert result["savings_runway_days"] == 16.9
    assert result["remaining_savings"] == 0


def test_scenarios_have_required_days(profile):
    assert [item["days_off_work"] for item in simulate_scenarios(profile, 8000)] == [3, 7, 14, 30]


def test_zero_expenses_has_unbounded_runway():
    empty_costs = FinancialProfile(0, 0, 0, 0, 0, 0, 0)
    result = analyze_finances(empty_costs, 0)
    assert result["savings_runway_days"] is None
    assert result["vulnerability_score"] == 0


def test_zero_savings_and_zero_income_are_finite():
    no_income = FinancialProfile(0, 20, 12000, 0, 0, 2, 5000)
    result = analyze_finances(no_income, 7)
    assert result["event_coverage_ratio"] == 0
    assert 0 <= result["vulnerability_score"] <= 100


def test_negative_financial_values_are_rejected(profile):
    invalid = FinancialProfile(-1, profile.work_days_per_month, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        analyze_finances(invalid, 1)


def test_occupation_mapping():
    result = assess_occupation("Construction Worker")
    assert result["risk_level"] == "high"
    assert "Falls from height" in result["hazards"]
    assert assess_occupation("Unknown role")["risk_level"] == "not_mapped"


def test_demo_and_assessment_api_are_connected():
    workers = client.get("/api/demo/workers").json()
    assert len(workers) == 6
    assert "health" not in workers[0]
    profile = client.get("/api/demo/workers/ahmed").json()
    response = client.post("/api/assessment", json={
        "worker_id": profile["id"], "name": profile["name"], "age": profile["age"],
        "occupation": profile["occupation"], "health": profile["health"],
        "daily_income": profile["daily_income"], "work_days_per_month": profile["work_days_per_month"],
        "monthly_household_expenses": profile["monthly_household_expenses"], "current_savings": profile["current_savings"],
        "health_savings": profile["health_savings"], "dependents": profile["dependents"],
        "estimated_medical_cost": profile["estimated_medical_cost"], "days_unable_to_work": 14,
    })
    assert response.status_code == 200
    result = response.json()
    assert result["health"]["risk_probability"] > 0
    assert [item["days_off_work"] for item in result["scenarios"]] == [3, 7, 14, 30]
    assert result["ai_explanation"]["provider"] == "deterministic_fallback"


def test_analytics_contains_only_aggregated_synthetic_data():
    result = client.get("/api/workforce/analytics").json()
    assert result["workers"] == 6
    assert len(result["occupation_distribution"]) == 6
    assert "health" not in result


def test_llm_sanitization_omits_names_and_identifiers():
    safe = sanitize_for_llm({"worker": {"id": "ahmed", "name": "Ahmed", "age": 42, "occupation": "Construction Worker"}, "health": {}, "financial": {}, "occupational": {}})
    assert safe["worker"] == {"age": 42, "occupation": "Construction Worker"}


def test_ai_modes_fall_back_without_configuration(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    monkeypatch.delenv("FEATHERLESS_MODEL", raising=False)
    assessment = client.post("/api/assessment", json={
        "name": "Ahmed", "age": 42, "occupation": "Construction Worker",
        "health": client.get("/api/demo/workers/ahmed").json()["health"],
        "daily_income": 1800, "work_days_per_month": 24, "monthly_household_expenses": 32000,
        "current_savings": 18000, "health_savings": 3000, "dependents": 4,
        "estimated_medical_cost": 8000, "days_unable_to_work": 14,
    }).json()
    for mode in ("english", "simple", "roman_urdu"):
        result = client.post("/api/ai/explain", json={"assessment": assessment, "mode": mode}).json()
        assert result["provider"] == "deterministic_fallback"
        assert result["mode"] == mode
        assert result["priority_actions"]


@pytest.mark.parametrize("failure", [httpx.TimeoutException("timed out"), ValueError("malformed JSON")])
def test_ai_failure_uses_deterministic_fallback(monkeypatch, failure):
    assessment = {
        "worker": {"name": "Synthetic", "occupation": "Construction Worker"},
        "health": {"risk_probability": 0.2, "risk_level": "low", "top_factors": []},
        "financial": {"savings_runway_days": 10, "income_loss": 1000},
        "occupational": {"hazards": ["Heat exposure"]},
        "days_unable_to_work": 7,
    }

    async def fail_call(*args, **kwargs):
        raise failure

    monkeypatch.setattr(resilience_agent, "call_featherless", fail_call)
    result = asyncio.run(generate_explanation(assessment))
    assert result["provider"] == "deterministic_fallback"
    assert result["priority_actions"]