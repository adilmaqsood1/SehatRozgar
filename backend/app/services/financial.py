from dataclasses import dataclass


@dataclass(frozen=True)
class FinancialProfile:
    daily_income: float
    work_days_per_month: int
    monthly_household_expenses: float
    current_savings: float
    health_savings: float
    dependents: int
    estimated_medical_cost: float


def vulnerability_level(score: int) -> str:
    if score <= 30:
        return "resilient"
    if score <= 60:
        return "moderate"
    if score <= 80:
        return "high"
    return "critical"


def analyze_finances(profile: FinancialProfile, days_unable_to_work: int) -> dict:
    values = (
        profile.daily_income,
        profile.monthly_household_expenses,
        profile.current_savings,
        profile.health_savings,
        profile.estimated_medical_cost,
    )
    if any(value < 0 for value in values) or profile.work_days_per_month < 0 or profile.dependents < 0:
        raise ValueError("Financial values must be zero or greater")
    if days_unable_to_work < 0:
        raise ValueError("Days unable to work must be zero or greater")

    monthly_income = profile.daily_income * profile.work_days_per_month
    income_loss = profile.daily_income * days_unable_to_work
    gross_event_exposure = income_loss + profile.estimated_medical_cost
    health_savings_used = min(profile.health_savings, profile.estimated_medical_cost)
    net_event_exposure = max(gross_event_exposure - health_savings_used, 0)
    daily_burn = profile.monthly_household_expenses / 30
    runway = profile.current_savings / daily_burn if daily_burn else None
    emergency_target = profile.monthly_household_expenses * 3
    emergency_gap = max(emergency_target - profile.current_savings, 0)
    coverage_ratio = profile.current_savings / max(net_event_exposure, 1)
    remaining_savings = max(profile.current_savings - net_event_exposure, 0)
    remaining_runway = remaining_savings / daily_burn if daily_burn else None

    runway_risk = 0 if runway is None else 1 - min(runway / 90, 1)
    expense_ratio = profile.monthly_household_expenses / monthly_income if monthly_income else (1 if profile.monthly_household_expenses else 0)
    expense_risk = min(max((expense_ratio - 0.3) / 0.9, 0), 1)
    fund_gap_risk = emergency_gap / emergency_target if emergency_target else 0
    dependent_risk = min(profile.dependents / 6, 1)
    event_coverage_risk = 1 - min(coverage_ratio, 1) if net_event_exposure else 0
    interruption_ratio = net_event_exposure / monthly_income if monthly_income else (2 if net_event_exposure else 0)
    interruption_risk = min(interruption_ratio / 2, 1)
    components = {
        "savings_runway": round(runway_risk * 25, 1),
        "expense_to_income": round(expense_risk * 20, 1),
        "emergency_fund_gap": round(fund_gap_risk * 20, 1),
        "dependents": round(dependent_risk * 10, 1),
        "event_coverage": round(event_coverage_risk * 15, 1),
        "income_interruption": round(interruption_risk * 10, 1),
    }
    score = int(round(sum(components.values())))

    return {
        "monthly_income": round(monthly_income, 2),
        "income_loss": round(income_loss, 2),
        "gross_event_exposure": round(gross_event_exposure, 2),
        "health_savings_used": round(health_savings_used, 2),
        "net_event_exposure": round(net_event_exposure, 2),
        "daily_household_burn": round(daily_burn, 2),
        "savings_runway_days": round(runway, 1) if runway is not None else None,
        "emergency_fund_target": round(emergency_target, 2),
        "emergency_fund_gap": round(emergency_gap, 2),
        "event_coverage_ratio": round(coverage_ratio, 3),
        "remaining_savings": round(remaining_savings, 2),
        "remaining_runway_days": round(remaining_runway, 1) if remaining_runway is not None else None,
        "vulnerability_score": score,
        "vulnerability_level": vulnerability_level(score),
        "vulnerability_components": components,
        "disclaimer": "Informational only; not for lending, employment, insurance pricing, or eligibility decisions.",
    }


def simulate_scenarios(profile: FinancialProfile, medical_cost: float) -> list[dict]:
    scenarios = []
    for days in (3, 7, 14, 30):
        result = analyze_finances(
            FinancialProfile(
                daily_income=profile.daily_income,
                work_days_per_month=profile.work_days_per_month,
                monthly_household_expenses=profile.monthly_household_expenses,
                current_savings=profile.current_savings,
                health_savings=profile.health_savings,
                dependents=profile.dependents,
                estimated_medical_cost=medical_cost,
            ),
            days,
        )
        scenarios.append({
            "days_off_work": days,
            "lost_income": result["income_loss"],
            "medical_cost": medical_cost,
            "total_exposure": result["net_event_exposure"],
            "remaining_savings": result["remaining_savings"],
            "coverage_ratio": result["event_coverage_ratio"],
            "remaining_runway": result["remaining_runway_days"],
        })
    return scenarios