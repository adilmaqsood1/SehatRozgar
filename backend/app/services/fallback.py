def explain_assessment(assessment: dict, mode: str = "english") -> dict:
    worker = assessment["worker"]
    health = assessment["health"]
    financial = assessment["financial"]
    occupational = assessment["occupational"]
    runway = financial["savings_runway_days"]
    runway_text = f"about {round(runway)} days" if runway is not None else "an unbounded period at the entered expense level"
    risk_pct = round(health["risk_probability"] * 100)
    if mode == "roman_urdu":
        summary = f"{worker['name']} ke liye model ne {health['risk_level']} statistical pattern dikhaya. Maujooda bachat taqreeban {runway_text} ke ghar ke kharch cover kar sakti hai."
        financial_text = f"Agar kaam ruk jaye to {financial['income_loss']:,.0f} PKR ki estimated income loss aur medical kharch savings par asar dal sakte hain."
    elif mode == "simple":
        summary = f"The health model found a {health['risk_level']} pattern. Your savings may cover {runway_text} of household costs. This is not a diagnosis."
        financial_text = f"A {assessment['days_unable_to_work']}-day break could mean about PKR {financial['income_loss']:,.0f} in lost income, before other costs."
    else:
        summary = f"For {worker['name']}, the model identified a {health['risk_level']} statistical risk pattern ({risk_pct}%). Current savings cover {runway_text} of estimated household expenses."
        financial_text = f"A {assessment['days_unable_to_work']}-day work interruption means estimated lost income of PKR {financial['income_loss']:,.0f}, before other event costs."
    factors = health["top_factors"][:2]
    factor_names = ", ".join(item["feature"] for item in factors) or "the supplied indicators"
    return {
        "summary": summary,
        "health_explanation": f"The statistical model's estimate is {risk_pct}%. {factor_names} had the largest baseline-perturbation contributions. This is not a diagnosis or causal finding.",
        "financial_explanation": financial_text,
        "occupational_explanation": f"The rule-based profile for {worker['occupation']} lists {', '.join(occupational['hazards'][:3]).lower() or 'no mapped hazards'}. It does not predict injury.",
        "priority_actions": [
            {"priority": "high", "category": "health", "title": "Review health indicators", "action": "Discuss the supplied measurements with a qualified healthcare professional.", "reason": "The model is a research indicator and cannot diagnose."},
            {"priority": "high", "category": "financial", "title": "Plan for interrupted work", "action": "Compare the 14-day scenario with available savings and essential expenses.", "reason": "Lost workdays can add to household costs."},
            {"priority": "medium", "category": "workplace", "title": "Review workplace controls", "action": "Ask about practical controls and protective equipment for the listed hazards.", "reason": "Hazards are based on occupation rules, not injury prediction."},
        ],
        "questions_for_healthcare_professional": ["What do these measurements mean for my health?", "Should any measurement be checked again?"],
        "financial_actions": ["Set aside an emergency reserve where possible.", "Review the 7-day and 14-day interruption scenarios."],
        "workplace_actions": ["Discuss hazard controls and protective equipment with a supervisor or safety representative."],
        "disclaimer": "AI-generated explanations summarize system outputs and may contain errors. This is not medical advice or diagnosis.",
        "provider": "deterministic_fallback",
        "mode": mode,
    }