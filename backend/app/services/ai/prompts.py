import json


SYSTEM_PROMPT = """You are the explanation layer of SehatRozgar, a research prototype for worker health and financial resilience.
You receive structured outputs from an empirical machine-learning model, deterministic financial calculations, and occupational hazard rules. Only explain the structured information supplied by the system.
Do not diagnose disease, prescribe medication, invent measurements, invent or modify probabilities, calculate or change financial metrics, change vulnerability scores, or determine lending, insurance, employment, creditworthiness, or eligibility.
Use clear accessible language. Separate health, financial, and workplace explanations. Encourage discussion with a qualified healthcare professional without treatment instructions. Feature contributions describe model behavior and are not causal findings.
Return only valid JSON matching the requested schema."""


def build_messages(assessment: dict, mode: str) -> list[dict]:
    style = {
        "english": "Use concise, plain English.",
        "simple": "Use very simple English, short sentences, and avoid medical, finance, and machine-learning jargon.",
        "roman_urdu": "Explain all user-facing text in natural Roman Urdu. Do not translate numbers, units, or technical feature identifiers.",
    }[mode]
    schema = {
        "summary": "string",
        "health_explanation": "string",
        "financial_explanation": "string",
        "occupational_explanation": "string",
        "priority_actions": [{"priority": "high|medium|low", "category": "health|financial|workplace", "title": "string", "action": "string", "reason": "string"}],
        "questions_for_healthcare_professional": ["string"],
        "financial_actions": ["string"],
        "workplace_actions": ["string"],
        "disclaimer": "string",
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{style}\nReturn JSON only with this shape: {json.dumps(schema)}\nStructured assessment: {json.dumps(assessment, ensure_ascii=True)}"},
    ]