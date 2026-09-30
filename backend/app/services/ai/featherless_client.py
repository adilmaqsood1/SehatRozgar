import json
import os
import re

import httpx

from app.services.ai.prompts import build_messages
from app.services.ai.schemas import AIExplanation


def sanitize_for_llm(assessment: dict) -> dict:
    worker = assessment.get("worker", {})
    health = assessment.get("health", {})
    financial = assessment.get("financial", {})
    occupational = assessment.get("occupational", {})
    return {
        "worker": {key: worker[key] for key in ("age", "occupation") if key in worker},
        "health": {
            key: health[key] for key in ("risk_probability", "risk_level", "top_factors") if key in health
        },
        "financial": {
            key: financial[key] for key in (
                "income_loss", "days_unable_to_work", "savings_runway_days",
                "vulnerability_score", "vulnerability_level", "emergency_fund_gap",
            ) if key in financial
        },
        "occupational": {
            key: occupational[key] for key in ("risk_level", "hazards") if key in occupational
        },
    }


async def call_featherless(assessment: dict, mode: str) -> AIExplanation:
    api_key = os.getenv("FEATHERLESS_API_KEY", "").strip()
    base_url = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1").rstrip("/")
    model = os.getenv("FEATHERLESS_MODEL", "").strip()
    if not api_key or not model:
        raise RuntimeError("Featherless is not configured")
    payload = {"model": model, "messages": build_messages(sanitize_for_llm(assessment), mode), "temperature": 0.2, "response_format": {"type": "json_object"}}
    async with httpx.AsyncClient(timeout=8.0) as client:
        last_error = None
        for attempt in range(2):
            try:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
                return AIExplanation.model_validate(json.loads(content))
            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError, KeyError, IndexError, TypeError, ValueError) as error:
                last_error = error
                if attempt == 0 and isinstance(error, (httpx.TimeoutException, httpx.RequestError)):
                    continue
                break
    raise RuntimeError(f"Featherless explanation unavailable: {last_error}")