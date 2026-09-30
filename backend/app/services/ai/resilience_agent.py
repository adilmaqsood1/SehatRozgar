from app.services.ai.featherless_client import call_featherless
from app.services.fallback import explain_assessment


async def generate_explanation(assessment: dict, mode: str = "english") -> dict:
    try:
        result = await call_featherless(assessment, mode)
        return {**result.model_dump(), "provider": "featherless", "mode": mode}
    except Exception:
        fallback = explain_assessment(assessment, mode)
        return {**fallback, "provider": "deterministic_fallback", "mode": mode}