import json
from pathlib import Path


HAZARD_PATH = Path(__file__).resolve().parents[1] / "datasets" / "occupational_hazards.json"


def assess_occupation(occupation: str) -> dict:
    knowledge = json.loads(HAZARD_PATH.read_text(encoding="utf-8"))
    entry = knowledge.get(occupation.strip().lower())
    if entry is None:
        return {"risk_level": "not_mapped", "hazards": [], "note": "No configured hazard profile for this occupation."}
    return {"risk_level": entry["risk_level"], "hazards": entry["hazards"], "note": "Rule-based occupational hazard assessment; not an injury prediction."}