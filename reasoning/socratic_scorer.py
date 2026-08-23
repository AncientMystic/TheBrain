"""
Socratic Protocol / PSYOP document scorer.

Applies the 20-criteria NCI PSYOP scoring system and the Systemic Bias Addendum.
The result is stored as provenance metadata, not truth status.
"""
import json
import time
import config
from core.llm import call_model_json


SCORING_CRITERIA = [
    "timing",
    "emotional_manipulation",
    "uniform_messaging",
    "missing_information",
    "simplistic_narratives",
    "tribal_division",
    "authority_overload",
    "call_for_urgent_action",
    "overuse_of_novelty",
    "financial_political_gain",
    "suppression_of_dissent",
    "false_dilemmas",
    "bandwagon_effect",
    "emotional_repetition",
    "cherry_picked_data",
    "logical_fallacies",
    "manufactured_outrage",
    "framing_techniques",
    "rapid_behavior_shifts",
    "historical_parallels",
]


SOCRATIC_SYSTEM_PROMPT = (
    "You are a rigorous epistemic analyst. "
    "Apply the Socratic Protocol with the 20-criteria PSYOP scoring system "
    "and the Systemic Bias Addendum. Always return only valid JSON."
)


SOCRATIC_PROMPT = """
Analyze the following document excerpt.

Document title: {title}
Metadata: {metadata}

Excerpt:
{excerpt}

Score every criterion from 1 to 5:
1 = not present
2 = mildly present
3 = moderately present
4 = strongly present
5 = overwhelmingly present
If insufficient evidence exists, set the criterion to null.

Also assess:
- source_hierarchy_level: 1-9, where 1 is raw primary source and 9 is AI-generated summary.
- data_model_policy: one of "data", "model", "policy", "unknown"
- enforcement_vector: short description if an enforcement arm is identified, otherwise null
- intentionality_triad: object with boolean fields "one_directional_bias", "coordinated_action", "institutional_self_preservation"
- lived_experience_cluster: boolean
- funding_gatekeeping_flags: object with boolean fields "funding_disclosed", "funder_has_interest", "peer_review_gatekeeping", "political_homogeneity", "suppression_of_dissent"
- summary: one paragraph summary of the assessment.

Return JSON with exactly these keys:
{{"psych_scores": {{"timing": 1, "emotional_manipulation": 1, "uniform_messaging": 1, "missing_information": 1, "simplistic_narratives": 1, "tribal_division": 1, "authority_overload": 1, "call_for_urgent_action": 1, "overuse_of_novelty": 1, "financial_political_gain": 1, "suppression_of_dissent": 1, "false_dilemmas": 1, "bandwagon_effect": 1, "emotional_repetition": 1, "cherry_picked_data": 1, "logical_fallacies": 1, "manufactured_outrage": 1, "framing_techniques": 1, "rapid_behavior_shifts": 1, "historical_parallels": 1}},
 "psych_score_total": 0,
 "source_hierarchy_level": 0,
 "data_model_policy": "unknown",
 "enforcement_vector": null,
 "intentionality_triad": {{}},
 "lived_experience_cluster": false,
 "funding_gatekeeping_flags": {{}},
 "summary": ""
}}
"""


def _fallback_assessment(reason=""):
    return {
        "psych_scores": {},
        "psych_score_total": 0,
        "source_hierarchy_level": 9,
        "data_model_policy": "unknown",
        "enforcement_vector": None,
        "intentionality_triad": {},
        "lived_experience_cluster": False,
        "funding_gatekeeping_flags": {},
        "summary": f"Socratic assessment unavailable: {reason}",
    }


def score_document(title: str, text: str, metadata: dict = None, source_hierarchy_level: int = None):
    """
    Score a document using the Socratic/PSYOP prompt.
    Returns a dict. If the model fails, returns a fallback assessment.
    """
    excerpt = (text or "")[:8000]
    if not excerpt.strip():
        return _fallback_assessment("empty document")

    prompt = SOCRATIC_PROMPT.format(
        title=title or "",
        metadata=json.dumps(metadata or {}, default=str),
        excerpt=excerpt,
    )

    try:
        data = call_model_json(
            prompt,
            max_tokens=2048,
            system=SOCRATIC_SYSTEM_PROMPT,
            unwrap_list=False,
            endpoint_type=None,
        )
    except Exception as e:
        if config.DEBUG_VERBOSE:
            print(f"    (Socratic scorer exception: {e})")
        data = None

    if not isinstance(data, dict):
        return _fallback_assessment("LLM did not return JSON")

    assessment = _fallback_assessment("incomplete LLM response")
    assessment.update(data)

    psych_scores = assessment.get("psych_scores") or {}
    if not isinstance(psych_scores, dict):
        psych_scores = {}
    total = 0
    for k in SCORING_CRITERIA:
        v = psych_scores.get(k)
        if isinstance(v, (int, float)):
            total += max(0, min(5, float(v)))
    if "psych_score_total" not in data or not isinstance(data.get("psych_score_total"), (int, float)):
        assessment["psych_score_total"] = total
    else:
        total = data["psych_score_total"]

    if assessment.get("enforcement_vector"):
        original = assessment.get("psych_score_total", total)
        elevated = min(100, original + 10)
        assessment["psych_score_total"] = elevated
        assessment["psych_score_total_before_enforcement_vector"] = original
        assessment["enforcement_vector_elevation_applied"] = True

    if source_hierarchy_level is not None:
        assessment["source_hierarchy_level"] = source_hierarchy_level

    assessment["scored_at"] = time.time()
    return assessment
