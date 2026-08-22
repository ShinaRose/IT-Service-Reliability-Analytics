"""JSON schemas enforced on every structured AI task output."""
from __future__ import annotations

from relplatform.generator.postmortems import CATEGORIES

NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "narrative": {"type": "string", "description": "1-3 sentence plain-English summary of the incident"},
        "likely_origin_service": {"type": "string"},
        "order_of_events": {
            "type": "array", "items": {"type": "string"}, "minItems": 1,
            "description": "short phrases describing what fired, in chronological order",
        },
    },
    "required": ["narrative", "likely_origin_service", "order_of_events"],
    "additionalProperties": False,
}

ROOT_CAUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause_category": {"type": "string", "enum": CATEGORIES},
        "contributing_factors": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
        "preventive_action": {"type": "string"},
    },
    "required": ["root_cause_category", "contributing_factors", "preventive_action"],
    "additionalProperties": False,
}
