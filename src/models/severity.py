"""
Severity and harm helpers for misinformation scenarios.

This module is intentionally separate from the IC simulator so the existing
binary pipeline can keep using infection-count objectives unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Mapping, Optional

SeverityLabel = Literal["binary", "low", "medium", "high"]
PropagationScaleMode = Literal["constant", "bucket", "linear"]


@dataclass(frozen=True)
class SeverityProfile:
    """
    Scenario-level misinformation severity.

    Parameters
    ----------
    severity_score:
        Continuous score in ``[0, 1]``.
    severity_label:
        Discrete bucket for reporting.
    base_harm_weight:
        Per-infected-node harm multiplier used by harm-aware evaluation.
    propagation_multiplier:
        Optional scale applied to the base IC propagation probability.
    construction_source:
        Human-readable provenance for the severity assignment.
    notes:
        Free-text notes for docs / logs.
    """

    severity_score: float
    severity_label: SeverityLabel
    base_harm_weight: float
    propagation_multiplier: float
    construction_source: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeverityRubric:
    """
    Simple, documented score thresholds and weights.
    """

    low_max: float = 0.33
    medium_max: float = 0.66
    harm_weight_low: float = 1.0
    harm_weight_medium: float = 1.6
    harm_weight_high: float = 2.4
    propagation_scale_low: float = 0.9
    propagation_scale_medium: float = 1.0
    propagation_scale_high: float = 1.1
    linear_scale_center: float = 1.0
    linear_scale_slope: float = 0.4
    linear_scale_min: float = 0.75
    linear_scale_max: float = 1.25


DEFAULT_SEVERITY_RUBRIC = SeverityRubric()


def clip_unit_interval(value: float) -> float:
    """Clip a float into ``[0, 1]``."""
    return max(0.0, min(1.0, float(value)))


def severity_label_from_score(
    score: float,
    rubric: SeverityRubric = DEFAULT_SEVERITY_RUBRIC,
) -> SeverityLabel:
    """Map a continuous score to a report-friendly bucket."""
    s = clip_unit_interval(score)
    if s <= rubric.low_max:
        return "low"
    if s <= rubric.medium_max:
        return "medium"
    return "high"


def harm_weight_from_label(
    label: SeverityLabel,
    rubric: SeverityRubric = DEFAULT_SEVERITY_RUBRIC,
) -> float:
    """Default per-infected-node harm multiplier for a severity bucket."""
    if label == "binary":
        return 1.0
    if label == "low":
        return rubric.harm_weight_low
    if label == "medium":
        return rubric.harm_weight_medium
    if label == "high":
        return rubric.harm_weight_high
    raise ValueError(f"Unknown severity label: {label}")


def propagation_multiplier_from_score(
    score: float,
    *,
    mode: PropagationScaleMode = "constant",
    rubric: SeverityRubric = DEFAULT_SEVERITY_RUBRIC,
) -> float:
    """
    Convert severity to a propagation multiplier.

    ``constant`` preserves the existing binary IC behavior.
    """
    s = clip_unit_interval(score)
    if mode == "constant":
        return 1.0
    if mode == "bucket":
        label = severity_label_from_score(s, rubric)
        if label == "low":
            return rubric.propagation_scale_low
        if label == "medium":
            return rubric.propagation_scale_medium
        return rubric.propagation_scale_high
    if mode == "linear":
        raw = rubric.linear_scale_center + rubric.linear_scale_slope * (s - 0.5)
        return max(rubric.linear_scale_min, min(rubric.linear_scale_max, raw))
    raise ValueError(f"Unknown propagation scale mode: {mode}")


def make_binary_profile() -> SeverityProfile:
    """Compatibility profile for the original binary experiments."""
    return SeverityProfile(
        severity_score=0.0,
        severity_label="binary",
        base_harm_weight=1.0,
        propagation_multiplier=1.0,
        construction_source="binary_baseline",
        notes="Backward-compatible binary IC configuration.",
    )


def make_profile_from_score(
    score: float,
    *,
    construction_source: str,
    propagation_mode: PropagationScaleMode = "constant",
    notes: str = "",
    rubric: SeverityRubric = DEFAULT_SEVERITY_RUBRIC,
) -> SeverityProfile:
    """
    Create a reusable scenario-level severity profile.
    """
    s = clip_unit_interval(score)
    label = severity_label_from_score(s, rubric)
    return SeverityProfile(
        severity_score=s,
        severity_label=label,
        base_harm_weight=harm_weight_from_label(label, rubric),
        propagation_multiplier=propagation_multiplier_from_score(
            s,
            mode=propagation_mode,
            rubric=rubric,
        ),
        construction_source=construction_source,
        notes=notes,
    )


def profile_from_mapping(
    data: Mapping[str, Any],
    *,
    default_source: str = "mapping",
) -> SeverityProfile:
    """Rehydrate a profile from JSON / CSV records."""
    score = float(data.get("severity_score", 0.0))
    label = str(data.get("severity_label", severity_label_from_score(score)))
    base_harm_weight = float(
        data.get(
            "base_harm_weight",
            harm_weight_from_label(label if label != "binary" else "binary"),
        )
    )
    propagation_multiplier = float(data.get("propagation_multiplier", 1.0))
    return SeverityProfile(
        severity_score=score,
        severity_label=label,  # type: ignore[arg-type]
        base_harm_weight=base_harm_weight,
        propagation_multiplier=propagation_multiplier,
        construction_source=str(data.get("construction_source", default_source)),
        notes=str(data.get("notes", "")),
    )


def scenario_profile_or_binary(
    maybe_data: Optional[Mapping[str, Any]],
    *,
    propagation_mode: PropagationScaleMode = "constant",
) -> SeverityProfile:
    """
    Convert optional scenario metadata into a usable profile.
    """
    if maybe_data is None:
        return make_binary_profile()
    if "severity_score" in maybe_data:
        return make_profile_from_score(
            float(maybe_data["severity_score"]),
            construction_source=str(maybe_data.get("construction_source", "scenario")),
            propagation_mode=propagation_mode,
            notes=str(maybe_data.get("notes", "")),
        )
    return profile_from_mapping(maybe_data)
