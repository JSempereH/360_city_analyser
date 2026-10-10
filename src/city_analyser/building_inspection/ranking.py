"""Deterministic ranking of direct facade evidence."""

from __future__ import annotations

from urllib.parse import urlencode

from ..image_quality import rank_weight
from .models import RANKING_FORMULA_VERSION, RankedView, ViewEvidence

__all__ = ["RANKING_FORMULA_VERSION", "build_viewer_url", "rank_views", "view_evidence_score", "view_score"]


def view_score(observed_fraction: float, projection_coverage: float, image_quality: dict | None) -> float:
    """The one definition of a view's score, shared by reports and the building table."""
    return observed_fraction * (0.25 + 0.75 * projection_coverage) * rank_weight(image_quality)


def view_evidence_score(evidence: ViewEvidence) -> float:
    observed_fraction = evidence.observed_pixels / (evidence.analysis_width * evidence.analysis_height)
    return view_score(observed_fraction, evidence.projection_coverage, evidence.image_quality)


def build_viewer_url(dataset_id: str, building_id: str, evidence: ViewEvidence) -> str:
    query = urlencode({
        "dataset": dataset_id,
        "image": evidence.image_id,
        "building": building_id,
        "yaw": f"{evidence.yaw_deg:.2f}",
        "pitch": f"{evidence.pitch_deg:.2f}",
        "fov": f"{evidence.fov_deg:.2f}",
    })
    return f"/viewer/?{query}"


def rank_views(evidence: tuple[ViewEvidence, ...], dataset_id: str, building_id: str) -> tuple[RankedView, ...]:
    ordered = sorted(
        evidence,
        key=lambda item: (
            -view_evidence_score(item),
            -item.observed_pixels,
            -item.projection_coverage,
            item.captured_at_ms if item.captured_at_ms is not None else 2**63 - 1,
            item.image_id,
        ),
    )
    ranked = []
    for rank, item in enumerate(ordered, start=1):
        analysis_pixels = item.analysis_width * item.analysis_height
        observed_fraction = item.observed_pixels / analysis_pixels
        inferred_fraction = item.inferred_pixels / analysis_pixels
        reasons = [
            f"Direct facade evidence occupies {observed_fraction:.2%} of the analysis panorama.",
            f"Direct evidence covers {item.projection_coverage:.1%} of the projected visible facade.",
        ]
        if item.inferred_pixels:
            reasons.append(f"{item.inferred_pixels:,} vegetation-occluded pixels are reported separately and do not increase the rank.")
        flags = (item.image_quality or {}).get("flags")
        if flags:
            reasons.append(f"Image quality flags ({', '.join(flags)}) halve its score; the view stays listed.")
        if item.sfm_depth_samples:
            reasons.append(f"SfM checked {item.sfm_depth_samples} sparse depth samples; {item.sfm_depth_conflicts} conflicted.")
        ranked.append(RankedView(
            rank=rank,
            evidence=item,
            score=round(view_evidence_score(item), 8),
            observed_fraction=round(observed_fraction, 8),
            inferred_fraction=round(inferred_fraction, 8),
            viewer_url=build_viewer_url(dataset_id, building_id, item),
            reasons=tuple(reasons),
        ))
    return tuple(ranked)
