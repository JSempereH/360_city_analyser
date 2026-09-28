"""Explainable checks for image-to-footprint association."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from ..multi_view_analysis import maximum_independent_indices

from .models import AssessmentCheck, EvidenceCollection, RankedView


ASSESSMENT_VERSION = 1
MINIMUM_INDEPENDENT_BASELINE_M = 3.0


def geometry_fingerprint(geometry: dict[str, Any]) -> str:
    canonical = json.dumps(geometry, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _independent_views(ranked_views: tuple[RankedView, ...]) -> list[RankedView]:
    indices = maximum_independent_indices([
        (view.evidence.recorded_longitude, view.evidence.recorded_latitude) for view in ranked_views
    ])
    return [ranked_views[index] for index in indices]


def assess_footprint(building_id: str, ranked_views: tuple[RankedView, ...], collection: EvidenceCollection) -> dict[str, Any]:
    independent_views = _independent_views(ranked_views)
    variants: dict[str, list[RankedView]] = defaultdict(list)
    for view in ranked_views:
        variants[geometry_fingerprint(view.evidence.footprint_geometry)].append(view)
    if len(variants) > 1:
        conclusion_code = "mixed_geometry_association"
        conclusion_label = "Direct facade evidence was associated with multiple exact footprint geometry variants."
    elif len(independent_views) > 1:
        conclusion_code = "consistent_multi_view_association"
        conclusion_label = "The same exact footprint geometry has direct facade evidence in multiple panoramas."
    else:
        conclusion_code = "single_view_association"
        conclusion_label = "Direct facade evidence is available in one panorama only."

    sources = sorted({view.evidence.footprint_source for view in ranked_views})
    samples = sum(view.evidence.sfm_depth_samples for view in ranked_views)
    conflicts = sum(view.evidence.sfm_depth_conflicts for view in ranked_views)
    pose_ious = [view.evidence.pose_facade_iou for view in ranked_views if view.evidence.pose_facade_iou is not None]
    checks = [
        AssessmentCheck(
            "multi_view_support",
            "supporting" if len(independent_views) > 1 else "caution",
            f"Direct facade evidence appears in {len(independent_views)} spatially independent panorama{'s' if len(independent_views) != 1 else ''}.",
            {"matching_views": len(ranked_views), "independent_views": len(independent_views), "minimum_baseline_m": MINIMUM_INDEPENDENT_BASELINE_M, "compatible_analyses": collection.compatible_analysis_count},
        ),
        AssessmentCheck(
            "geometry_consistency",
            "supporting" if len(variants) == 1 else "caution",
            f"Matching views use {len(variants)} exact geometry variant{'s' if len(variants) != 1 else ''}.",
            {"variant_count": len(variants)},
        ),
        AssessmentCheck(
            "geometry_provenance",
            "informational",
            f"Geometry sources used by matching analyses: {', '.join(sources)}.",
            {"sources": sources},
        ),
        AssessmentCheck(
            "pose_alignment",
            "informational" if pose_ious else "not_available",
            "Facade alignment metrics are available." if pose_ious else "No facade alignment metric is available.",
            {"minimum_facade_iou": min(pose_ious) if pose_ious else None, "maximum_facade_iou": max(pose_ious) if pose_ious else None},
        ),
        AssessmentCheck(
            "sfm_depth_evidence",
            "supporting" if samples and not conflicts else "caution" if conflicts else "not_available",
            f"SfM contributed {samples} sparse samples with {conflicts} conflicts." if samples else "No sparse SfM samples intersected this footprint.",
            {"samples": samples, "conflicts": conflicts},
        ),
        AssessmentCheck(
            "input_completeness",
            "caution" if collection.stale_analysis_count or collection.incomplete_analysis_count or collection.invalid_analysis_count else "supporting",
            "Only compatible and complete analysis artifacts contribute to this report.",
            {
                "manifest_panoramas": collection.manifest_panorama_count,
                "compatible_analyses": collection.compatible_analysis_count,
                "stale_analyses": collection.stale_analysis_count,
                "incomplete_analyses": collection.incomplete_analysis_count,
                "invalid_analyses": collection.invalid_analysis_count,
            },
        ),
    ]
    geometry_variants = []
    for fingerprint, views in sorted(variants.items()):
        first = views[0].evidence
        geometry_variants.append({
            "fingerprint_sha256": fingerprint,
            "footprint_source": first.footprint_source,
            "alternative_footprint_id": first.alternative_footprint_id,
            "image_ids": [view.evidence.image_id for view in views],
            "geometry": first.footprint_geometry,
        })
    return {
        "version": ASSESSMENT_VERSION,
        "scope": "image_to_footprint_association",
        "building_id": building_id,
        "conclusion": {"code": conclusion_code, "label": conclusion_label},
        "checks": [check.to_dict() for check in checks],
        "geometry_variants": geometry_variants,
        "limitations": [
            "This assessment concerns image-to-footprint association, not surveyed positional accuracy.",
            "Projection coverage is not a calibrated probability that the footprint is correct.",
            "Unanalyzed panoramas provide no positive or negative evidence.",
            "Street imagery cannot establish hidden structural properties.",
        ],
        "non_assessments": [{
            "code": "gem_structural_classification",
            "status": "not_performed",
            "reason": "The current evidence pipeline does not establish or calibrate the attributes required for GEM classification.",
        }],
    }
