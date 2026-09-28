"""Dependency-light domain models for building inspection reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class InputIssue:
    code: str
    image_id: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ViewEvidence:
    image_id: str
    local_file: str
    source_page: str | None
    captured_at_ms: int | None
    recorded_longitude: float
    recorded_latitude: float
    recorded_heading_deg: float
    analysis_width: int
    analysis_height: int
    mask_ordinal: int
    observed_pixels: int
    inferred_pixels: int
    projection_coverage: float
    visible_facades: int
    footprint_source: str
    alternative_footprint_id: str | None
    footprint_geometry: dict[str, Any]
    pose_facade_iou: float | None
    pose_facade_iou_gain: float | None
    sfm_depth_samples: int
    sfm_depth_conflicts: int
    yaw_deg: float
    pitch_deg: float
    fov_deg: float
    analysis_relative_path: str
    analysis_sha256: str
    panorama_relative_path: str
    panorama_sha256: str
    id_mask_relative_path: str
    id_mask_sha256: str
    inferred_mask_relative_path: str
    inferred_mask_sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "captured_at_ms": self.captured_at_ms,
            "source_page": self.source_page,
            "camera": {
                "longitude": self.recorded_longitude,
                "latitude": self.recorded_latitude,
                "heading_deg": self.recorded_heading_deg,
            },
            "analysis": {
                "width": self.analysis_width,
                "height": self.analysis_height,
                "relative_path": self.analysis_relative_path,
                "sha256": self.analysis_sha256,
                "artifacts": [
                    {"path": self.panorama_relative_path, "sha256": self.panorama_sha256, "role": "source_panorama"},
                    {"path": self.id_mask_relative_path, "sha256": self.id_mask_sha256, "role": "building_id_mask"},
                    {"path": self.inferred_mask_relative_path, "sha256": self.inferred_mask_sha256, "role": "vegetation_inference_mask"},
                ],
            },
            "evidence": {
                "observed_pixels": self.observed_pixels,
                "inferred_pixels": self.inferred_pixels,
                "projection_coverage": self.projection_coverage,
                "visible_facades": self.visible_facades,
                "footprint_source": self.footprint_source,
                "alternative_footprint_id": self.alternative_footprint_id,
                "sfm_depth_samples": self.sfm_depth_samples,
                "sfm_depth_conflicts": self.sfm_depth_conflicts,
            },
            "pose_alignment": {
                "facade_iou": self.pose_facade_iou,
                "facade_iou_gain": self.pose_facade_iou_gain,
            },
        }


@dataclass(frozen=True, slots=True)
class EvidenceCollection:
    views: tuple[ViewEvidence, ...]
    manifest_panorama_count: int
    compatible_analysis_count: int
    stale_analysis_count: int
    incomplete_analysis_count: int
    invalid_analysis_count: int
    issues: tuple[InputIssue, ...]
    input_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class RankedView:
    rank: int
    evidence: ViewEvidence
    score: float
    observed_fraction: float
    inferred_fraction: float
    viewer_url: str
    reasons: tuple[str, ...]
    screenshot_url: str | None = None
    clean_screenshot_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = self.evidence.public_dict()
        payload.update({
            "rank": self.rank,
            "ranking": {
                "formula_version": "direct-facade-evidence-v1",
                "score": self.score,
                "observed_fraction": self.observed_fraction,
                "inferred_fraction": self.inferred_fraction,
                "reasons": list(self.reasons),
            },
            "viewer": {
                "url": self.viewer_url,
                "yaw_deg": self.evidence.yaw_deg,
                "pitch_deg": self.evidence.pitch_deg,
                "fov_deg": self.evidence.fov_deg,
                "framing_basis": "observed_mask_bounds",
            },
            "screenshot_url": self.screenshot_url,
            "clean_screenshot_url": self.clean_screenshot_url,
        })
        return payload


@dataclass(frozen=True, slots=True)
class AssessmentCheck:
    code: str
    status: str
    summary: str
    facts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
