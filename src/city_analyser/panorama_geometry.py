"""Shared equirectangular panorama projection helpers."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any


def perspective_tile(
    panorama: Any,
    yaw: float,
    pitch: float,
    size: int,
    field_of_view_degrees: float,
    height: int | None = None,
) -> tuple[Any, Any, Any]:
    """Sample a rectilinear view and return its panorama source coordinates.

    ``size`` and ``field_of_view_degrees`` are the width and horizontal field
    of view; ``height`` (default: square) extends the view vertically with
    the same focal length, e.g. a portrait tile spanning a facade base to roof.
    """
    np = __import__("numpy")
    source_height, source_width = panorama.shape[:2]
    height = size if height is None else height
    aspect = height / size
    columns = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 - 1
    rows = (np.arange(height, dtype=np.float32) + 0.5) / height * 2 - 1
    horizontal, vertical = np.meshgrid(columns, -rows)
    field = math.tan(math.radians(field_of_view_degrees) / 2)
    direction_x = horizontal * field
    direction_y = vertical * field * aspect
    direction_z = np.ones_like(direction_x)
    magnitude = np.sqrt(direction_x ** 2 + direction_y ** 2 + direction_z ** 2)
    direction_x /= magnitude
    direction_y /= magnitude
    direction_z /= magnitude
    pitched_y = math.cos(pitch) * direction_y + math.sin(pitch) * direction_z
    pitched_z = -math.sin(pitch) * direction_y + math.cos(pitch) * direction_z
    latitude = np.arcsin(np.clip(pitched_y, -1, 1))
    longitude = np.arctan2(direction_x, pitched_z) + yaw
    source_x = ((longitude / (2 * math.pi) + 0.5) % 1 * source_width).astype(np.int32)
    source_y = np.clip((0.5 - latitude / math.pi) * source_height, 0, source_height - 1).astype(np.int32)
    return panorama[source_y, source_x], source_x, source_y


@lru_cache(maxsize=64)
def _tile_sampling_grid(
    torch: Any, width: int, height: int, yaw: float, pitch: float, size: int, field_of_view_degrees: float,
    tile_height: int, device: Any,
) -> Any:
    """``grid_sample`` grid that renders one view from a wrap-padded panorama.

    Same rays as ``perspective_tile``. The panorama is padded with one wrapped
    column on each side (``width + 2``) so bilinear sampling is continuous
    across the +/-180 degree seam.
    """
    aspect = tile_height / size
    columns = (torch.arange(size, device=device, dtype=torch.float64) + 0.5) / size * 2 - 1
    rows = (torch.arange(tile_height, device=device, dtype=torch.float64) + 0.5) / tile_height * 2 - 1
    vertical, horizontal = torch.meshgrid(-rows, columns, indexing="ij")
    field = math.tan(math.radians(field_of_view_degrees) / 2)
    direction_x = horizontal * field
    direction_y = vertical * field * aspect
    direction_z = torch.ones_like(direction_x)
    magnitude = torch.sqrt(direction_x ** 2 + direction_y ** 2 + direction_z ** 2)
    direction_x, direction_y, direction_z = direction_x / magnitude, direction_y / magnitude, direction_z / magnitude
    pitched_y = math.cos(pitch) * direction_y + math.sin(pitch) * direction_z
    pitched_z = -math.sin(pitch) * direction_y + math.cos(pitch) * direction_z
    latitude = torch.asin(pitched_y.clamp(-1, 1))
    longitude = torch.atan2(direction_x, pitched_z) + yaw
    # Continuous source coordinates in pixel-edge units (pixel i spans [i, i + 1]).
    source_x = torch.remainder(longitude / (2 * math.pi) + 0.5, 1.0) * width
    source_y = (0.5 - latitude / math.pi) * height
    grid_x = (source_x + 1) / (width + 2) * 2 - 1
    grid_y = source_y / height * 2 - 1
    return torch.stack((grid_x, grid_y), dim=-1).float()[None]


def perspective_tiles_torch(
    torch: Any,
    panorama: Any,
    views: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    size: int,
    field_of_view_degrees: float,
    height: int | None = None,
) -> Any:
    """Bilinear rectilinear views of a ``(1, C, H, W)`` float panorama, on its device.

    Returns ``(len(views), C, height, size)``. The batched counterpart of
    ``perspective_tile`` for model input; ``perspective_tile`` stays the
    reference for screenshots and SfM.
    """
    height = size if height is None else height
    source_height, source_width = panorama.shape[-2:]
    padded = torch.cat((panorama[..., -1:], panorama, panorama[..., :1]), dim=-1)
    grids = torch.cat([
        _tile_sampling_grid(torch, source_width, source_height, float(yaw), float(pitch), size, float(field_of_view_degrees), height, panorama.device)
        for yaw, pitch in views
    ])
    return torch.nn.functional.grid_sample(
        padded.expand(len(views), -1, -1, -1), grids, mode="bilinear", padding_mode="border", align_corners=False,
    )


# Grids depend only on the geometry, and every panorama of one analysis
# resolution reuses them. Callers must treat the returned tensors as read-only.
@lru_cache(maxsize=32)
def panorama_to_tile_grid(
    torch: Any, width: int, height: int, yaw: float, pitch: float, field_of_view_degrees: float, device: Any,
    aspect: float = 1.0,
) -> tuple[Any, Any]:
    """Inverse of ``perspective_tile``: where every panorama pixel falls in one tile.

    Returns a ``grid_sample`` grid of shape (1, height, width, 2) in normalized
    tile coordinates (``align_corners=False``) and a boolean mask of the
    panorama pixels the tile actually sees. Pulling tile values per panorama
    pixel leaves no holes, unlike scattering tile pixels into the panorama.
    ``aspect`` is the tile's height / width, matching ``perspective_tile``.
    """
    rows = torch.arange(height, device=device, dtype=torch.float32)
    columns = torch.arange(width, device=device, dtype=torch.float32)
    latitude = (0.5 - (rows + 0.5) / height) * math.pi
    longitude = ((columns + 0.5) / width - 0.5) * 2 * math.pi - yaw
    cos_latitude = torch.cos(latitude)[:, None]
    world_x = cos_latitude * torch.sin(longitude)[None, :]
    world_y = torch.sin(latitude)[:, None].expand(height, width)
    world_z = cos_latitude * torch.cos(longitude)[None, :]
    tile_y = math.cos(pitch) * world_y - math.sin(pitch) * world_z
    tile_z = math.sin(pitch) * world_y + math.cos(pitch) * world_z
    field = math.tan(math.radians(field_of_view_degrees) / 2)
    safe_z = torch.where(tile_z > 1e-6, tile_z, torch.ones_like(tile_z))
    horizontal = world_x / (safe_z * field)
    vertical = -tile_y / (safe_z * field * aspect)
    visible = (tile_z > 1e-6) & (horizontal.abs() <= 1) & (vertical.abs() <= 1)
    return torch.stack((horizontal, vertical), dim=-1)[None], visible
