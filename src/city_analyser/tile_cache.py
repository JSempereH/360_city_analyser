"""OpenStreetMap map tiles kept on disk, so the viewer's maps work without a network once an area was seen.

The viewer asks ``/tiles/{z}/{x}/{y}.png``. A tile on disk is served from there; one that is not is
fetched once from tile.openstreetmap.org, with an identifying User-Agent as its usage policy asks,
and kept. Nothing is prefetched: the cache holds what was viewed, as a browser cache would, and a
rehearsal of a presentation fills it for that area. Without a network a missing tile is a 404 and
the map shows its background there.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .analysis_cache import atomic_write_bytes

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "360-city-analyser-viewer/1.0 (local viewer; https://github.com/JSempereH/360_city_analyser)"
TILE_PATH = re.compile(r"/tiles/(\d{1,2})/(\d{1,7})/(\d{1,7})\.png")
MAX_ZOOM = 19
MAX_TILE_BYTES = 1_000_000


def tile_cache_directory(data_root: Path) -> Path:
    configured = os.environ.get("CITY_ANALYSER_TILE_CACHE_DIR", "").strip()
    return Path(configured) if configured else data_root / ".tile-cache"


def tile(data_root: Path, z: int, x: int, y: int, *, fetch: bool = True) -> bytes | None:
    """The PNG of one tile, from the cache or fetched once and kept; None when unknown and unreachable."""
    if not 0 <= z <= MAX_ZOOM or not (0 <= x < 2**z and 0 <= y < 2**z):
        return None
    path = tile_cache_directory(data_root) / str(z) / str(x) / f"{y}.png"
    try:
        return path.read_bytes()
    except OSError:
        pass
    if not fetch or os.environ.get("CITY_ANALYSER_OFFLINE", "").strip() == "1":
        return None
    try:
        with urlopen(Request(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT}), timeout=10) as response:
            content = response.read(MAX_TILE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError):
        return None
    if len(content) > MAX_TILE_BYTES or not content.startswith(b"\x89PNG"):
        return None
    try:
        atomic_write_bytes(path, content)
    except OSError:
        pass
    return content
