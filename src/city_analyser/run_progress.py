"""Live progress of a batch run: a small state file, and a picture of the city.

``ProgressTracker`` (used by ``city-analyser-analyze``) rewrites
``batch-progress.json`` every few seconds. ``progress_html`` and the
``city-analyser-progress`` command turn it, plus the run ledger, into a
self-contained page: a progress bar with an estimated time to finish, counts,
the latest errors, and a map of every panorama colored by status. The map is a
PNG rendered here, so it works in a browser, from Google Drive and inside a
notebook alike, however many panoramas there are.
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import io
import json
import math
import time
from pathlib import Path
from typing import Any

from .analysis_cache import atomic_write_json
from .run_ledger import DONE, FAILED, SKIPPED, ledger_path

PROGRESS_NAME = "batch-progress.json"
PROGRESS_VERSION = 1
# The rate is measured over the most recent completions, so an early cache-hit burst does not promise an impossible finish.
RATE_WINDOW = 40
WRITE_INTERVAL_S = 5.0
RECENT_ERRORS = 15
STATUS_COLORS = {"done": (42, 157, 143), "failed": (214, 40, 40), "skipped": (141, 153, 174), "pending": (120, 170, 220)}
STATUS_LABELS = {"done": "Done", "failed": "Failed (will retry)", "skipped": "Skipped (permanent)", "pending": "Pending"}


def progress_path(data_root: Path, dataset_id: str) -> Path:
    return data_root / dataset_id / PROGRESS_NAME


class ProgressTracker:
    """Counts, rate and ETA of one run, persisted atomically at most every few seconds."""

    def __init__(self, path: Path, total: int, already_done: int, meta: dict[str, Any], clock: Any = time.monotonic):
        self.path, self.total, self.meta, self._clock = path, total, meta, clock
        self.already_done = already_done
        self.done_now = self.failed = self.skipped = 0
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._started = clock()
        self._completions: collections.deque[float] = collections.deque(maxlen=RATE_WINDOW)
        self._errors: collections.deque[dict[str, str]] = collections.deque(maxlen=RECENT_ERRORS)
        self._last_write = float("-inf")
        self.state = "running"
        self.current: list[str] = []

    def completed(self) -> int:
        return self.already_done + self.done_now + self.skipped

    def rate_per_second(self) -> float | None:
        if len(self._completions) < 2:
            return None
        span = self._completions[-1] - self._completions[0]
        return (len(self._completions) - 1) / span if span > 0 else None

    def eta_seconds(self) -> float | None:
        rate = self.rate_per_second()
        remaining = self.total - self.completed()
        if remaining <= 0:
            return 0.0
        return remaining / rate if rate else None

    def finished_one(self, image_id: str, ok: bool, error: str | None = None, permanent: bool = False) -> None:
        if ok:
            self.done_now += 1
        elif permanent:
            self.skipped += 1
        else:
            self.failed += 1
        if ok or permanent:
            self._completions.append(self._clock())
        if error:
            self._errors.append({"image": image_id, "error": error[:300], "permanent": permanent, "at": time.strftime("%H:%M:%S")})
        self.write()

    def snapshot(self) -> dict[str, Any]:
        eta = self.eta_seconds()
        rate = self.rate_per_second()
        return {
            "version": PROGRESS_VERSION, "state": self.state, "started_at": self.started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(self._clock() - self._started, 1), "total": self.total, "completed": self.completed(),
            "done_before_this_run": self.already_done, "done_this_run": self.done_now, "failed": self.failed, "skipped": self.skipped,
            "rate_per_minute": round(rate * 60, 2) if rate else None, "eta_s": round(eta) if eta is not None else None,
            "recent_errors": list(self._errors), "current": self.current, **self.meta,
        }

    def write(self, force: bool = False) -> None:
        now = self._clock()
        if not force and now - self._last_write < WRITE_INTERVAL_S:
            return
        self._last_write = now
        try:
            atomic_write_json(self.path, self.snapshot())
        except OSError:
            pass  # progress is advisory; never stop the run because Drive hiccuped

    def close(self, state: str) -> None:
        self.state = state
        self.current = []
        self.write(force=True)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(seconds)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days} d {hours} h"
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def load_progress(data_root: Path, dataset_id: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(progress_path(data_root, dataset_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("version") == PROGRESS_VERSION else None


def image_statuses(data_root: Path, dataset_id: str, images: list[dict[str, Any]], stage: str = "analysis") -> dict[str, str]:
    """``done``, ``failed``, ``skipped`` or ``pending`` per panorama for one stage, from the latest ledger entry."""
    latest: dict[str, str] = {}
    try:
        lines = ledger_path(data_root, dataset_id).read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("stage") == stage and isinstance(entry.get("image"), str):
            latest[entry["image"]] = {DONE: "done", FAILED: "failed", SKIPPED: "skipped"}.get(str(entry.get("status")), "pending")
    return {str(image["id"]): latest.get(str(image["id"]), "pending") for image in images}


def _position(image: dict[str, Any]) -> tuple[float, float] | None:
    geometry = image.get("selected_geometry") or image.get("computed_geometry") or image.get("geometry")
    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if isinstance(coordinates, list) and len(coordinates) >= 2 and all(isinstance(v, (int, float)) for v in coordinates[:2]):
        return float(coordinates[0]), float(coordinates[1])
    return None


def render_map(images: list[dict[str, Any]], statuses: dict[str, str], width: int = 1100, height: int | None = None) -> bytes:
    """PNG of every panorama position, colored by status; pending is drawn first so finished work stays visible."""
    from PIL import Image, ImageDraw

    points = [(p, statuses.get(str(image["id"]), "pending")) for image in images if (p := _position(image)) is not None]
    if not points:
        return _png(Image.new("RGB", (width, height or 300), (20, 33, 61)))
    longitudes, latitudes = [p[0][0] for p in points], [p[0][1] for p in points]
    west, east, south, north = min(longitudes), max(longitudes), min(latitudes), max(latitudes)
    scale_x = math.cos(math.radians((south + north) / 2))
    span_x, span_y = max((east - west) * scale_x, 1e-9), max(north - south, 1e-9)
    margin = 24
    if height is None:
        # Follow the city's shape instead of leaving empty bands, within sensible bounds.
        height = int(min(max((width - 2 * margin) * span_y / span_x + 2 * margin, 260), 760))
    canvas = Image.new("RGB", (width, height), (20, 33, 61))
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    offset_x = (width - span_x * scale) / 2
    offset_y = (height - span_y * scale) / 2
    radius = 3 if len(points) < 20000 else 2 if len(points) < 100000 else 1
    draw = ImageDraw.Draw(canvas)
    for status in ("pending", "skipped", "done", "failed"):
        color = STATUS_COLORS[status]
        for (longitude, latitude), item in points:
            if item != status:
                continue
            x = offset_x + (longitude - west) * scale_x * scale
            y = height - (offset_y + (latitude - south) * scale)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return _png(canvas)


def _png(image: Any) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


_STYLE = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14213d;--muted:#5b6678;--line:#dfe3ea;--bar:#e6eaf0;--done:#2a9d8f;--failed:#d62828;--skipped:#8d99ae;--pending:#78aadc}
@media(prefers-color-scheme:dark){:root{--bg:#0b1426;--card:#14213d;--ink:#eef2f9;--muted:#a7b3c7;--line:#2a3a63;--bar:#0b1426}}
*{box-sizing:border-box}body{margin:0;padding:16px;overflow-wrap:anywhere;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;display:grid;gap:16px;grid-template-columns:minmax(0,1fr)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
h1{margin:0 0 4px;font-size:1.3rem}h2{margin:0 0 8px;font-size:1rem}.muted{color:var(--muted)}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.8rem;font-weight:600;color:#fff;background:var(--skipped)}
.badge.running{background:var(--done)}.badge.interrupted,.badge.aborted{background:var(--failed)}.badge.finished{background:#1d6fb8}
.bar{height:22px;background:var(--bar);border-radius:11px;overflow:hidden;display:flex}.bar span{display:block;height:100%}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px}
.stat b{display:block;font-size:1.25rem}.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:8px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
img{width:100%;height:auto;border-radius:8px;display:block}
ul.errors{list-style:none;margin:0;padding:0;font-size:.9rem}ul.errors li{padding:8px 0;border-bottom:1px solid var(--line)}ul.errors li:last-child{border-bottom:0}
code{font-size:.85em;word-break:break-all}
"""


def _manifest_images(data_root: Path, dataset_id: str) -> list[dict[str, Any]]:
    # Read directly: this module must import in a bare notebook kernel without the ML or GIS stack.
    path = data_root / dataset_id / "manifest.json"
    if not path.is_file():
        raise ValueError(f"no dataset at {path.parent}: download it first (city-analyser-download --output {path.parent}) or copy it there")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest: {path}") from error
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        raise ValueError(f"manifest has no image list: {dataset_id}")
    return [image for image in images if isinstance(image, dict) and image.get("local_file")]


def progress_html(data_root: Path, dataset_id: str, refresh_seconds: int | None = None) -> str:
    images = _manifest_images(data_root, dataset_id)
    progress = load_progress(data_root, dataset_id) or {}
    statuses = image_statuses(data_root, dataset_id, images, str(progress.get("stage", "analysis")))
    counts = collections.Counter(statuses.values())
    total = max(len(images), 1)
    state = str(progress.get("state", "not started"))
    eta = format_duration(progress.get("eta_s")) if state == "running" else "-"
    rate = progress.get("rate_per_minute")
    fraction = {name: 100 * counts.get(name, 0) / total for name in ("done", "skipped", "failed", "pending")}
    map_png = base64.b64encode(render_map(images, statuses)).decode("ascii")
    refresh = f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">' if refresh_seconds else ""
    legend = "".join(
        f'<span><span class="dot" style="background:var(--{name})"></span>{STATUS_LABELS[name]}: <b>{counts.get(name, 0):,}</b></span>'
        for name in ("done", "failed", "skipped", "pending")
    )
    errors = "".join(
        f"<li><code>{html.escape(str(item.get('image', '')))}</code> "
        f'<span class="muted">{"permanent, skipped" if item.get("permanent") else "will retry"} · {html.escape(str(item.get("at", "")))}</span>'
        f"<br>{html.escape(str(item.get('error', '')))}</li>"
        for item in reversed(progress.get("recent_errors", []))
    ) or '<li class="muted">No errors in this run.</li>'
    bar = "".join(f'<span style="width:{fraction[name]:.3f}%;background:var(--{name})"></span>' for name in ("done", "skipped", "failed"))
    percent = 100 * (counts.get("done", 0) + counts.get("skipped", 0)) / total
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<title>{html.escape(dataset_id)} progress</title><style>{_STYLE}</style></head><body><main>
<section class="card"><h1>{html.escape(dataset_id)} <span class="badge {html.escape(state)}">{html.escape(state)}</span></h1>
<div class="muted">Updated {html.escape(str(progress.get('updated_at', 'never')))} UTC · backend {html.escape(str(progress.get('backend', '?')))} · {html.escape(str(progress.get('stage', '')))}</div>
<div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent:.1f}" style="margin-top:12px">{bar}</div>
<div class="stats"><div class="stat"><b>{percent:.1f} %</b><span class="muted">{counts.get('done', 0) + counts.get('skipped', 0):,} of {len(images):,} resolved</span></div>
<div class="stat"><b>{html.escape(eta)}</b><span class="muted">estimated time left</span></div>
<div class="stat"><b>{rate if rate is not None else '-'}</b><span class="muted">panoramas / minute (recent)</span></div>
<div class="stat"><b>{html.escape(format_duration(progress.get('elapsed_s')))}</b><span class="muted">elapsed this run</span></div></div>
<div class="legend">{legend}</div></section>
<section class="card"><h2>Map of panoramas</h2><img alt="Map of panorama positions colored by status" src="data:image/png;base64,{map_png}"></section>
<section class="card"><h2>Latest errors</h2><ul class="errors">{errors}</ul></section>
</main></body></html>"""


def watch(data_root: Path, dataset_id: str, every: float = 15.0, output: Path | None = None, process: Any = None) -> None:
    """Regenerate the page until the run stops; inside a notebook the page is shown in place.

    ``process`` (a ``subprocess.Popen``) ends the watch when the run's process
    exits, even if it died before writing any progress.
    """
    try:
        from IPython.display import HTML, clear_output, display
    except ImportError:
        HTML = clear_output = display = None  # type: ignore[assignment]
    while True:
        page = progress_html(data_root, dataset_id)
        if output is not None:
            output.write_text(page, encoding="utf-8")
        if display is not None:
            clear_output(wait=True)
            display(HTML(page))
        state = (load_progress(data_root, dataset_id) or {}).get("state")
        if state in {"finished", "aborted", "interrupted"}:
            return
        if process is not None and process.poll() is not None:
            print(f"The run's process exited with code {process.returncode}; its log has the details.")
            return
        time.sleep(every)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the progress page of a batch run (map, ETA, errors).")
    parser.add_argument("dataset")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=None, help="HTML file to write (default: <dataset>/progress.html)")
    parser.add_argument("--watch", type=float, default=None, metavar="SECONDS", help="keep regenerating every SECONDS until the run stops")
    args = parser.parse_args()
    output = args.output or args.data_root / args.dataset / "progress.html"
    try:
        if args.watch:
            watch(args.data_root, args.dataset, args.watch, output)
        else:
            output.write_text(progress_html(args.data_root, args.dataset, refresh_seconds=None), encoding="utf-8")
    except ValueError as error:
        parser.error(str(error))
    progress = load_progress(args.data_root, args.dataset) or {}
    print(f"{output}  state={progress.get('state', 'not started')} completed={progress.get('completed', 0)}/{progress.get('total', '?')} eta={format_duration(progress.get('eta_s'))}")


if __name__ == "__main__":
    main()
