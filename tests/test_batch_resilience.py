import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from city_analyser import batch_analysis
from city_analyser.analysis_backend import AnalysisBackend
from city_analyser.run_ledger import RunLedger, ledger_path, settings_fingerprint
from city_analyser.run_progress import ProgressTracker, format_duration, image_statuses, load_progress, progress_html, render_map

LOCAL = AnalysisBackend("local")


def _dataset(root: Path, count: int = 4) -> list[dict]:
    (root / "set" / "images").mkdir(parents=True)
    images = []
    for number in range(count):
        (root / "set" / "images" / f"{number}.jpg").write_bytes(b"jpeg" * (number + 1))
        images.append({
            "id": str(number), "local_file": f"images/{number}.jpg", "computed_compass_angle_deg": 0,
            "computed_geometry": {"type": "Point", "coordinates": [4.9 + number * 1e-4, 52.3]},
        })
    (root / "set" / "manifest.json").write_text(json.dumps({"images": images}), encoding="utf-8")
    return images


class Harness:
    """Patches the model and OSM steps and counts how often each runs."""

    def __init__(self, failing=(), permanent=(), flaky=None):
        self.perceived: list[str] = []
        self.associated: list[str] = []
        self.failing, self.permanent, self.flaky = set(failing), set(permanent), dict(flaky or {})

    def perceive(self, backend, data_root, model_root, image_path, progress):
        name = image_path.stem
        self.perceived.append(name)
        if name in self.permanent:
            raise ValueError("bad panorama")
        if name in self.failing:
            raise RuntimeError("gpu fell over")
        if self.flaky.get(name, 0) > 0:
            self.flaky[name] -= 1
            raise RuntimeError("transient")
        return "perception"

    def associate(self, data_root, dataset_id, image, inputs, perception, features, progress, sfm_refinement=None):
        self.associated.append(image["id"])
        return {"building_ids": {"way/1": {}}}

    def patches(self):
        return (
            patch.object(batch_analysis, "perceive", side_effect=self.perceive),
            patch.object(batch_analysis, "associate_panorama", side_effect=self.associate),
            patch.object(batch_analysis, "nearby_buildings", return_value=[]),
            patch.object(batch_analysis, "RETRY_BACKOFF_S", 0.0),
            patch("builtins.print"),
        )


def run(root: Path, harness: Harness, **options):
    patches = harness.patches()
    for item in patches:
        item.start()
    try:
        return batch_analysis.analyze_dataset(root, root, "set", backend=LOCAL, **options)
    finally:
        for item in patches:
            item.stop()


class ResumeTests(unittest.TestCase):
    def test_second_run_does_no_work_and_does_not_validate_from_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            first = Harness()
            run(root, first)
            second = Harness()
            with patch.object(batch_analysis, "load_analysis", side_effect=AssertionError("ledger should answer")):
                report = run(root, second)
        self.assertEqual(sorted(first.associated), ["0", "1", "2", "3"])
        self.assertEqual((second.perceived, second.associated), ([], []))
        self.assertEqual((report["already_done"], report["panoramas"], report["state"]), (4, 0, "finished"))

    def test_changed_settings_or_a_replaced_image_are_redone_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            run(root, Harness())
            (root / "set" / "images" / "2.jpg").write_bytes(b"a different and longer image")
            replaced = Harness()
            run(root, replaced)
            self.assertEqual(replaced.associated, ["2"])
            with patch.dict(os.environ, {"BUILDING_ANALYSIS_MIN_CONFIDENCE": "0.41"}):
                reconfigured = Harness()
                run(root, reconfigured)
        self.assertEqual(sorted(reconfigured.associated), ["0", "1", "2", "3"])

    def test_force_redoes_everything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 2)
            run(root, Harness())
            forced = Harness()
            run(root, forced, force=True)
        self.assertEqual(sorted(forced.associated), ["0", "1"])

    def test_analyses_made_before_the_ledger_existed_are_validated_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 2)
            with patch.object(batch_analysis, "load_analysis", return_value={"ok": True}) as load:
                report = run(root, Harness())
                self.assertEqual(load.call_count, 2)
                self.assertEqual(report["already_done"], 2)
                load.side_effect = AssertionError("now the ledger knows")
                self.assertEqual(run(root, Harness())["already_done"], 2)


class FailureTests(unittest.TestCase):
    def test_one_failing_panorama_does_not_stop_the_run_and_only_it_is_retried_next_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            first = Harness(failing={"1"})
            report = run(root, first, max_attempts=2)
            ledger = RunLedger(ledger_path(root, "set"))
            retried = Harness()
            run(root, retried)
        self.assertEqual(first.perceived.count("1"), 2)  # two attempts
        self.assertEqual(sorted(first.associated), ["0", "2", "3"])
        self.assertEqual((report["failed"], report["state"]), (1, "finished"))
        self.assertEqual(ledger.get("analysis", "1")["status"], "failed")
        self.assertEqual(ledger.total_failures("analysis", "1"), 1)
        self.assertEqual(retried.associated, ["1"])

    def test_transient_errors_are_retried_within_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 2)
            harness = Harness(flaky={"0": 2})
            report = run(root, harness, max_attempts=3)
        self.assertEqual(harness.perceived.count("0"), 3)
        self.assertEqual((report["failed"], sorted(harness.associated)), (0, ["0", "1"]))

    def test_permanent_errors_are_recorded_and_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 3)
            first = Harness(permanent={"1"})
            report = run(root, first)
            second = Harness(permanent={"1"})
            again = run(root, second)
        self.assertEqual((report["skipped"], first.perceived.count("1")), (1, 1))  # no retry for bad input
        self.assertEqual((again["already_resolved"], second.perceived), (1, []))

    def test_run_aborts_after_consecutive_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 6)
            harness = Harness(failing={str(n) for n in range(6)})
            report = run(root, harness, max_attempts=1, max_consecutive_failures=2)
            progress = load_progress(root, "set")
        self.assertEqual(report["state"], "aborted")
        self.assertLess(len(harness.perceived), 6)
        self.assertEqual(progress["state"], "aborted")

    def test_a_success_resets_the_consecutive_failure_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 5)
            harness = Harness(failing={"0", "2", "4"})
            report = run(root, harness, max_attempts=1, max_consecutive_failures=2)
        self.assertEqual((report["state"], report["failed"]), ("finished", 3))

    def test_stop_request_finishes_cleanly_and_the_rerun_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 4)
            stop = threading.Event()
            harness = Harness()
            original = harness.perceive
            harness.perceive = lambda *args: (stop.set() if args[3].stem == "1" else None, original(*args))[1]
            report = run(root, harness, stop=stop)
            interrupted_progress = load_progress(root, "set")
            resumed = Harness()
            run(root, resumed)
        self.assertEqual(report["state"], "interrupted")
        self.assertEqual(interrupted_progress["state"], "interrupted")
        self.assertEqual(sorted(harness.associated + resumed.associated), ["0", "1", "2", "3"])
        self.assertEqual(set(harness.associated) & set(resumed.associated), set())

    def test_perception_only_mode_has_its_own_ledger_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 2)
            run(root, Harness(), perception_only=True)
            analysis = Harness()
            run(root, analysis)
            ledger = RunLedger(ledger_path(root, "set"))
        self.assertEqual(sorted(analysis.associated), ["0", "1"])  # perception is not analysis
        self.assertEqual(ledger.get("perception", "0")["status"], "done")


class LedgerTests(unittest.TestCase):
    def test_torn_last_line_and_garbage_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            ledger = RunLedger(path)
            fingerprint = settings_fingerprint("analysis", {"a": 1})
            ledger.record("analysis", "7", "done", fingerprint, {"size": 3, "mtime_ns": 9})
            with open(path, "a", encoding="utf-8") as output:
                output.write("not json\n{\"image\": \"8\", \"stage\": \"ana")
            reloaded = RunLedger(path)
        self.assertTrue(reloaded.is_done("analysis", "7", fingerprint, {"size": 3, "mtime_ns": 9}))
        self.assertFalse(reloaded.is_done("analysis", "7", fingerprint, {"size": 4, "mtime_ns": 9}))
        self.assertFalse(reloaded.is_done("analysis", "7", settings_fingerprint("analysis", {"a": 2}), {"size": 3, "mtime_ns": 9}))
        self.assertIsNone(reloaded.get("analysis", "8"))


class ProgressTests(unittest.TestCase):
    def test_rate_and_eta_follow_recent_completions(self):
        clock = iter(range(0, 1000, 2))  # two seconds per tick
        with tempfile.TemporaryDirectory() as directory:
            tracker = ProgressTracker(Path(directory) / "p.json", 10, 0, {}, clock=lambda: next(clock))
            for number in range(5):
                tracker.finished_one(str(number), True)
            snapshot = tracker.snapshot()
        self.assertEqual(snapshot["completed"], 5)
        self.assertGreater(snapshot["rate_per_minute"], 0)
        self.assertIsNotNone(snapshot["eta_s"])

    def test_errors_are_kept_and_unknown_rate_gives_no_eta(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = ProgressTracker(Path(directory) / "p.json", 3, 1, {"stage": "analysis"})
            tracker.finished_one("9", False, "boom")
            self.assertIsNone(tracker.eta_seconds())
            self.assertEqual(tracker.snapshot()["recent_errors"][0]["image"], "9")
            tracker.close("finished")
            self.assertEqual(json.loads((Path(directory) / "p.json").read_text())["state"], "finished")

    def test_format_duration(self):
        self.assertEqual(format_duration(None), "unknown")
        self.assertEqual(format_duration(42), "42 s")
        self.assertEqual(format_duration(125), "2 min 05 s")
        self.assertEqual(format_duration(7500), "2 h 05 min")
        self.assertEqual(format_duration(200000), "2 d 7 h")


class PageTests(unittest.TestCase):
    def test_page_shows_counts_errors_escaped_and_an_embedded_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = _dataset(root, 4)
            harness = Harness(failing={"1"}, permanent={"2"})
            run(root, harness, max_attempts=1)
            page = progress_html(root, "set", refresh_seconds=10)
            statuses = image_statuses(root, "set", images)
        self.assertEqual(sorted(statuses.values()), ["done", "done", "failed", "skipped"])
        self.assertIn('content="10"', page)
        self.assertIn("data:image/png;base64,", page)
        self.assertIn("Failed (will retry)", page)
        self.assertIn("RuntimeError: gpu fell over", page)
        self.assertNotIn("<script", page)

    def test_error_text_cannot_inject_markup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 1)
            harness = Harness()
            harness.perceive = lambda *args: (_ for _ in ()).throw(RuntimeError("<img src=x onerror=alert(1)>"))
            run(root, harness, max_attempts=1)
            page = progress_html(root, "set")
        self.assertNotIn("<img src=x", page)
        self.assertIn("&lt;img src=x", page)

    def test_map_png_has_the_requested_size_and_handles_no_positions(self):
        from io import BytesIO
        from PIL import Image

        images = [{"id": "1", "computed_geometry": {"coordinates": [4.9, 52.3]}}, {"id": "2", "computed_geometry": {"coordinates": [4.91, 52.31]}}]
        with Image.open(BytesIO(render_map(images, {"1": "done"}, 300, 200))) as picture:
            self.assertEqual(picture.size, (300, 200))
        with Image.open(BytesIO(render_map([{"id": "3"}], {}, 100, 80))) as picture:
            self.assertEqual(picture.size, (100, 80))


class LoadAnalysisTests(unittest.TestCase):
    def test_a_known_record_avoids_rereading_the_manifest(self):
        from city_analyser import building_analysis

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "set" / "analysis"
            analysis.mkdir(parents=True)
            (analysis / "1.json").write_text("{}", encoding="utf-8")
            (analysis / "1-ids.png").write_bytes(b"x")
            (analysis / "1-facades.png").write_bytes(b"x")
            with patch.object(building_analysis, "_dataset_image", side_effect=AssertionError("manifest parsed")):
                self.assertIsNone(building_analysis.load_analysis(root, "set", "1", {"id": "1"}))  # invalid, but without a manifest parse


if __name__ == "__main__":
    unittest.main()
