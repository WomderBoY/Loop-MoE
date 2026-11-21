import fcntl
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_expanded

spec = importlib.util.spec_from_file_location(
    "cpu_resume", Path(__file__).resolve().parents[1] / "scripts/resume_postprocess.py")
cpu_resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu_resume)


class CpuResumeTests(unittest.TestCase):
    def test_check_is_read_only_and_never_invokes_gpu_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_expanded.ExpandedTests().make_run(root)
            before = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
            with patch("subprocess.run", side_effect=AssertionError("No GPU or subprocess check expected")):
                config, prompts, shards = cpu_resume.validate_inputs(root / "config.json", root / "out")
            after = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(config["models"], ["test"])
            self.assertEqual(len(shards), 2)

    def test_incomplete_collection_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_expanded.ExpandedTests().make_run(root)
            (root / "out/shards/001/test/trace.jsonl.gz").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Collection is incomplete"):
                cpu_resume.validate_inputs(root / "config.json", root / "out")

    def test_cpu_resume_cannot_overlap_the_active_driver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_expanded.ExpandedTests().make_run(root)
            with (root / "out/.run.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(cpu_resume, "finish") as finish:
                    with self.assertRaises(BlockingIOError):
                        cpu_resume.run(SimpleNamespace(config=root / "config.json", out_dir=root / "out",
                                                       check_only=False))
                    finish.assert_not_called()
