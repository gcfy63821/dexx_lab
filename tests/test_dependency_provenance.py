"""Verify pinned source provenance without installing or importing dependencies."""
import importlib.util
from importlib.metadata import PackageNotFoundError
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class RaycasterProvenanceTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "provenance_checker", ROOT / "tutorial/00_setup/check_versions.py")
        self.checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.checker)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.url = "https://example.invalid/raycaster.git"
        self.sha = "1" * 40
        (self.root / "requirements.txt").write_text(
            f"simple-raycaster @ git+{self.url}@{self.sha}\n")

    def verify(self, metadata):
        dist = SimpleNamespace(read_text=lambda name: metadata)
        with mock.patch.object(self.checker, "distribution", return_value=dist):
            self.checker.check_raycaster_revision(self.root)

    def test_expected_revision_from_requirements_passes(self):
        self.verify(json.dumps({"url": self.url,
                                "vcs_info": {"vcs": "git", "commit_id": self.sha}}))

    def test_uv_url_without_git_suffix_passes(self):
        self.verify(json.dumps({"url": self.url.removesuffix(".git"),
                                "vcs_info": {"vcs": "git", "commit_id": self.sha}}))

    def test_wrong_commit_source_or_missing_provenance_fails(self):
        examples = [None, "{invalid", "null", "[]", "{}",
                    json.dumps({"url": self.url, "dir_info": {"editable": True}}),
                    json.dumps({"url": self.url, "vcs_info": {"vcs": "git", "commit_id": "2" * 40}}),
                    json.dumps({"url": "https://example.invalid/fork.git",
                                "vcs_info": {"vcs": "git", "commit_id": self.sha}}),
                    json.dumps({"url": "https://example.invalid/raycaster-fork",
                                "vcs_info": {"vcs": "git", "commit_id": self.sha}}),
                    json.dumps({"url": self.url, "vcs_info": {"vcs": "git", "requested_revision": self.sha}})]
        for metadata in examples:
            with self.subTest(metadata=metadata), self.assertRaisesRegex(RuntimeError, "--force-reinstall"):
                self.verify(metadata)

    def test_missing_distribution_fails(self):
        with mock.patch.object(self.checker, "distribution", side_effect=PackageNotFoundError("simple-raycaster")):
            with self.assertRaises(PackageNotFoundError):
                self.checker.check_raycaster_revision(self.root)


if __name__ == "__main__":
    unittest.main()
