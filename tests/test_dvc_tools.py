import subprocess

import pytest

from agent.tools import dvc_tools as dt
from agent.tools import mlflow_tools as mt
from agent.tools.base import ToolInputError


def _git(path, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=path, check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init")
    (tmp_path / "data").mkdir()
    dvc_file = tmp_path / "data" / "d.csv.dvc"
    dvc_file.write_text("outs:\n- md5: aaa111bbb222\n  size: 100\n  path: d.csv\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "v1")
    _git(tmp_path, "tag", "data-v1")
    dvc_file.write_text("outs:\n- md5: ccc333ddd444\n  size: 150\n  path: d.csv\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "v2")
    _git(tmp_path, "tag", "data-v2")
    monkeypatch.setattr(dt, "ROOT", tmp_path)
    return tmp_path


def test_list_and_current_version(repo):
    listed = dt.list_tracked_datasets()["datasets"]
    assert listed[0]["dataset"] == "data/d.csv" and listed[0]["dvc_md5"] == "ccc333ddd444"
    ver = dt.get_dataset_version("data/d.csv")
    assert ver["version"] == "v2" and ver["dvc_md5"] == "ccc333ddd444"
    assert len(ver["dataset_git_commit"]) == 40


def test_compare_versions_uses_git_and_dvc_diff(repo, monkeypatch):
    monkeypatch.setattr(dt, "_dvc", lambda *a, **k: (0, '{"modified": [{"path": "d.csv"}]}', ""))
    res = dt.compare_dataset_versions("data-v1", "data-v2", "data/d.csv")
    assert res["identical"] is False and res["size_delta_bytes"] == 50
    assert res["dvc_diff"]["modified"][0]["path"] == "d.csv"


def test_explain_reproduce_is_read_only_and_names_commands(repo):
    res = dt.explain_reproduce_version("data-v1", "data/d.csv")
    assert res["executed"] is False and res["expected_md5"] == "aaa111bbb222"
    assert res["steps"][0] == "git checkout data-v1 -- data/d.csv.dvc"


def test_check_data_changes_reads_dvc_status(repo, monkeypatch):
    monkeypatch.setattr(dt, "_dvc", lambda *a, **k: (0, "{}", ""))
    assert dt.check_data_changes("data/d.csv")["data_changed"] is False
    status = '{"data/d.csv.dvc": [{"changed outs": {"data/d.csv": "modified"}}]}'
    monkeypatch.setattr(dt, "_dvc", lambda *a, **k: (0, status, ""))
    assert dt.check_data_changes("data/d.csv")["data_changed"] is True


@pytest.mark.parametrize("ref", ["--output=x", "-h", "a..b", "no-such-tag", "x;rm", ""])
def test_bad_refs_rejected(repo, ref):
    with pytest.raises(ToolInputError):
        dt.explain_reproduce_version(ref, "data/d.csv")


@pytest.mark.parametrize("path", ["../secret.csv", "/etc/passwd", "data/missing.csv", "a b"])
def test_bad_dataset_paths_rejected(repo, path):
    with pytest.raises(ToolInputError):
        dt.get_dataset_version(path)


def test_command_allowlist_blocks_state_changing_commands(repo):
    for sub in ("push", "checkout", "reset", "clean"):
        with pytest.raises(ToolInputError):
            dt._git(sub)
    for sub in ("push", "pull", "checkout", "remove"):
        with pytest.raises(ToolInputError):
            dt._dvc(sub)


def test_run_dataset_version_verified_against_git(repo, monkeypatch):
    v1_commit = subprocess.run(["git", "rev-parse", "data-v1"], cwd=repo, capture_output=True, text=True).stdout.strip()
    from types import SimpleNamespace as NS

    def fake_run(md5):
        tags = {
            "mlflow.runName": "exp25",
            "dataset.version": "v1",
            "dataset.dvc_md5": md5,
            "dataset.git_commit": v1_commit,
            "dataset.dvc_file": "data/d.csv.dvc",
        }
        return NS(info=NS(run_id="e" * 32), data=NS(tags=tags))

    monkeypatch.setattr(mt, "_client", lambda: object())
    monkeypatch.setattr(mt, "resolve_run", lambda c, ref: fake_run("aaa111bbb222"))
    ok = dt.get_run_dataset_version("exp25")
    assert ok["dataset_version"] == "v1" and ok["verified_against_git"] is True
    monkeypatch.setattr(mt, "resolve_run", lambda c, ref: fake_run("tampered"))
    assert dt.get_run_dataset_version("exp25")["verified_against_git"] is False


def test_run_without_provenance_is_reported_not_invented(repo, monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.setattr(mt, "_client", lambda: object())
    monkeypatch.setattr(mt, "resolve_run", lambda c, ref: NS(info=NS(run_id="f" * 32), data=NS(tags={})))
    res = dt.get_run_dataset_version("x")
    assert res["provenance_recorded"] is False and res["dataset_version"] is None
