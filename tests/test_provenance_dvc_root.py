from training import provenance


def test_dvc_status_is_skipped_inside_dvc_stage(monkeypatch):
    monkeypatch.setenv("DVC_ROOT", "/some/repo")
    monkeypatch.setattr(provenance, "_dvc_status_impl", lambda: (_ for _ in ()).throw(AssertionError("must not run")))
    assert provenance.dvc_status() == "unknown"


def test_dvc_status_delegates_outside_dvc(monkeypatch):
    monkeypatch.delenv("DVC_ROOT", raising=False)
    monkeypatch.setattr(provenance, "_dvc_status_impl", lambda: "up-to-date")
    assert provenance.dvc_status() == "up-to-date"