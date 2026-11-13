import pytest
from kubernetes.client.exceptions import ApiException

from agent.tools import kubernetes_tools as k8s
from agent.tools.base import ToolUnavailableError, execute

TOOL = {t.name: t for t in k8s.TOOLS}
APP = {"app": "breast-cancer-api"}


def run(name, **kw):
    return execute(name, TOOL[name].fn, request_id="t1", **kw)


def pod(name="breast-cancer-api-abc-1", phase="Running", ready=True, restarts=0, state=None, last=None,
        labels=None, image="breast-cancer-api:1.0.0", conditions=None):
    cs = {"name": "api", "ready": ready, "restartCount": restarts, "image": image, "imageID": "docker://sha256:abc",
          "state": state or {"running": {"startedAt": "2026-11-13T10:00:00Z"}}}
    if last:
        cs["lastState"] = {"terminated": last}
    return {
        "metadata": {"name": name, "labels": labels or dict(APP), "creationTimestamp": "2026-11-13T10:00:00Z"},
        "spec": {"nodeName": "docker-desktop", "containers": [{
            "name": "api", "image": image,
            "readinessProbe": {"httpGet": {"path": "/readyz", "port": "http"}},
            "livenessProbe": {"httpGet": {"path": "/healthz", "port": "http"}},
            "resources": {"limits": {"memory": "256Mi"}}}]},
        "status": {"phase": phase, "containerStatuses": [cs], "conditions": conditions or []},
    }


def deployment(desired=2, ready=2, image="breast-cancer-api:1.0.0", label="1.0.0", conditions=None):
    return {
        "metadata": {"name": "breast-cancer-api", "generation": 1, "labels": {"app.kubernetes.io/version": label}},
        "spec": {"replicas": desired, "strategy": {"type": "RollingUpdate"},
                 "template": {"spec": {"containers": [{"image": image}]}}},
        "status": {"readyReplicas": ready, "availableReplicas": ready, "updatedReplicas": desired,
                   "observedGeneration": 1, "conditions": conditions or []},
    }


@pytest.fixture
def cluster(monkeypatch):
    c = {"pods": [pod("breast-cancer-api-abc-1"), pod("breast-cancer-api-abc-2")], "deployments": [deployment()],
         "services": [], "slices": [], "events": [], "log_text": "INFO model loaded\n", "logs": [],
         "configmap": {"data": {"APP_VERSION": "1.0.0"}}, "patches": []}

    def read_pod(name):
        for p in c["pods"]:
            if p["metadata"]["name"] == name:
                return p
        raise k8s.ToolInputError("resource not found in namespace mlops-agent")

    def read_logs(name, tail, previous):
        c["logs"].append((name, tail, previous))
        return c["log_text"]

    monkeypatch.setattr(k8s, "_list_pods", lambda: [p for p in c["pods"] if p["metadata"]["labels"].get("app") == "breast-cancer-api"])
    monkeypatch.setattr(k8s, "_read_pod", read_pod)
    monkeypatch.setattr(k8s, "_read_logs", read_logs)
    monkeypatch.setattr(k8s, "_list_events", lambda: c["events"])
    monkeypatch.setattr(k8s, "_list_services", lambda: c["services"])
    monkeypatch.setattr(k8s, "_list_endpoint_slices", lambda svc: c["slices"])
    monkeypatch.setattr(k8s, "_list_deployments", lambda: c["deployments"])
    monkeypatch.setattr(k8s, "_read_configmap", lambda: c["configmap"])
    monkeypatch.setattr(k8s, "_patch_deployment", lambda body: c["patches"].append(body) or {})
    return c


def event(reason, msg, obj="breast-cancer-api-abc-1", typ="Warning", ts="2026-11-13T10:05:00Z"):
    return {"type": typ, "reason": reason, "message": msg, "lastTimestamp": ts, "count": 3,
            "involvedObject": {"kind": "Pod", "name": obj}}


def test_list_pods_and_deployments(cluster):
    pods = run("k8s_list_pods")
    assert pods.status == "ok" and pods.data["count"] == 2
    assert pods.data["pods"][0]["ready"] == "1/1" and pods.data["pods"][0]["phase"] == "Running"
    dep = run("k8s_list_deployments").data["deployments"][0]
    assert dep["desired"] == 2 and dep["ready"] == 2 and dep["rollout_complete"] is True


def test_healthy_cluster_is_all_clear(cluster):
    d = run("k8s_find_unhealthy_pods").data
    assert d["all_clear"] is True and d["unhealthy_pods"] == [] and d["healthy_pods"] == 2


def test_detects_crashloop_oomkilled_and_restarts(cluster):
    cluster["pods"][1] = pod("breast-cancer-api-abc-2", ready=False, restarts=4,
                             state={"waiting": {"reason": "CrashLoopBackOff"}},
                             last={"reason": "OOMKilled", "exitCode": 137})
    cluster["events"] = [event("BackOff", "Back-off restarting failed container", "breast-cancer-api-abc-2")]
    d = run("k8s_find_unhealthy_pods").data
    assert d["all_clear"] is False and len(d["unhealthy_pods"]) == 1
    text = " ".join(d["unhealthy_pods"][0]["problems"])
    assert "CrashLoopBackOff" in text and "OOMKilled" in text and "restarted 4" in text
    assert d["warning_events"][0]["reason"] == "BackOff"


def test_detects_running_but_not_ready_and_deployment_shortfall(cluster):
    cluster["pods"] = [pod(), pod("breast-cancer-api-abc-2", ready=False)]
    cluster["deployments"] = [deployment(desired=2, ready=1, conditions=[
        {"type": "Available", "status": "False", "reason": "MinimumReplicasUnavailable", "message": "x"}])]
    d = run("k8s_find_unhealthy_pods").data
    assert "not ready (0/1)" in d["unhealthy_pods"][0]["problems"]
    assert any("1/2 replicas ready" in i for i in d["deployment_issues"])
    assert any("MinimumReplicasUnavailable" in i for i in d["deployment_issues"])


def test_no_deployment_or_pods_is_reported_not_assumed_healthy(cluster):
    cluster["pods"], cluster["deployments"] = [], []
    d = run("k8s_find_unhealthy_pods").data
    assert d["all_clear"] is False and "no deployment" in d["deployment_issues"][0]


def test_pod_status_includes_probes_and_pod_events_only(cluster):
    cluster["events"] = [event("Unhealthy", "Readiness probe failed: HTTP 503"),
                         event("Unhealthy", "other pod", "breast-cancer-api-abc-2")]
    d = run("k8s_get_pod_status", pod_name="breast-cancer-api-abc-1").data
    assert d["probes"]["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert len(d["warning_events"]) == 1 and "503" in d["warning_events"][0]["message"]


def test_pod_name_validation_and_scope(cluster):
    cluster["pods"].append(pod("other-pod", labels={"app": "something-else"}))
    for bad in ("../etc/passwd", "UPPER", "a; rm -rf /", "", None, 5):
        assert run("k8s_get_pod_status", pod_name=bad).status == "rejected"
    r = run("k8s_get_pod_logs", pod_name="other-pod")
    assert r.status == "rejected" and "approved application" in r.error
    assert cluster["logs"] == []  # logs were never fetched for the out-of-scope pod


def test_logs_tail_validation_and_truncation(cluster):
    cluster["log_text"] = "x" * 5000
    ok = run("k8s_get_pod_logs", pod_name="breast-cancer-api-abc-1", tail_lines=500)
    assert ok.status == "rejected"
    r = run("k8s_get_pod_logs", pod_name="breast-cancer-api-abc-1", tail_lines=20, previous=True)
    assert r.status == "ok" and r.data["truncated"] is True and len(r.data["log_tail"]) == k8s.LOG_TAIL_CHARS
    assert cluster["logs"][-1] == ("breast-cancer-api-abc-1", 20, True)


def test_logs_are_redacted_by_wrapper(cluster):
    cluster["log_text"] = "connecting with password=hunter2-super-secret\n"
    r = run("k8s_get_pod_logs", pod_name="breast-cancer-api-abc-1")
    assert "hunter2-super-secret" not in str(r.data)


def test_services_report_ready_endpoints(cluster):
    cluster["services"] = [{"metadata": {"name": "breast-cancer-api"},
                            "spec": {"type": "ClusterIP", "clusterIP": "10.0.0.1", "selector": APP,
                                     "ports": [{"name": "http", "port": 80, "targetPort": "http", "protocol": "TCP"}]}}]
    cluster["slices"] = [{"endpoints": [{"conditions": {"ready": True}}, {"conditions": {"ready": False}}]}]
    s = run("k8s_list_services").data["services"][0]
    assert s["type"] == "ClusterIP" and s["ready_endpoints"] == 1 and s["total_endpoints"] == 2


def test_deployed_version_consistent_and_mismatch(cluster):
    d = run("k8s_get_deployed_version").data
    assert d["declared_versions"] == {"deployment_label": "1.0.0", "image_tag": "1.0.0", "configmap_app_version": "1.0.0"}
    assert d["mismatches"] == [] and "not queried" in d["note"]
    cluster["pods"][0] = pod(image="breast-cancer-api:0.9.0")
    cluster["configmap"] = {"data": {"APP_VERSION": "2.0.0"}}
    m = run("k8s_get_deployed_version").data["mismatches"]
    assert any("ConfigMap" in x for x in m) and any("abc-1" in x for x in m)


def test_restart_rejected_without_confirmation_or_outside_scope(cluster):
    assert run("k8s_rollout_restart").status == "rejected"
    assert run("k8s_rollout_restart", confirmed="yes").status == "rejected"
    r = run("k8s_rollout_restart", namespace="kube-system", confirmed=True)
    assert r.status == "rejected" and "mlops-agent" in r.error
    assert run("k8s_rollout_restart", deployment="coredns", confirmed=True).status == "rejected"
    assert cluster["patches"] == []  # nothing was changed


def test_restart_confirmed_patches_only_the_annotation(cluster):
    r = run("k8s_rollout_restart", confirmed=True)
    assert r.status == "ok" and len(cluster["patches"]) == 1
    ann = cluster["patches"][0]["spec"]["template"]["metadata"]["annotations"]
    assert "kubectl.kubernetes.io/restartedAt" in ann


def test_only_restart_is_mutating_and_reads_take_no_namespace():
    assert [t.name for t in k8s.TOOLS if t.mutating] == ["k8s_rollout_restart"]
    assert all("namespace" not in t.params for t in k8s.TOOLS if not t.mutating)
    assert all(t.name.startswith("k8s_") for t in k8s.TOOLS)


def test_api_errors_are_classified():
    def boom(status):
        def f(*a, **kw):
            raise ApiException(status=status, reason="x")
        return f
    assert execute("t", lambda: k8s._call(boom(403))).status == "unavailable"
    assert execute("t", lambda: k8s._call(boom(503))).status == "unavailable"
    assert execute("t", lambda: k8s._call(boom(404))).status == "rejected"


def test_unreachable_cluster_returns_unavailable_without_data(monkeypatch):
    monkeypatch.setattr(k8s, "_clients", {})
    def no_config(**kw):
        raise OSError("no kubeconfig")
    monkeypatch.setattr(k8s.config, "load_kube_config", no_config)
    r = run("k8s_list_pods")
    assert r.status == "unavailable" and r.data is None


def test_connection_failure_is_unavailable(monkeypatch):
    def down():
        raise RuntimeError("HTTPConnectionPool: Max retries exceeded with url")
    monkeypatch.setattr(k8s, "_list_pods", down)
    r = run("k8s_list_pods")
    assert r.status == "unavailable" and r.data is None