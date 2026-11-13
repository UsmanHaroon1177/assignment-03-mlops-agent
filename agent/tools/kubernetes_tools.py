"""Kubernetes operations tools (official `kubernetes` client).

Scope is fixed: namespace `mlops-agent`, label app=breast-cancer-api, deployment `breast-cancer-api`.
Read tools accept no namespace. The only mutating tool (k8s_rollout_restart) rejects any other
namespace/deployment and refuses to run unless `confirmed=True`, which the orchestrator (not the
model) sets after the user explicitly approves.
Every cluster call goes through a small `_raw` function returning plain dicts, so tests mock those.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from agent.tools.base import ToolInputError, ToolSpec, ToolUnavailableError

NAMESPACE = "mlops-agent"
APP_LABEL = "breast-cancer-api"
DEPLOYMENT = "breast-cancer-api"
CONFIGMAP = "breast-cancer-api-config"
TIMEOUT = 10
LOG_TAIL_CHARS = 3000
MAX_TAIL_LINES = 200
_POD_RE = re.compile(r"^[a-z0-9]([a-z0-9.\-]{0,251}[a-z0-9])?$")
_clients: dict[str, Any] = {}


# ---------------------------------------------------------------- cluster plumbing
def _load() -> dict[str, Any]:
    if not _clients:
        try:
            config.load_kube_config(context=os.getenv("KUBE_CONTEXT") or None)
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailableError(f"Kubernetes config could not be loaded ({type(exc).__name__})") from exc
        _clients.update(core=client.CoreV1Api(), apps=client.AppsV1Api(),
                        disc=client.DiscoveryV1Api(), api=client.ApiClient())
    return _clients


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, _request_timeout=TIMEOUT, **kwargs)
    except ApiException as exc:
        if exc.status == 404:
            raise ToolInputError(f"resource not found in namespace {NAMESPACE}") from exc
        if exc.status in (400, 409, 422):
            raise ToolInputError(f"Kubernetes rejected the request (HTTP {exc.status})") from exc
        raise ToolUnavailableError(f"Kubernetes API error (HTTP {exc.status}: {exc.reason})") from exc


def _dict(obj: Any) -> dict[str, Any]:
    return _load()["api"].sanitize_for_serialization(obj)


def _selector() -> str:
    return f"app={APP_LABEL}"


def _list_pods() -> list[dict]:
    return _dict(_call(_load()["core"].list_namespaced_pod, NAMESPACE, label_selector=_selector())).get("items", [])


def _read_pod(name: str) -> dict:
    return _dict(_call(_load()["core"].read_namespaced_pod, name, NAMESPACE))


def _read_logs(name: str, tail: int, previous: bool) -> str:
    return _call(_load()["core"].read_namespaced_pod_log, name, NAMESPACE,
                 container="api", tail_lines=tail, previous=previous) or ""


def _list_events() -> list[dict]:
    return _dict(_call(_load()["core"].list_namespaced_event, NAMESPACE)).get("items", [])


def _list_services() -> list[dict]:
    return _dict(_call(_load()["core"].list_namespaced_service, NAMESPACE, label_selector=_selector())).get("items", [])


def _list_endpoint_slices(service: str) -> list[dict]:
    return _dict(_call(_load()["disc"].list_namespaced_endpoint_slice, NAMESPACE,
                       label_selector=f"kubernetes.io/service-name={service}")).get("items", [])


def _list_deployments() -> list[dict]:
    return _dict(_call(_load()["apps"].list_namespaced_deployment, NAMESPACE, label_selector=_selector())).get("items", [])


def _read_configmap() -> dict:
    return _dict(_call(_load()["core"].read_namespaced_config_map, CONFIGMAP, NAMESPACE))


def _patch_deployment(body: dict) -> dict:
    return _dict(_call(_load()["apps"].patch_namespaced_deployment, DEPLOYMENT, NAMESPACE, body))


# ---------------------------------------------------------------- helpers
def _age(ts: str | None) -> int | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))


def _tag(image: str | None) -> str | None:
    if not image:
        return None
    last = image.rsplit("/", 1)[-1]
    if "@" in last:
        return None
    return last.rsplit(":", 1)[1] if ":" in last else "latest"


def _container(cs: dict) -> dict:
    state = cs.get("state") or {}
    kind = next(iter(state), None)
    detail = state.get(kind) or {} if kind else {}
    last = (cs.get("lastState") or {}).get("terminated")
    return {
        "name": cs.get("name"), "ready": bool(cs.get("ready")), "restart_count": cs.get("restartCount", 0),
        "state": kind or "unknown", "reason": detail.get("reason"), "exit_code": detail.get("exitCode"),
        "message": (detail.get("message") or "")[:200] or None,
        "last_terminated": {"reason": last.get("reason"), "exit_code": last.get("exitCode"),
                            "finished_at": last.get("finishedAt")} if last else None,
        "image": cs.get("image"), "image_id": cs.get("imageID"),
    }


def _pod_summary(pod: dict) -> dict:
    meta, spec, status = pod.get("metadata") or {}, pod.get("spec") or {}, pod.get("status") or {}
    cons = [_container(c) for c in status.get("containerStatuses") or []]
    total = len(spec.get("containers") or cons)
    specs = spec.get("containers") or [{}]
    return {
        "name": meta.get("name"), "phase": status.get("phase"),
        "ready": f"{sum(1 for c in cons if c['ready'])}/{total}",
        "restarts": sum(c["restart_count"] for c in cons), "node": spec.get("nodeName"),
        "pod_ip": status.get("podIP"), "age_seconds": _age(meta.get("creationTimestamp")),
        "image": specs[0].get("image"), "deleting": bool(meta.get("deletionTimestamp")), "containers": cons,
    }


def _pod_problems(summary: dict, raw: dict) -> list[str]:
    out: list[str] = []
    if summary["deleting"]:
        out.append("pod is terminating")
    if summary["phase"] != "Running":
        out.append(f"phase is {summary['phase']}")
    for cond in (raw.get("status") or {}).get("conditions") or []:
        if cond.get("type") == "PodScheduled" and cond.get("status") == "False":
            out.append(f"not scheduled: {cond.get('reason')} {(cond.get('message') or '')[:150]}".strip())
    for c in summary["containers"]:
        if c["state"] == "waiting" and c["reason"]:
            out.append(f"container {c['name']} waiting: {c['reason']}")
        if c["state"] == "terminated":
            out.append(f"container {c['name']} terminated: {c['reason']} (exit {c['exit_code']})")
        if c["last_terminated"]:
            lt = c["last_terminated"]
            out.append(f"container {c['name']} last terminated: {lt['reason']} (exit {lt['exit_code']})")
        if c["restart_count"]:
            out.append(f"container {c['name']} restarted {c['restart_count']} time(s)")
    ready, total = summary["ready"].split("/")
    if summary["phase"] == "Running" and not summary["deleting"] and ready != total:
        out.append(f"not ready ({summary['ready']})")
    return out


def _deployment_summary(d: dict) -> dict:
    meta, spec, st = d.get("metadata") or {}, d.get("spec") or {}, d.get("status") or {}
    desired = spec.get("replicas", 1)
    ready, available, updated = st.get("readyReplicas", 0), st.get("availableReplicas", 0), st.get("updatedReplicas", 0)
    containers = ((spec.get("template") or {}).get("spec") or {}).get("containers") or [{}]
    complete = (updated == desired and ready == desired and available == desired
                and (st.get("observedGeneration") or 0) >= (meta.get("generation") or 0))
    return {
        "name": meta.get("name"), "desired": desired, "ready": ready, "available": available, "updated": updated,
        "image": containers[0].get("image"), "version_label": (meta.get("labels") or {}).get("app.kubernetes.io/version"),
        "strategy": (spec.get("strategy") or {}).get("type"), "rollout_complete": complete,
        "conditions": [{"type": c.get("type"), "status": c.get("status"), "reason": c.get("reason"),
                        "message": (c.get("message") or "")[:200]} for c in st.get("conditions") or []],
    }


def _deployment_issues(s: dict) -> list[str]:
    out = []
    if s["ready"] < s["desired"]:
        out.append(f"{s['name']}: only {s['ready']}/{s['desired']} replicas ready")
    for c in s["conditions"]:
        bad = (c["type"] in ("Available", "Progressing") and c["status"] == "False") or \
              (c["type"] == "ReplicaFailure" and c["status"] == "True")
        if bad:
            out.append(f"{s['name']}: condition {c['type']}={c['status']} ({c['reason']}) {c['message']}".strip())
    return out


def _event_summary(e: dict) -> dict:
    obj = e.get("involvedObject") or {}
    return {
        "type": e.get("type"), "reason": e.get("reason"), "object": f"{obj.get('kind')}/{obj.get('name')}",
        "message": (e.get("message") or "")[:300], "count": e.get("count") or 1,
        "last_seen": e.get("lastTimestamp") or e.get("eventTime") or (e.get("metadata") or {}).get("creationTimestamp"),
    }


def _events(only_warnings: bool, limit: int, object_name: str | None = None) -> list[dict]:
    rows = []
    for e in _list_events():
        name = (e.get("involvedObject") or {}).get("name") or ""
        if object_name is not None and name != object_name:
            continue
        if object_name is None and not name.startswith(APP_LABEL):
            continue
        if only_warnings and e.get("type") != "Warning":
            continue
        rows.append(_event_summary(e))
    rows.sort(key=lambda r: r["last_seen"] or "", reverse=True)
    return rows[:limit]


def _check_pod_name(pod_name: Any) -> str:
    if not isinstance(pod_name, str) or not _POD_RE.match(pod_name):
        raise ToolInputError("pod_name must be a valid Kubernetes pod name (lowercase letters, digits, '-', '.')")
    return pod_name


def _scoped_pod(pod_name: str) -> dict:
    pod = _read_pod(_check_pod_name(pod_name))
    if ((pod.get("metadata") or {}).get("labels") or {}).get("app") != APP_LABEL:
        raise ToolInputError(f"pod is not part of the approved application ({APP_LABEL})")
    return pod


def _int_arg(value: Any, default: int, low: int, high: int, name: str) -> int:
    value = default if value in (None, "") else value
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ToolInputError(f"{name} must be an integer between {low} and {high}")
    return value


def _bool_arg(value: Any, default: bool, name: str) -> bool:
    value = default if value is None else value
    if not isinstance(value, bool):
        raise ToolInputError(f"{name} must be true or false")
    return value


# ---------------------------------------------------------------- tools
def list_deployments() -> dict[str, Any]:
    deps = [_deployment_summary(d) for d in _list_deployments()]
    return {"namespace": NAMESPACE, "deployments": deps, "count": len(deps)}


def list_pods() -> dict[str, Any]:
    pods = [_pod_summary(p) for p in _list_pods()]
    return {"namespace": NAMESPACE, "pods": pods, "count": len(pods)}


def get_pod_status(pod_name: Any) -> dict[str, Any]:
    pod = _scoped_pod(pod_name)
    summary = _pod_summary(pod)
    c0 = ((pod.get("spec") or {}).get("containers") or [{}])[0]
    return {
        "pod": summary, "problems": _pod_problems(summary, pod),
        "conditions": [{"type": c.get("type"), "status": c.get("status"), "reason": c.get("reason"),
                        "message": (c.get("message") or "")[:200]} for c in (pod.get("status") or {}).get("conditions") or []],
        "probes": {k: c0[k] for k in ("readinessProbe", "livenessProbe") if c0.get(k)},
        "resources": c0.get("resources"),
        "warning_events": _events(True, 10, object_name=pod_name),
    }


def get_pod_logs(pod_name: Any, tail_lines: Any = 50, previous: Any = False) -> dict[str, Any]:
    tail = _int_arg(tail_lines, 50, 1, MAX_TAIL_LINES, "tail_lines")
    prev = _bool_arg(previous, False, "previous")
    _scoped_pod(pod_name)
    text = _read_logs(pod_name, tail, prev)
    return {"pod": pod_name, "previous_container": prev, "tail_lines": tail,
            "truncated": len(text) > LOG_TAIL_CHARS, "log_tail": text[-LOG_TAIL_CHARS:]}


def list_services() -> dict[str, Any]:
    out = []
    for s in _list_services():
        meta, spec = s.get("metadata") or {}, s.get("spec") or {}
        endpoints = [e for sl in _list_endpoint_slices(meta.get("name")) for e in sl.get("endpoints") or []]
        ready = [e for e in endpoints if (e.get("conditions") or {}).get("ready", True) is not False]
        out.append({
            "name": meta.get("name"), "type": spec.get("type"), "cluster_ip": spec.get("clusterIP"),
            "selector": spec.get("selector"),
            "ports": [{"name": p.get("name"), "port": p.get("port"), "target_port": p.get("targetPort"),
                       "protocol": p.get("protocol")} for p in spec.get("ports") or []],
            "ready_endpoints": len(ready), "total_endpoints": len(endpoints),
        })
    return {"namespace": NAMESPACE, "services": out, "count": len(out)}


def find_unhealthy_pods() -> dict[str, Any]:
    raw = _list_pods()
    deps = [_deployment_summary(d) for d in _list_deployments()]
    unhealthy, healthy = [], 0
    for p in raw:
        s = _pod_summary(p)
        problems = _pod_problems(s, p)
        if problems:
            unhealthy.append({"name": s["name"], "phase": s["phase"], "ready": s["ready"],
                              "restarts": s["restarts"], "problems": problems})
        else:
            healthy += 1
    issues = [i for d in deps for i in _deployment_issues(d)]
    if not deps:
        issues.append(f"no deployment with label app={APP_LABEL} found in namespace {NAMESPACE}")
    elif not raw:
        issues.append("deployment exists but has no pods")
    return {"namespace": NAMESPACE, "pods_total": len(raw), "healthy_pods": healthy, "unhealthy_pods": unhealthy,
            "deployment_issues": issues, "warning_events": _events(True, 10),
            "all_clear": not unhealthy and not issues}


def get_events(only_warnings: Any = True, limit: Any = 20) -> dict[str, Any]:
    warn = _bool_arg(only_warnings, True, "only_warnings")
    n = _int_arg(limit, 20, 1, 50, "limit")
    return {"namespace": NAMESPACE, "only_warnings": warn, "events": _events(warn, n)}


def get_deployed_version() -> dict[str, Any]:
    deps = [_deployment_summary(d) for d in _list_deployments()]
    if not deps:
        return {"found": False, "note": f"no deployment with label app={APP_LABEL} in namespace {NAMESPACE}"}
    dep = deps[0]
    pods = [_pod_summary(p) for p in _list_pods()]
    try:
        cm_version = ((_read_configmap().get("data")) or {}).get("APP_VERSION")
    except ToolInputError:
        cm_version = None
    tag = _tag(dep["image"])
    mismatches = []
    if dep["version_label"] and tag and dep["version_label"] != tag:
        mismatches.append(f"deployment label {dep['version_label']} differs from image tag {tag}")
    if cm_version and tag and cm_version != tag:
        mismatches.append(f"ConfigMap APP_VERSION {cm_version} differs from image tag {tag}")
    mismatches += [f"pod {p['name']} runs image {p['image']} (deployment wants {dep['image']})"
                   for p in pods if p["image"] != dep["image"]]
    return {
        "found": True, "deployment": dep["name"], "image": dep["image"],
        "declared_versions": {"deployment_label": dep["version_label"], "image_tag": tag,
                              "configmap_app_version": cm_version},
        "rollout_complete": dep["rollout_complete"],
        "pods": [{"name": p["name"], "image": p["image"], "ready": p["ready"],
                  "image_id": (p["containers"][0]["image_id"] if p["containers"] else None)} for p in pods],
        "mismatches": mismatches,
        "note": "Versions are as declared in the Deployment/ConfigMap; the running app's /version endpoint was not queried.",
    }


def rollout_restart(namespace: Any = NAMESPACE, deployment: Any = DEPLOYMENT, confirmed: Any = False) -> dict[str, Any]:
    namespace = NAMESPACE if namespace in (None, "") else namespace
    deployment = DEPLOYMENT if deployment in (None, "") else deployment
    if namespace != NAMESPACE:
        raise ToolInputError(f"cluster changes are only allowed in namespace '{NAMESPACE}'")
    if deployment != DEPLOYMENT:
        raise ToolInputError(f"only deployment '{DEPLOYMENT}' may be restarted")
    if confirmed is not True:
        raise ToolInputError("explicit user confirmation is required before changing cluster state")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _patch_deployment({"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": stamp}}}}})
    return {"namespace": NAMESPACE, "deployment": DEPLOYMENT, "action": "rollout restart requested",
            "restarted_at": stamp, "note": "requested only; use k8s_list_pods / k8s_list_deployments to see the result."}


TOOLS: list[ToolSpec] = [
    ToolSpec("k8s_list_deployments", "Deployments of the ML app: desired/ready/available replicas, image, version label, conditions.", {}, list_deployments),
    ToolSpec("k8s_list_pods", "Pods of the ML app: phase, ready count, restarts, node, age, container states.", {}, list_pods),
    ToolSpec("k8s_get_pod_status", "Detailed status of one app pod: problems, conditions, probes, resources and its warning events.",
             {"pod_name": "name of an app pod (from k8s_list_pods)"}, get_pod_status),
    ToolSpec("k8s_get_pod_logs", "Log tail of one app pod.",
             {"pod_name": "name of an app pod", "tail_lines": "optional int 1-200, default 50",
              "previous": "optional bool: logs of the previous (crashed) container, default false"}, get_pod_logs),
    ToolSpec("k8s_list_services", "Services of the ML app: type, cluster IP, ports, selector and ready endpoint count.", {}, list_services),
    ToolSpec("k8s_find_unhealthy_pods", "Health check: unhealthy pods with reasons (restarts, OOMKilled, crash loops, not ready), deployment issues and warning events.", {}, find_unhealthy_pods),
    ToolSpec("k8s_get_events", "Recent events for the ML app objects.",
             {"only_warnings": "optional bool, default true", "limit": "optional int 1-50, default 20"}, get_events),
    ToolSpec("k8s_get_deployed_version", "Deployed application version from Deployment label, image tag and ConfigMap, with mismatches.", {}, get_deployed_version),
    ToolSpec("k8s_rollout_restart", "Restart the app deployment (needs explicit user confirmation; approved namespace only).",
             {"namespace": "optional; must be mlops-agent", "deployment": "optional; must be breast-cancer-api",
              "confirmed": "set only by the system after the user approves; never guess"},
             rollout_restart, mutating=True),
]