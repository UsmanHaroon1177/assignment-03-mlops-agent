"""Real-cluster smoke test for the read-only Kubernetes tools (never restarts anything).
Usage: python -m scripts.try_k8s
"""
import json

from agent.tools import kubernetes_tools as k8s
from agent.tools.base import execute

TOOL = {t.name: t for t in k8s.TOOLS}


def show(name, **kw):
    r = execute(name, TOOL[name].fn, request_id="smoke", **kw)
    print(f"\n== {name} {kw} -> {r.status} ({r.duration_ms} ms)")
    print(json.dumps(r.data if r.data is not None else {"error": r.error}, indent=2)[:2500])
    return r


def main():
    show("k8s_list_deployments")
    pods = show("k8s_list_pods")
    show("k8s_list_services")
    show("k8s_find_unhealthy_pods")
    show("k8s_get_events", only_warnings=False, limit=8)
    show("k8s_get_deployed_version")
    if pods.status == "ok" and pods.data["pods"]:
        name = pods.data["pods"][0]["name"]
        show("k8s_get_pod_status", pod_name=name)
        show("k8s_get_pod_logs", pod_name=name, tail_lines=10)
    print("\n-- guard checks (must be rejected) --")
    show("k8s_rollout_restart")
    show("k8s_rollout_restart", namespace="kube-system", confirmed=True)


if __name__ == "__main__":
    main()