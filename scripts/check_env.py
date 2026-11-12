"""Environment check for the MLOps agent project. Prints OK/WARN/FAIL per item."""
import importlib
import importlib.metadata as md
import subprocess
import sys

MODULES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("dotenv", "python-dotenv"),
    ("requests", "requests"),
    ("httpx", "httpx"),
    ("google.genai", "google-genai"),
    ("mlflow", "mlflow"),
    ("dvc", "dvc"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("kubernetes", "kubernetes"),
    ("pytest", "pytest"),
]

failures = 0


def report(level, name, detail=""):
    global failures
    if level == "FAIL":
        failures += 1
    print(f"[{level:4}] {name} {detail}".rstrip())


def run(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, shell=False)
        return res.returncode, (res.stdout or res.stderr).strip()
    except Exception as exc:  # missing binary, timeout, ...
        return 1, str(exc)


def main():
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (3, 10) <= (v.major, v.minor) <= (3, 13):
        report("OK", "python", ver)
    else:
        report("WARN", "python", f"{ver} (3.10-3.13 expected)")

    if sys.prefix == sys.base_prefix:
        report("WARN", "virtualenv", "not active - run .venv\\Scripts\\Activate.ps1")
    else:
        report("OK", "virtualenv", "active")

    for mod, dist in MODULES:
        try:
            importlib.import_module(mod)
            report("OK", f"module {mod}", md.version(dist))
        except Exception as exc:
            report("FAIL", f"module {mod}", f"{type(exc).__name__}: {exc}")

    for name, cmd in [
        ("git", ["git", "--version"]),
        ("dvc", ["dvc", "--version"]),
        ("docker", ["docker", "--version"]),
        ("kubectl", ["kubectl", "version", "--client"]),
    ]:
        code, out = run(cmd)
        if code == 0:
            report("OK", f"cli {name}", out.splitlines()[0] if out else "")
        else:
            report("FAIL", f"cli {name}", out[:120])

    code, out = run(["docker", "info", "--format", "{{.MemTotal}}"])
    if code == 0 and out.isdigit():
        gb = int(out) / (1024 ** 3)
        level = "OK" if 3.0 <= gb <= 6.0 else "WARN"
        report(level, "docker daemon", f"memory visible to Docker: {gb:.1f} GB")
    else:
        report("FAIL", "docker daemon", "not running? " + out[:120])

    code, out = run(["kubectl", "config", "current-context"])
    if code == 0 and out == "docker-desktop":
        report("OK", "kubectl context", out)
    else:
        report("FAIL", "kubectl context", f"{out!r} (expected docker-desktop)")

    code, out = run(["kubectl", "get", "nodes", "--no-headers"])
    if code == 0 and " Ready" in out:
        report("OK", "kubernetes node", out.split()[0] + " Ready")
    else:
        report("FAIL", "kubernetes node", out[:120] + " (still starting?)")

    print()
    print("ALL GOOD" if failures == 0 else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
