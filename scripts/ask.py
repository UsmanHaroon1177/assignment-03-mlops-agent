"""Ask the agent one question using real Gemini + real tools.

  python -m scripts.ask --list-models
  python -m scripts.ask "Which dataset version was used for the best run?"
  python -m scripts.ask --verbose "Is my prediction API healthy?"
Pending state-changing actions are shown and need an explicit 'yes' here.
"""
import argparse
import json
import sys

from dotenv import load_dotenv

from agent.llm import GeminiLLM, LLMError, list_models
from agent.logging_utils import configure
from agent.orchestrator import Orchestrator


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--verbose", action="store_true", help="show structured JSON logs")
    a = p.parse_args()
    load_dotenv()
    configure("INFO" if a.verbose else "ERROR")

    if a.list_models:
        try:
            print("\n".join(list_models()))
        except LLMError as exc:
            print(f"could not list models: {exc}")
            return 1
        return 0
    if not a.question:
        p.error("question is required")

    llm = GeminiLLM()
    print(f"models (in order): {llm.models}\n")
    orch = Orchestrator(llm)
    resp = orch.run(a.question)
    print(resp.answer)
    print(f"\nanswered by: {llm.last_model}")
    print("\n-- tool evidence --")
    for r in resp.tool_results:
        print(f"{r.tool:35s} {r.status:12s} {r.duration_ms} ms")
    if resp.llm_error:
        print(f"\nLLM error: {resp.llm_error}")
    for pend in resp.pending_actions:
        if input(f"\nConfirm [{pend.id}] {pend.tool}? Type 'yes' to run: ").strip().lower() == "yes":
            res = orch.confirm(pend.id)
            print(f"{res.tool}: {res.status}", json.dumps(res.data or {"error": res.error}, indent=2)[:1500])
        else:
            orch.reject(pend.id)
            print("not executed")
    return 0


if __name__ == "__main__":
    sys.exit(main())