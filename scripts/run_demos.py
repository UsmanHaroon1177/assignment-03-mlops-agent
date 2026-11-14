"""Run the five demo scenarios against real services and save answers to evidence/.
Usage: python -m scripts.run_demos [1 2 3 4 5]   (no arguments = all)
Pending state-changing actions are listed but never confirmed here.
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.llm import GeminiLLM
from agent.logging_utils import configure
from agent.orchestrator import Orchestrator

SCENARIOS = {
    1: ("dataset_provenance",
        "Which dataset version was used to train the best run by f1, and how do I reproduce that dataset version?"),
    2: ("model_selection",
        "Compare the last five training runs and recommend the best candidate based on F1 score. "
        "Is the highest F1 enough to call it production-ready?"),
    3: ("pipeline_execution",
        "What is the status of the latest Airflow training run and of each of its tasks? If anything failed, "
        "explain what the available diagnostics show."),
    4: ("deployment_diagnosis",
        "Is my prediction API healthy on Kubernetes? Check pods, services, events, logs and the deployed version."),
    5: ("multi_tool",
        "Find the best recent model, identify the dataset version used to train it, check the deployment status, "
        "and explain whether retraining is necessary."),
}


def main() -> None:
    load_dotenv()
    configure("ERROR")
    wanted = [int(a) for a in sys.argv[1:]] or sorted(SCENARIOS)
    llm = GeminiLLM()
    orch = Orchestrator(llm)
    out_dir = Path("evidence")
    out_dir.mkdir(exist_ok=True)
    for n in wanted:
        name, question = SCENARIOS[n]
        resp = orch.run(question)
        lines = [f"SCENARIO {n}: {name}", f"QUESTION: {question}", f"REQUEST ID: {resp.request_id}",
                 f"MODEL: {llm.last_model}", "", resp.answer, "", "-- tool evidence --"]
        lines += [f"{r.tool:35s} {r.status:12s} {r.duration_ms} ms" for r in resp.tool_results]
        if resp.pending_actions:
            lines += ["", "PENDING (not executed): " + ", ".join(p.tool for p in resp.pending_actions)]
        text = "\n".join(lines)
        (out_dir / f"demo{n}_{name}.txt").write_text(text, encoding="utf-8")
        print(text, "\n" + "=" * 80)


if __name__ == "__main__":
    main()