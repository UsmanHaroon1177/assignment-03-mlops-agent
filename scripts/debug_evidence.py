"""Print the evidence names the model cited in its final answer (debug only)."""
import json
import sys

from dotenv import load_dotenv

from agent.llm import GeminiLLM
from agent.orchestrator import Orchestrator
from agent.validation import parse_action

load_dotenv()
llm = GeminiLLM()
orig = llm.generate
last = {}


def wrapped(system, messages):
    text = orig(system, messages)
    last["text"] = text
    return text


llm.generate = wrapped
resp = Orchestrator(llm).run(" ".join(sys.argv[1:]))
action = parse_action(last["text"])
for f in action.known_facts:
    print(json.dumps({"claim": f.claim[:70], "evidence": f.evidence}))
print("ok tools:", sorted({r.tool for r in resp.tool_results if r.status == "ok"}))