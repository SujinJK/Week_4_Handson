"""A hand-rolled agentic loop -- no LangChain, no LangGraph.

Demonstrates four agentic design patterns explicitly, each as its own
visible step instead of something a framework does for you behind the
scenes:

  1. PLANNING   -- a dedicated call asks Claude to sketch a short plan
                   before touching any tool.
  2. TOOL USE   -- a manual `while` loop detects stop_reason == "tool_use",
                   runs the matching Python function, and feeds the result
                   back in a tool_result block. A server-side tool
                   (web_search) is mixed into the same tool list.
  3. REFLECTION -- after a draft answer, a second call with its own system
                   prompt (a "critic") checks it and can send it back for
                   one bounded retry.
  4. MULTI-AGENT-- the critic in step 3 is a second role with its own
                   system prompt and its own structured output schema, not
                   just a re-prompt of the same agent -- a minimal
                   worker/critic pair. See README.md for how this scales to
                   full multi-agent orchestration.

Run:
    python ingest.py   # once, to build the vector store (see rag_tool.py)
    python agent.py     # interactive loop
"""
import json
import pathlib
import sys

import anthropic
from dotenv import load_dotenv

# Claude's output can contain Unicode characters (e.g. a proper minus sign,
# "-", U+2212) that Windows' default console codepage (cp1252) has no
# mapping for -- printing one crashes with UnicodeEncodeError instead of
# just showing a wrong glyph. Force UTF-8 on stdout so that can't happen.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import ingest
from tools import CLIENT_TOOLS, TOOLS

load_dotenv()

MODEL = "claude-opus-4-8"
MAX_TOOL_ITERATIONS = 10
MAX_REFLECTION_CYCLES = 2

PLAN_SYSTEM_PROMPT = (
    "You are the planning step of an agent with three tools: `calculator` "
    "(arithmetic), `search_knowledge_base` (searches whatever documents are "
    "currently indexed in the local knowledge base -- contents vary, so try "
    "it for any question that might depend on indexed documents rather than "
    "assuming a fixed topic), and `web_search` (the public web). Given the "
    "user's question, write a short plan -- 2 to 4 bullet points -- naming "
    "which of these tools you expect to need, in what order, and why. Do "
    "not answer the question yet, and do not call any tool. Just the plan."
)

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to three tools: `calculator`, "
    "`search_knowledge_base` (searches whatever documents are currently "
    "indexed in the local knowledge base), and `web_search` (the public "
    "web). Follow the plan you were given, calling tools as needed -- do "
    "not guess at facts that might be covered by the knowledge base without "
    "checking search_knowledge_base first. When you cite a "
    "search_knowledge_base snippet, reference its number like this: [1]. "
    "Once you are confident in your final answer, state it directly "
    "instead of calling more tools."
)

CRITIC_SYSTEM_PROMPT = (
    "You are a strict reviewer checking another agent's draft answer "
    "before it reaches the user. Given the original question and the "
    "draft answer, check: (1) does it actually answer what was asked, "
    "(2) is every factual or policy claim grounded in a cited source "
    "rather than guessed, (3) is any calculation or lookup missing that "
    "the question requires. Respond with verdict 'approve' if the answer "
    "is good as-is, or 'revise' with a list of specific, actionable issues "
    "if not."
)

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "revise"]},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "issues"],
    "additionalProperties": False,
}


def make_plan(client: anthropic.Anthropic, question: str) -> str:
    """Planning step: ask Claude to sketch its approach before it acts."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=PLAN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


_SERVER_TOOL_BLOCK_TYPES = {
    "server_tool_use",
    "web_search_tool_result",
    "web_fetch_tool_result",
    "bash_code_execution_tool_result",
    "code_execution_tool_result",
    "text_editor_code_execution_tool_result",
}


def _drop_server_tool_blocks(messages: list[dict]) -> list[dict]:
    """Strip server-side-tool content blocks out of assistant turns.

    Recovery step for the container_id BadRequestError below: web_search's
    dynamic filtering runs code_execution under the hood, and if a
    client-side tool call cuts the turn short while that session is still
    open, the API leaves it "pending" with no documented way for us to
    resume it (response.container is empty in that case, not the
    populated id the docs describe for the standalone code_execution
    tool). Removing the orphaned blocks from history clears the pending
    state so the conversation can continue without web_search."""
    cleaned = []
    for msg in messages:
        if msg["role"] != "assistant" or not isinstance(msg["content"], list):
            cleaned.append(msg)
            continue
        kept = [b for b in msg["content"] if getattr(b, "type", None) not in _SERVER_TOOL_BLOCK_TYPES]
        if kept:
            cleaned.append({"role": "assistant", "content": kept})
    return cleaned


def run_agent_loop(client: anthropic.Anthropic, messages: list[dict]) -> tuple[str, list[dict]]:
    """The manual act/observe loop: call the API, and while Claude is asking
    for a client-side tool, run it and feed the result back. Also handles
    pause_turn, which the server-side web_search tool returns if its own
    internal search loop hits its iteration cap -- resending the same
    turn lets it resume automatically (see shared/tool-use-concepts.md).

    For readers new to this: `client.messages.create(...)` below is one
    network request to Claude -- you send the conversation so far
    (`messages`) and get one `response` back. `response.stop_reason` is
    Claude's way of telling us *why* it stopped generating text: because
    it wants to use a tool ("tool_use"), because it's fully done
    ("end_turn"), or because a server-side tool needs the turn resent
    ("pause_turn"). Reading that one field is how this whole loop decides
    what to do next -- run a tool and go again, or return the final
    answer.
    """
    tools = TOOLS
    web_search_dropped = False
    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=AGENT_SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                tools=tools,
                messages=messages,
            )
        except anthropic.BadRequestError as exc:
            if web_search_dropped or "container_id is required" not in str(exc):
                raise
            # A client-tool call interrupted a web_search-driven code-execution
            # session with no way for us to resume it -- see
            # _drop_server_tool_blocks(). Recover once by clearing the orphaned
            # blocks and continuing the rest of this turn without web_search.
            print("  [recovering from a stuck web_search session -- retrying without web_search]")
            messages = _drop_server_tool_blocks(messages)
            tools = [t for t in TOOLS if t.get("name") != "web_search"]
            web_search_dropped = True
            continue

        messages = messages + [{"role": "assistant", "content": response.content}]

        # Server-side web_search calls never hit the "tool_use" branch below
        # (Anthropic executes them, not us) -- log them here instead, or the
        # trace would show a search happened only by implication.
        for block in response.content:
            if block.type == "server_tool_use":
                print(f"  -> [server] {block.name}({block.input})")
            elif block.type == "web_search_tool_result":
                content = block.content
                if isinstance(content, list):
                    titles = [r.title for r in content[:3] if hasattr(r, "title")]
                    print(f"  <- [server] {len(content)} result(s): {titles}")
                else:
                    print(f"  <- [server] web_search error: {content}")

        if response.stop_reason == "pause_turn":
            print("  [web_search hit its internal step limit -- resuming]")
            continue

        if response.stop_reason == "max_tokens":
            # Silently returning a partial answer here would look identical to a
            # real final answer -- flag it instead of letting truncation hide.
            print("  [warning: hit max_tokens -- answer below may be truncated]")

        if response.stop_reason != "tool_use":
            # Concatenate every text block, not just the first -- Claude can
            # split its answer across more than one text block in the same
            # response (e.g. text before and after a thinking block), and
            # taking only the first one silently truncates the rest.
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return final_text, messages

        # stop_reason == "tool_use": at least one client-side tool call is
        # pending (server-side web_search calls, if any, are already
        # resolved inline and show up as server_tool_use/_result blocks,
        # which the loop below skips since their type isn't "tool_use").
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_fn = CLIENT_TOOLS.get(block.name)
            if tool_fn is None:
                continue
            print(f"  -> {block.name}({block.input})")
            result = tool_fn(**block.input)
            preview = result if len(result) <= 200 else result[:200] + "..."
            print(f"  <- {preview}")
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages = messages + [{"role": "user", "content": tool_results}]

    raise RuntimeError(f"Agent did not finish within {MAX_TOOL_ITERATIONS} tool iterations")


def critique_answer(client: anthropic.Anthropic, question: str, answer: str) -> dict:
    """Reflection step: a second, differently-prompted call reviews the draft answer."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=CRITIC_SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": CRITIQUE_SCHEMA}},
        messages=[{"role": "user", "content": f"Question: {question}\n\nDraft answer:\n{answer}"}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def answer_question(client: anthropic.Anthropic, question: str) -> str:
    """Run one full agent turn: plan, act (with tools), reflect, and retry
    at most MAX_REFLECTION_CYCLES times if the critic flags real issues."""
    print("\nPLAN:")
    plan = make_plan(client, question)
    print(plan)

    messages = [{"role": "user", "content": f"Question: {question}\n\nYour plan:\n{plan}"}]

    print("\nACTING:")
    answer, messages = run_agent_loop(client, messages)

    for cycle in range(1, MAX_REFLECTION_CYCLES + 1):
        print("\nREFLECTION:")
        critique = critique_answer(client, question, answer)
        print(f"  verdict={critique['verdict']}")
        for issue in critique["issues"]:
            print(f"  - {issue}")

        if critique["verdict"] == "approve" or cycle == MAX_REFLECTION_CYCLES:
            break

        feedback = "; ".join(critique["issues"]) or "Double-check completeness and grounding."
        messages = messages + [{
            "role": "user",
            "content": f"A reviewer flagged issues with your answer: {feedback}\nAddress them and give a final answer.",
        }]
        print("\nACTING (retry):")
        answer, messages = run_agent_loop(client, messages)

    print(f"\nFINAL ANSWER:\n{answer}\n")
    return answer


def main() -> None:
    """Entry point for `python agent.py` — the interactive question loop."""
    if not pathlib.Path(ingest.DB_DIR).exists():
        print("No vector store found. Run `python ingest.py` first.")
        return

    client = anthropic.Anthropic()
    print("Nimbus Agent. Ask a question (math, policy, or general knowledge), or /quit to exit.\n")
    while True:
        question = input("you> ").strip()
        if not question:
            continue
        if question in ("/quit", "/exit"):
            break
        answer_question(client, question)


if __name__ == "__main__":
    main()
