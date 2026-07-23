# Week 4 — Agentic Design Patterns, From Scratch

A hand-rolled agent for **Nimbus Cloud Storage** support questions. No LangChain, no LangGraph — the entire agent loop (planning, tool calls, reflection) is plain Python calling the Claude API directly, so every mechanic a framework would normally hide is visible in the code.

It reuses the RAG retriever built in [Week 3](../Week_3_Handson) as one of its tools, alongside a calculator and live web search.

## What it does

Ask it a question — Nimbus policy, plain arithmetic, or something needing the open web — and it will:

1. **Plan** — sketch which tools it expects to need, before touching any of them.
2. **Act** — call tools (in a loop, since answering may take more than one tool call) and observe their results.
3. **Reflect** — a second, differently-prompted pass reviews the draft answer for grounding and completeness before it reaches you, and can send it back for one retry if something's wrong.

Every step prints to the terminal, so you can watch the plan, each tool call and its result, and the reviewer's verdict — not just the final answer.

## Pipeline diagram

```
question
   │
   ▼
┌─────────────┐
│   PLAN      │  one Claude call, no tools — "what will I need, and why?"
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│   ACT / OBSERVE  (loop)                      │
│   while stop_reason == "tool_use":           │
│     - calculator            (we run it)      │
│     - search_knowledge_base (we run it,      │
│       reusing Week 3's hybrid+rerank RAG)     │
│     - web_search             (Claude/Anthropic│
│       run it server-side, results come back  │
│       inline)                                │
└──────┬────────────────────────────────────────┘
       │  draft answer
       ▼
┌─────────────┐
│  REFLECT    │  a second call, different system prompt ("critic"),
│             │  structured output: {verdict, issues}
└──────┬──────┘
       │
       ├── verdict == "revise" → feed issues back → ACT/OBSERVE again (max 2 cycles)
       └── verdict == "approve" → final answer printed
```

## Project structure

```
Week_4_Handson/
├── agent.py          # the whole agent: plan / act-observe loop / reflect
├── tools.py           # calculator + search_knowledge_base + tool schemas
├── rag_tool.py         # Week 3's hybrid search + rerank, trimmed to one function
├── ingest.py           # builds the Chroma vector store (same as Week 3)
├── chunking.py         # semantic chunker (copied from Week 3, unchanged)
├── corpus/             # the Nimbus sample docs (copied from Week 3)
├── tests/
│   ├── test_tools.py    # calculator arithmetic + safety (rejects non-math input)
│   └── test_chunking.py # chunking correctness (same tests as Week 3)
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for pytest
cp .env.example .env                  # then add your real ANTHROPIC_API_KEY
python ingest.py                      # builds chroma_db/ from corpus/
```

## Running it

```bash
python agent.py
```

```
you> If I overpaid a $49.99 subscription by mistake and got charged twice, how much should Nimbus refund me, and what's the refund window?

PLAN:
- Use search_knowledge_base to find Nimbus's refund policy and window.
- Use calculator to compute the refund amount if a formula is needed.
- Answer directly if no live/current information is required.

ACTING:
  -> search_knowledge_base({'query': 'refund policy duplicate charge overpayment'})
  <- [1] (source: refund_policy.md) ...
  -> calculator({'expression': '49.99'})
  <- 49.99

REFLECTION:
  verdict=approve

FINAL ANSWER:
Nimbus would refund the duplicate charge of $49.99 [1], and refund requests
must be submitted within 30 days of the charge [1].
```

## Example prompts to try

Each of these exercises a different part of the pipeline. Run `python agent.py`, paste one in at the `you>` prompt, and watch which tool(s) get called and whether the reflection step approves or revises.

| # | Prompt | What it tests | Expected grounding (from `corpus/`) |
|---|---|---|---|
| 1 | `If I cancel my annual plan 10 days after buying it, do I get a refund?` | Pure `search_knowledge_base` — single lookup, no math | Yes — annual plans get a prorated refund if canceled within the first 30 days (`refund_policy.md`) |
| 2 | `I'm on the Business plan and got double-charged this month. If Nimbus refunds the error and I also want to downgrade to Pro, how much will I be refunded and what will my new monthly rate be?` | `search_knowledge_base` + `calculator` chained | Refund: full $60 duplicate charge within 5 business days. New rate: $15/month, effective next billing cycle (not immediate) |
| 3 | `What's 15% of $2,400?` | Pure `calculator`, plan should skip the knowledge base entirely | $360 |
| 4 | `What's the latest stable version of Python, and does Nimbus's desktop app support it?` | Forces `web_search`; no fixed "correct" answer — checks whether the agent admits the second half isn't documented instead of fabricating a compatibility claim | Corpus never mentions Python/tech-stack requirements for the desktop app at all |
| 5 | `Does Nimbus offer a lifetime storage plan?` | Grounding on an absence — no such plan exists | No — only Starter/Pro/Business, all monthly subscriptions (`product_faq.md`) |
| 6 | `If a file is deleted from Trash after the 30-day window, can Nimbus's support team recover it from backups, and how long do backups last?` | Designed to trip the reflection/retry loop — a wrong answer would invent a backup retention period | No recovery either way — Nimbus keeps no backups beyond the 30-day Trash window at all (`product_faq.md`) |
| 7 | `As an employee, what MFA method should I use to access a customer's Tier 3 billing data, and could this ever qualify as a reportable security incident?` | Cross-section grounding — tests whether it conflates authorized access with an incident | Hardware keys preferred (or TOTP; SMS last resort). Authorized Tier 3 access (with manager approval) isn't an incident — only unauthorized access is, reportable to security@nimbus.example within 1 hour (`security_policy.md`) |

If an answer diverges from the expected grounding above, check the `REFLECTION:` section first — a good run of this agent should have the critic (verdict `revise`) catch it before it reaches you.

## The four agentic patterns, and where to find them

| Pattern | Where | What it looks like here |
|---|---|---|
| **Planning** | `make_plan()` in `agent.py` | A separate, tool-free Claude call that writes 2-4 bullet points on what it expects to need before it does anything. Keeps "deciding what to do" visibly separate from "doing it." |
| **Tool use** | `run_agent_loop()` in `agent.py` | A plain `while` loop: call the API, check `stop_reason`, run any pending client-side tool, feed the result back as a `tool_result`, repeat until Claude stops asking for tools. |
| **Reflection** | `critique_answer()` + the retry loop in `answer_question()` | A second call, with a *different* system prompt ("You are a strict reviewer..."), scores the draft answer and can force up to 2 retry cycles with specific feedback fed back into the loop. |
| **Multi-agent (minimal)** | The worker (`AGENT_SYSTEM_PROMPT`) + critic (`CRITIC_SYSTEM_PROMPT`) pair | Two distinct roles, two distinct system prompts, two distinct jobs — one answers, one checks. This is the smallest possible multi-agent system: a worker and a reviewer. See "Where this stops" below for how far it's from real multi-agent orchestration. |

## The three tools

| Tool | Kind | What it does |
|---|---|---|
| `calculator` | client-side (we execute it) | Evaluates a plain arithmetic expression via Python's `ast` module — restricted to numbers and `+ - * / ** % //` and parentheses. No `eval()` anywhere: a name lookup, function call, or attribute access is rejected outright, so the model can never smuggle code through the expression string. See `tests/test_tools.py` for the rejection tests. |
| `search_knowledge_base` | client-side (we execute it) | Wraps Week 3's proven retrieval pipeline: vector search + BM25 keyword search fused with Reciprocal Rank Fusion, then cross-encoder reranking down to the top 3 chunks. Same corpus, same code, just packaged as a callable tool instead of a standalone script. |
| `web_search` | **server-side** | Declared with `{"type": "web_search_20260209", "name": "web_search"}` — Anthropic runs the actual search and returns results inline in the same response. Our loop never executes this one; it only has to recognize that a `tool_use`-type block with this name never appears (server tool activity shows up as different block types), and to resend the turn unchanged if the server-side search hits its own internal step limit (`stop_reason == "pause_turn"`). We still log it (see `run_agent_loop()`) by scanning for `server_tool_use` / `web_search_tool_result` blocks — otherwise a search would happen invisibly, which would defeat the point of a hand-rolled, fully-visible loop. |

Mixing a server-side tool into the same `tools` list as two client-side ones was the trickiest part of the loop to get right — see "Things to be aware of" below.

## Things to be aware of

- **`stop_reason` has three cases that matter here, not two.** `"tool_use"` means at least one *client-side* tool call is waiting on us; `"end_turn"` means Claude is done; `"pause_turn"` means the server-side `web_search` tool hit its own internal iteration cap mid-turn and needs the exact same conversation resent (not a new "continue" message) so it can resume. Missing the third case makes the loop silently return an incomplete answer instead of finishing the search.
- **Client-side and server-side tool activity can appear in the very same response.** If Claude calls `search_knowledge_base` *and* `web_search` in one turn, the response's `stop_reason` is `"tool_use"` (because of the pending client tool), while the web search's blocks (`server_tool_use`, `web_search_tool_result`) are already fully resolved and just get skipped over when the loop scans for blocks of type `"tool_use"`.
- **The reflection retry is bounded.** `MAX_REFLECTION_CYCLES = 2` — without a cap, a critic that's never satisfied (or a worker that can't fix what's flagged) would loop forever.
- **No conversation memory across questions**, same limitation as Week 3's RAG — every question starts a fresh `messages` list.
- **The planning step doesn't have tools attached and never resolves any** — it's a pure "think before you act" call. It doesn't force the acting loop to follow the plan exactly; the plan is passed into the acting loop's context as guidance, and Claude is free to deviate if the plan turns out to be wrong once it actually starts searching.

## Bugs we actually hit while building this (and what they taught us)

Five real bugs showed up the first few times the loop ran against live questions, not in theory — worth recording honestly rather than pretending the first version worked:

- **Reading only the first text block silently truncated answers.** The initial code pulled the final answer with `next(b.text for b in response.content if b.type == "text")` — grabbing only the *first* text block. Claude can split its answer across more than one text block in a single response (for instance, text before and after it revises its own phrasing mid-turn). The result: answers that cut off mid-sentence (`"...Per Nimbus policy, "`) with no error and no warning — the reflection step actually caught this every time (a real point in favor of the reflection pattern), flagging the answer as incomplete, but the bounded retry couldn't fix a bug in *our* code, only in the model's *content*. Fixed by concatenating every text block: `"".join(b.text for b in response.content if b.type == "text")`.
- **`max_tokens` was tuned too low for adaptive thinking plus tool history.** Bumped from 2048 to 4096 as a first attempt at fixing the truncation above. This turned out not to be the actual root cause (the text-block bug was), but it's a real, separate risk — the loop now explicitly checks `stop_reason == "max_tokens"` and prints a warning, since a silently truncated answer is otherwise indistinguishable from a complete one.
- **The critic has no notion of "today's date."** Neither system prompt mentions the current date, so when `web_search` returned a real, current Python release, the critic flagged the release date as "implausible" and "likely fabricated" purely because it postdates its own training data — a false positive, not a real problem with the answer. This is a legitimate limitation of the reflection pattern as built here: a critic reasoning from training knowledge alone will sometimes distrust correct, current information pulled in by a tool it doesn't fully "trust." A more robust fix would pass today's date (and ideally the raw retrieved snippets, not just the draft answer) into the critic's context.
- **A Windows console encoding crash, not just a wrong character.** Claude occasionally writes a proper Unicode minus sign (`−`, U+2212) instead of a plain hyphen in its plan text. Windows' default console codepage (`cp1252`) has no mapping for it, so a plain `print()` didn't just show the wrong glyph — it raised `UnicodeEncodeError` and killed the whole run. Fixed by forcing `sys.stdout` to UTF-8 at startup (`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`) so any Unicode Claude produces prints safely regardless of the terminal's default codepage.
- **A genuinely gnarly one: `web_search`'s internal code execution can get "stuck" mid-turn.** `web_search_20260209`'s dynamic filtering runs `code_execution` under the hood. When Claude called `search_knowledge_base` (our client-side tool) in the *same* turn as that internal code-execution activity, our loop had to pause to resolve the client tool — and the next API call then 400'd with `"container_id is required when there are pending tool uses generated by code execution with tools."` The documented fix for the standalone `code_execution` tool is to track `response.container.id` and pass it back on the next call, but for this hidden, web_search-driven session `response.container` was empty every time we checked — there's no documented way to resume it. The fix that actually worked: catch that specific `BadRequestError`, strip any orphaned server-tool content blocks out of the conversation history, drop `web_search` from the tool list, and retry — the model already had what the client tool found, so it can usually finish the answer without the web anyway (and it did, honestly noting web search "isn't actually available... in this environment" instead of pretending it worked). This is a workaround, not a documented root-cause fix — flagging it as an open question rather than a solved one.
- **A hardcoded tool description silently disabled the tool when the underlying corpus changed.** Both `PLAN_SYSTEM_PROMPT` and `AGENT_SYSTEM_PROMPT` originally described `search_knowledge_base` as "Nimbus Cloud Storage's internal policy documents." That was accurate when `chroma_db/` only ever held the Nimbus corpus — but during testing, `chroma_db/` was rebuilt from an unrelated PDF (a dissertation, via a different corpus directory passed to `ingest.py`), and the tool itself worked fine when called directly. The agent, however, never called it: the planning step read its own prompt's claim about the tool's scope, reasoned "a dissertation wouldn't be in a Nimbus-policy tool," and skipped it — a confident, well-reasoned refusal built on a stale premise, not a retrieval failure. The lesson is that a tool's *documented* purpose in a system prompt and its *actual* current contents can silently drift apart, and nothing in the planning step can catch that, since it reasons from what it's told rather than what's indexed. Fixed by rewording both prompts to describe the tool generically — "searches whatever documents are currently indexed" — instead of naming a fixed domain.

## What a framework like LangGraph would add

This project is deliberately "from scratch" so every mechanic is visible. Here's honestly what you'd give up by hand-rolling it, and what a framework buys you once the agent grows past this size:

- **State persistence / checkpointing.** If the process crashes mid-loop here, the whole conversation is gone — `messages` only lives in a Python variable. LangGraph (and similar frameworks) checkpoint state to disk/a database after every step, so a crashed run can resume exactly where it left off.
- **Built-in retry and error handling.** Our loop has one hardcoded retry cap (`MAX_TOOL_ITERATIONS`) and no automatic backoff on transient failures beyond what the `anthropic` SDK already does. Frameworks provide configurable retry policies per node.
- **Graph-based branching.** Our control flow is a straight line with one loop and one bounded retry, written directly in Python `if`/`for`. A framework represents this as an explicit graph of nodes and edges, which makes complex branching (e.g., "if the critic flags a grounding issue, go back to search; if it flags a math issue, go back to calculator") declarative instead of nested conditionals.
- **Visualization and tracing tooling.** We print to stdout. Frameworks typically ship a UI or trace viewer showing the graph, each node's input/output, and timing, without you writing any print statements.
- **Multi-agent orchestration at scale.** Our "multi-agent" pattern is two system prompts and two sequential calls. A real multi-agent framework manages many agents running concurrently, message-passing between them, and shared/scoped memory — a fundamentally bigger problem than a worker-critic pair.
- **Memory management across turns.** Nothing here persists between questions. Frameworks often provide built-in short-term/long-term memory abstractions.

None of this makes the framework "better" for a project this size — the whole point of Week 4 was to see what's actually happening under the hood before reaching for something that does it automatically.

## Where this stops (honest limitations)

- The "plan" isn't enforced — Claude can ignore it once it starts acting, and nothing checks that it was followed.
- The critic and the worker are the same model with different prompts, not genuinely independent agents — they can share the same blind spots, and can also produce false-positive critiques (observed: flagging a real, current fact as "fabricated" because it postdates the model's training — see "Bugs we actually hit" above).
- No persistence, no checkpointing, no resuming a crashed run.
- Tested corpus is the same small 5-document Nimbus sample from Week 3 — findings about tool selection quality don't necessarily generalize to a much larger or noisier tool surface.

## Testing

```bash
python -m pytest tests/ -q
```

`tests/test_tools.py` checks the calculator's arithmetic and, more importantly, that it *rejects* anything that isn't plain arithmetic (name lookups, function calls, attribute access, string literals) — since a bare `eval()` on model-supplied input would be a code-execution vulnerability. `tests/test_chunking.py` is carried over unchanged from Week 3 since `chunking.py` is the same code.
