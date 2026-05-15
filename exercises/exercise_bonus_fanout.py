"""Bonus — Multi-reviewer fan-out với LangGraph Send API.

Khi confidence < 58%, thay vì hỏi 1 reviewer, agent gửi cùng bộ câu hỏi
cho 2 reviewer song song (2 thread riêng biệt). Khi cả 2 trả lời xong,
node merge gộp câu trả lời lại rồi synthesize 1 review duy nhất.

Flow:
    fetch_pr → analyze → route
        ↓ (escalate)
    fan_out ──Send──→ reviewer_1 (interrupt)
             └─Send──→ reviewer_2 (interrupt)
    merge_answers → synthesize → commit

Run:
    python exercises/exercise_bonus_fanout.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from typing import Annotated

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()

# Mở rộng ReviewState để chứa answers từ nhiều reviewer
class FanoutState(ReviewState, total=False):
    reviewer_answers: Annotated[list[dict], lambda a, b: a + b]  # reducer: gộp list


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    return {"pr_title": pr.title, "pr_diff": pr.diff,
            "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": (
                "You are a senior code reviewer. Provide structured output. "
                "Rate confidence using these STRICT bands:\n"
                "- 0.74–0.95: trivial mechanical change with ZERO open questions.\n"
                "- 0.60–0.72: small feature, mostly safe, 1–2 minor open questions.\n"
                "- 0.30–0.57: security red flags — weak hashing, SQL injection, "
                "plaintext secrets, missing auth, hard-coded credentials.\n"
                "If confidence < 0.58, populate escalation_questions with 2–4 specific questions."
            )},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}")
    return {"analysis": analysis}


def node_route(state):
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:   decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:      decision = "escalate"
    else:                             decision = "human_approval"
    console.print(f"[cyan]→ route[/cyan] → [bold]{decision}[/bold] ({c:.0%})")
    return {"decision": decision}


def node_fan_out(state: FanoutState):
    """Tạo 2 nhánh song song — mỗi nhánh là 1 reviewer riêng."""
    console.print("[cyan]→ fan_out[/cyan] — gửi câu hỏi cho 2 reviewer song song")
    questions = state["analysis"].escalation_questions or ["What is the intent of this PR?"]
    # Send tạo 2 nhánh chạy song song, mỗi nhánh nhận reviewer_id khác nhau
    return [
        Send("reviewer_node", {**state, "reviewer_id": "reviewer_1", "questions": questions}),
        Send("reviewer_node", {**state, "reviewer_id": "reviewer_2", "questions": questions}),
    ]


def node_reviewer(state: dict):
    """Mỗi reviewer node chạy độc lập trong nhánh riêng."""
    reviewer_id = state["reviewer_id"]
    questions = state["questions"]
    console.print(f"[cyan]→ reviewer ({reviewer_id})[/cyan]")

    answers = interrupt({
        "kind": "escalation",
        "reviewer_id": reviewer_id,
        "confidence": state["analysis"].confidence,
        "summary": state["analysis"].summary,
        "questions": questions,
    })

    console.print(f"  [green]✓[/green] {reviewer_id} answered {len(answers)} question(s)")
    # reviewer_answers dùng list reducer → tự động gộp kết quả từ cả 2 nhánh
    return {"reviewer_answers": [{"reviewer": reviewer_id, "answers": answers}]}


def node_merge_answers(state: FanoutState):
    """Gộp câu trả lời từ cả 2 reviewer thành 1 context."""
    console.print("[cyan]→ merge_answers[/cyan]")
    all_answers = state.get("reviewer_answers", [])
    console.print(f"  [green]✓[/green] nhận được {len(all_answers)} bộ câu trả lời")

    # Tạo combined Q&A: nếu 2 reviewer trả lời khác nhau → hiện cả 2
    combined: dict[str, list[str]] = {}
    for block in all_answers:
        rid = block["reviewer"]
        for q, a in block["answers"].items():
            combined.setdefault(q, [])
            combined[q].append(f"[{rid}] {a}")

    # Lưu vào escalation_answers để node_synthesize dùng
    merged = {q: " | ".join(answers) for q, answers in combined.items()}
    return {"escalation_answers": merged}


def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM synthesizing with merged answers...[/dim]"):
        refined = llm.invoke([
            {"role": "system", "content": "Refine the code review using answers from multiple reviewers."},
            {"role": "user", "content": f"Diff:\n{state['pr_diff']}\n\nMerged Q&A:\n{qa}"},
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request",
        "confidence": a.confidence,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("escalation_answers"):
        lines.append("\n_Multi-reviewer Q&A:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}**\n> {ans}")
    body = "\n".join(lines)

    if state.get("escalation_answers") or state.get("human_choice") == "approve":
        try:
            post_review_comment(state["pr_url"], body)
            console.print(f"  [green]✓[/green] posted comment")
            return {"final_action": "committed"}
        except Exception as e:
            console.print(f"  [red]✗[/red] {e}")
            return {"final_action": "commit_failed"}
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]")
    a = state["analysis"]
    body = f"### Auto review (confidence {a.confidence:.0%})\n\n{a.summary}"
    try:
        post_review_comment(state["pr_url"], body)
        return {"final_action": "auto_committed"}
    except Exception as e:
        console.print(f"  [red]✗[/red] {e}")
        return {"final_action": "commit_failed"}


def build_fanout_graph():
    g = StateGraph(FanoutState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("fan_out", node_fan_out), ("reviewer_node", node_reviewer),
        ("merge_answers", node_merge_answers), ("synthesize", node_synthesize),
        ("human_approval", node_human_approval), ("commit", node_commit),
        ("auto_approve", node_auto_approve),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "fan_out"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    # fan_out dùng Send → LangGraph tự route vào reviewer_node (không cần add_edge)
    # Sau khi cả 2 reviewer_node xong → merge
    g.add_edge("reviewer_node", "merge_answers")
    g.add_edge("merge_answers", "synthesize")
    g.add_edge("synthesize", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload: dict) -> dict:
    kind = payload["kind"]
    reviewer_id = payload.get("reviewer_id", "reviewer")
    if kind == "approval_request":
        console.print(Panel.fit(payload["summary"], title="Approve?", border_style="green"))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    # escalation
    console.print(Panel.fit(
        payload["summary"],
        title=f"Escalation — {reviewer_id} (conf={payload['confidence']:.0%})",
        border_style="yellow",
    ))
    return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Bonus — Multi-reviewer fan-out[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_fanout_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = asyncio.get_event_loop().run_until_complete(
        app.ainvoke({"pr_url": args.pr, "thread_id": thread_id, "reviewer_answers": []}, cfg)
    ) if False else app.invoke(
        {"pr_url": args.pr, "thread_id": thread_id, "reviewer_answers": []}, cfg
    )

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        result = app.invoke(Command(resume=handle_interrupt(payload)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")


if __name__ == "__main__":
    main()
