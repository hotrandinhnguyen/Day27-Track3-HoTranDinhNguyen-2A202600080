"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path
from exercises.exercise_4_audit import build_graph

load_dotenv()

# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None

# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")

tab_review, tab_calibration, tab_timetravel = st.tabs(["Review", "Calibration", "Time-travel"])

# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    try:
        import aiosqlite

        async def _list_threads():
            async with aiosqlite.connect(db_path()) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(
                    """
                    SELECT thread_id, pr_url,
                           MAX(CASE WHEN risk_level='high' THEN 2
                                    WHEN risk_level='med'  THEN 1 ELSE 0 END) AS worst_risk,
                           MAX(timestamp) AS last_event
                    FROM audit_events
                    GROUP BY thread_id, pr_url
                    ORDER BY last_event DESC
                    LIMIT 10
                    """
                ) as cur:
                    return [dict(r) for r in await cur.fetchall()]

        threads = asyncio.run(_list_threads())
        for t in threads:
            risk_badge = {"2": "🔴", "1": "🟡", "0": "🟢"}.get(str(t["worst_risk"]), "⚪")
            label = f"{risk_badge} {t['pr_url'].split('/')[-1]}  `{t['thread_id'][:8]}`"
            if st.button(label, key=t["thread_id"]):
                st.session_state.thread_id = t["thread_id"]
                st.session_state.pr_url = t["pr_url"]
                st.rerun()
    except Exception:
        st.caption("(no sessions yet)")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")
    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])
    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, q in enumerate(payload.get("questions", [])):
            answers[q] = st.text_input(q, key=f"q_{i}")
        if st.form_submit_button("Submit answers"):
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Tab 1: Review ─────────────────────────────────────────────────────────
with tab_review:
    with st.form("start"):
        pr_url = st.text_input(
            "PR URL", value=st.session_state.pr_url,
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        )
        submitted = st.form_submit_button("Run review")

    if submitted and pr_url:
        st.session_state.pr_url = pr_url
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.interrupt_payload = None
        st.session_state.final = None
        with st.spinner("Fetching PR + asking the LLM..."):
            result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.final = result

    payload = st.session_state.interrupt_payload
    if payload is not None:
        kind = payload["kind"]
        answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
        if answer is not None:
            with st.spinner("Resuming..."):
                result = asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                ))
            if "__interrupt__" in result:
                st.session_state.interrupt_payload = result["__interrupt__"][0].value
            else:
                st.session_state.interrupt_payload = None
                st.session_state.final = result
            st.rerun()

    if st.session_state.final is not None:
        final = st.session_state.final
        action = final.get("final_action", "?")
        if action.startswith(("auto", "committed")):
            st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
        elif action == "rejected":
            st.warning("Rejected — no comment posted")
        else:
            st.info(f"final_action = {action}")
        st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
                   f"`python -m audit.replay --thread {st.session_state.thread_id}`")


# ─── Tab 2: Calibration ────────────────────────────────────────────────────
with tab_calibration:
    st.header("Confidence Calibration")
    st.caption("So sánh confidence LLM tự báo với tỷ lệ human thực sự approve.")

    try:
        import aiosqlite
        import pandas as pd

        async def _load_calibration():
            async with aiosqlite.connect(db_path()) as conn:
                conn.row_factory = aiosqlite.Row
                # Tổng quan theo decision
                async with conn.execute("""
                    SELECT decision,
                           COUNT(*) as count,
                           ROUND(AVG(confidence), 3) as avg_conf,
                           ROUND(MIN(confidence), 3) as min_conf,
                           ROUND(MAX(confidence), 3) as max_conf
                    FROM audit_events
                    WHERE action IN ('analyze', 'human_approval', 'auto_approve')
                    GROUP BY decision
                    ORDER BY avg_conf DESC
                """) as cur:
                    summary = [dict(r) for r in await cur.fetchall()]

                # Tất cả events để vẽ chart
                async with conn.execute("""
                    SELECT action, confidence, decision, risk_level, timestamp
                    FROM audit_events
                    ORDER BY timestamp DESC
                    LIMIT 200
                """) as cur:
                    events = [dict(r) for r in await cur.fetchall()]

            return summary, events

        summary, events = asyncio.run(_load_calibration())

        if not summary:
            st.info("Chưa có data — chạy vài review trước.")
        else:
            # Bảng tổng quan
            st.subheader("Avg confidence theo decision")
            df_summary = pd.DataFrame(summary)
            st.dataframe(df_summary, width="stretch")

            # Chart phân phối confidence
            st.subheader("Phân phối confidence theo action")
            df_events = pd.DataFrame(events)
            if not df_events.empty:
                analyze_rows = df_events[df_events["action"] == "analyze"]["confidence"]
                if not analyze_rows.empty:
                    binned = analyze_rows.value_counts(bins=10).sort_index()
                    binned.index = [str(i) for i in binned.index]
                    st.bar_chart(binned)

            # Insight
            st.subheader("Insight")
            approve_rows = [r for r in summary if r["decision"] == "approve"]
            auto_rows = [r for r in summary if r["decision"] == "auto"]
            if approve_rows:
                avg = approve_rows[0]["avg_conf"]
                if avg < 0.65:
                    msg = "under-confident"
                elif avg < 0.80:
                    msg = "well-calibrated"
                else:
                    msg = "over-confident"
                st.metric("Avg confidence khi human approve", f"{avg:.0%}", help=msg)
                st.caption(f"Model có vẻ **{msg}** trên tập data hiện tại.")
            if auto_rows:
                st.metric("Avg confidence khi auto-approve", f"{auto_rows[0]['avg_conf']:.0%}")

            # Risk distribution
            st.subheader("Phân bố risk level")
            risk_counts = df_events["risk_level"].value_counts()
            st.bar_chart(risk_counts)

    except Exception as e:
        st.error(f"Lỗi load data: {e}")


# ─── Tab 3: Time-travel ────────────────────────────────────────────────────
with tab_timetravel:
    st.header("Time-travel")
    st.caption("Chọn một checkpoint cũ của session bất kỳ, resume với câu trả lời khác và so sánh kết quả.")

    tt_thread_id = st.text_input(
        "Thread ID", placeholder="paste thread_id từ sidebar hoặc audit replay",
        key="tt_thread_id",
    )

    async def _get_state_history(thread_id: str):
        async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
            await cp.setup()
            app = build_graph(cp)
            cfg = {"configurable": {"thread_id": thread_id}}
            snapshots = []
            async for snap in app.aget_state_history(cfg):
                snapshots.append(snap)
            return snapshots

    if tt_thread_id:
        try:
            snapshots = asyncio.run(_get_state_history(tt_thread_id))
            if not snapshots:
                st.warning("Không tìm thấy checkpoint cho thread này.")
            else:
                st.success(f"Tìm thấy {len(snapshots)} checkpoint(s).")

                # Hiện bảng các checkpoint
                rows = []
                for i, snap in enumerate(snapshots):
                    vals = snap.values
                    rows.append({
                        "index": i,
                        "next": str(snap.next),
                        "decision": vals.get("decision", "-"),
                        "final_action": vals.get("final_action", "-"),
                        "confidence": f"{vals['analysis'].confidence:.0%}" if vals.get("analysis") else "-",
                    })

                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                # Chọn checkpoint để resume
                st.subheader("Resume từ checkpoint")
                snap_idx = st.number_input(
                    "Chọn index checkpoint (0 = mới nhất)", min_value=0,
                    max_value=max(0, len(snapshots) - 1), value=0, step=1,
                )
                selected_snap = snapshots[int(snap_idx)]
                st.json({
                    "next": str(selected_snap.next),
                    "decision": selected_snap.values.get("decision", "-"),
                    "human_choice": selected_snap.values.get("human_choice", "-"),
                })

                # Nếu checkpoint đang chờ interrupt → cho phép resume với câu trả lời khác
                if selected_snap.next and selected_snap.next != ():
                    st.info(f"Checkpoint này đang chờ node: `{selected_snap.next}` — có thể resume với câu trả lời khác.")
                    tt_choice = st.selectbox("Chọn lại quyết định", ["approve", "reject", "edit"])
                    tt_feedback = st.text_input("Feedback mới (tùy chọn)", key="tt_feedback")

                    if st.button("Resume từ checkpoint này", type="primary"):
                        new_thread_id = str(uuid.uuid4())

                        async def _resume_from_snapshot(snap, choice, feedback, new_tid):
                            async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
                                await cp.setup()
                                app = build_graph(cp)
                                # Copy state vào thread mới
                                new_cfg = {"configurable": {"thread_id": new_tid}}
                                await app.aupdate_state(new_cfg, snap.values, as_node=next(iter(snap.next)) if snap.next else None)
                                # Resume
                                result = await app.ainvoke(
                                    Command(resume={"choice": choice, "feedback": feedback}),
                                    new_cfg,
                                )
                                return result

                        with st.spinner("Resuming từ checkpoint..."):
                            try:
                                result = asyncio.run(_resume_from_snapshot(
                                    selected_snap, tt_choice, tt_feedback, new_thread_id,
                                ))
                                action = result.get("final_action", "?")
                                st.success(f"Kết quả mới: **{action}**  (thread mới: `{new_thread_id[:8]}`)")
                                if result.get("analysis"):
                                    st.caption(f"Confidence cuối: {result['analysis'].confidence:.0%}")
                            except Exception as e:
                                st.error(f"Lỗi resume: {e}")
                else:
                    st.info("Checkpoint này đã hoàn thành — không cần resume.")

        except Exception as e:
            st.error(f"Lỗi load history: {e}")
