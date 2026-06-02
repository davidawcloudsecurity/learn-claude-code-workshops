# Bedrock-compatible replacement for the chat_panel in provided.py.
# This version uses local session state instead of server-side session APIs.
"""
Chat panel adapted for Bedrock-based agent. No server-side session listing —
sessions live in st.session_state only.
"""
import json
from pathlib import Path

import streamlit as st

# Re-export everything from provided that the rest of the app needs
from provided import DATA, SYSTEM, TOOLS, metrics, deploys, diff  # noqa: F401


def _offline(fn: str):
    st.caption(f"agent offline — implement `{fn}()` in `agent_bedrock.py`")
    st.chat_input("ask…", disabled=True, key=f"off_{fn}")


def _text(content) -> str:
    if not content:
        return ""
    if isinstance(content, list):
        return "".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")
    return str(content)


def chat_panel():
    import agent_bedrock as agent

    st.markdown("##### SRE AGENT")

    try:
        agent_id = agent.setup_agent()
    except NotImplementedError:
        return _offline("setup_agent")
    st.caption(f"agent · `{agent_id}`")

    try:
        env_id = agent.setup_environment()
    except NotImplementedError:
        return _offline("setup_environment")
    st.caption(f"env · `{env_id}`")

    try:
        log_id = agent.upload_log()
    except NotImplementedError:
        return _offline("upload_log")
    st.caption(f"log · `{log_id}`")

    # ── Session management (local) ───────────────────────────────────────
    sid = st.session_state.get("sid")

    # Get list of local sessions
    sessions = list((st.session_state.get("bedrock_sessions") or {}).keys())

    pick_col, new_col, del_col = st.columns([6, 1, 1])

    if sessions:
        if sid and sid not in sessions:
            sid = sessions[0]
            st.session_state.sid = sid

        def _on_pick():
            st.session_state.sid = st.session_state.session_picker

        if sid:
            st.session_state.session_picker = sid

        pick_col.selectbox(
            "session",
            sessions,
            format_func=lambda v: f"{v[-8:]}",
            disabled=not sessions,
            label_visibility="collapsed",
            key="session_picker",
            on_change=_on_pick,
        )
    else:
        pick_col.selectbox(
            "session", ["(none)"], disabled=True, label_visibility="collapsed"
        )

    if new_col.button("", icon=":material/add:", help="new session", use_container_width=True):
        try:
            new_sid = agent.start_session(agent_id, env_id, log_id)
        except NotImplementedError:
            st.toast("implement `start_session()` in `agent_bedrock.py`")
        else:
            st.session_state.sid = new_sid
            st.session_state.hist = []
            st.rerun()

    if del_col.button(
        "", icon=":material/delete:", help="delete session",
        use_container_width=True, disabled=not sid
    ):
        try:
            agent.delete_session(st.session_state.sid)
        except NotImplementedError:
            st.toast("implement `delete_session()` in `agent_bedrock.py`")
        else:
            if "hist" in st.session_state:
                del st.session_state["hist"]
            del st.session_state["sid"]
            st.rerun()

    sid = st.session_state.get("sid")
    if not sid:
        st.caption("no sessions — click **+** to start one")
        st.chat_input("ask…", disabled=True, key="off_nosession")
        return

    st.caption(f"`{sid}` — local session (in-memory)")

    # Initialize history
    if "hist" not in st.session_state:
        st.session_state.hist = []

    chat = st.container(height=400, border=False)
    with chat:
        for role, text in st.session_state.hist:
            with st.chat_message(role):
                st.markdown(text)

    if q := st.chat_input("ask the agent…"):
        st.session_state.hist.append(("user", q))
        with chat:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                text_ph = st.empty()
                buf = ""
                tool_boxes: dict[str, object] = {}
                try:
                    for ev in agent.stream_reply(st.session_state.sid, q):
                        if ev.type == "agent.message":
                            buf += _text(ev.content)
                            text_ph.markdown(buf)
                        elif ev.type == "agent.custom_tool_use":
                            box = st.status(f"local · {ev.name}", state="running")
                            args = json.dumps(ev.input)
                            box.caption("args")
                            box.code(args if args != "{}" else "(none)", language="json")
                            tool_boxes[ev.id] = box
                        elif ev.type == "user.custom_tool_result":
                            box = tool_boxes.pop(ev.custom_tool_use_id, None)
                            if box:
                                box.caption("result")
                                box.code(
                                    _text(ev.content)[:1500] or "(empty)",
                                    language="text",
                                )
                                box.update(state="complete")
                        elif ev.type == "session.status_idle":
                            for b in tool_boxes.values():
                                b.update(state="complete")
                            break
                except NotImplementedError:
                    st.warning("implement `stream_reply()` / `handle_tool()` in `agent_bedrock.py`")
                    return
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
        st.session_state.hist.append(("assistant", buf))
