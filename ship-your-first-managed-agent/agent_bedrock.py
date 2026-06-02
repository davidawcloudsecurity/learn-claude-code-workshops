# Refactored agent.py — uses AWS Bedrock Converse API instead of Anthropic Managed Agents.
# Drop-in replacement: swap `import agent` → `import agent_bedrock as agent` in provided.py
"""
SRE Agent backed by Amazon Bedrock (Converse API).
No Anthropic API key required — uses IAM/instance-profile credentials.
"""
import json
import re
from pathlib import Path

import boto3
import streamlit as st

from provided import DATA, SYSTEM, metrics, deploys, diff

# ── Bedrock client ────────────────────────────────────────────────────────
_REGION = "us-east-1"
_MODEL_ID = "anthropic.claude-sonnet-4-20250514-v1:0"

bedrock = boto3.client("bedrock-runtime", region_name=_REGION)


# ── Tool definitions (Bedrock Converse format) ────────────────────────────
TOOLS_CONVERSE = [
    {
        "toolSpec": {
            "name": "get_metrics",
            "description": "Timeseries for a service+metric over the incident window.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "description": "Service name, e.g. checkout"},
                        "metric": {"type": "string", "description": "Metric name, e.g. p99_latency_ms"},
                    },
                    "required": ["service", "metric"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_recent_deploys",
            "description": "Deploys in the last 6 hours.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_diff",
            "description": "Unified diff for a commit SHA.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "commit": {"type": "string", "description": "Commit SHA (full or short)"},
                    },
                    "required": ["commit"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "search_log",
            "description": "Search the application log (data/app.log) for lines matching a regex pattern. Returns matching lines with line numbers. Use this instead of reading the whole log.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "max_lines": {"type": "integer", "description": "Max lines to return (default 50)"},
                    },
                    "required": ["pattern"],
                }
            },
        }
    },
]


# ── 1. Agent setup (no-op for Bedrock — config is just model + system) ───
@st.cache_resource
def setup_agent() -> str:
    """Returns a synthetic agent ID. With Bedrock there's no server-side agent object."""
    return f"bedrock:{_MODEL_ID}"


# ── 2. Environment (not needed for Bedrock) ──────────────────────────────
@st.cache_resource
def setup_environment() -> str:
    return "local-environment"


# ── 3. Upload log (not needed — we read locally) ─────────────────────────
@st.cache_resource
def upload_log() -> str:
    log_path = DATA / "app.log"
    if log_path.exists():
        return f"local:{log_path}"
    return "log-not-found"


# ── 4. Session (managed in st.session_state) ─────────────────────────────
def start_session(agent_id: str, env_id: str, log_file_id: str) -> str:
    """Initialize a new conversation in session state."""
    import uuid

    sid = f"session-{uuid.uuid4().hex[:8]}"
    if "bedrock_sessions" not in st.session_state:
        st.session_state.bedrock_sessions = {}
    st.session_state.bedrock_sessions[sid] = []  # message history
    return sid


# ── 5. Stream reply (agentic loop with Bedrock Converse) ─────────────────
def stream_reply(session_id: str, user_text: str):
    """
    Agentic loop: send user message, handle tool calls until end_turn.
    Yields event-like objects compatible with the UI in provided.py.
    """
    if "bedrock_sessions" not in st.session_state:
        st.session_state.bedrock_sessions = {}
    if session_id not in st.session_state.bedrock_sessions:
        st.session_state.bedrock_sessions[session_id] = []

    messages = st.session_state.bedrock_sessions[session_id]

    # Add user message
    messages.append({"role": "user", "content": [{"text": user_text}]})

    # Agentic loop
    while True:
        response = bedrock.converse(
            modelId=_MODEL_ID,
            system=[{"text": SYSTEM}],
            messages=messages,
            toolConfig={"tools": TOOLS_CONVERSE},
        )

        stop_reason = response["stopReason"]
        output_message = response["output"]["message"]
        messages.append(output_message)

        # Process content blocks
        for block in output_message["content"]:
            if "text" in block:
                yield _Event("agent.message", content=[_TextBlock(block["text"])])
            elif "toolUse" in block:
                tool = block["toolUse"]
                tool_id = tool["toolUseId"]
                tool_name = tool["name"]
                tool_input = tool.get("input", {})

                # Yield tool use event
                yield _Event(
                    "agent.custom_tool_use",
                    name=tool_name,
                    input=tool_input,
                    id=tool_id,
                )

                # Execute tool locally
                result = handle_tool(tool_name, tool_input)

                # Yield tool result event
                yield _Event(
                    "user.custom_tool_result",
                    custom_tool_use_id=tool_id,
                    content=[_TextBlock(result)],
                )

                # Add tool result to messages for next iteration
                # Collect all tool uses first, then send all results
                # (handled below after the for loop)

        # If stop reason is tool_use, collect all tool results and continue
        if stop_reason == "tool_use":
            tool_results = []
            for block in output_message["content"]:
                if "toolUse" in block:
                    tool = block["toolUse"]
                    result = handle_tool(tool["name"], tool.get("input", {}))
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool["toolUseId"],
                            "content": [{"text": result}],
                        }
                    })
            messages.append({"role": "user", "content": tool_results})
            continue  # loop back for next model turn

        # end_turn or max_tokens — we're done
        yield _Event("session.status_idle", stop_reason=_StopReason("end_turn"))
        break


# ── 6. Local tool handlers ────────────────────────────────────────────────
def handle_tool(name: str, args: dict) -> str:
    if name == "get_metrics":
        service = args.get("service", "")
        metric = args.get("metric", "")
        data = metrics.get(service, {}).get(metric)
        if data is None:
            return json.dumps({"error": f"no data for {service}.{metric}"})
        return json.dumps(data)

    if name == "get_recent_deploys":
        return deploys

    if name == "get_diff":
        commit = args.get("commit", "")
        if commit[:7] in diff:
            return diff
        return f"no diff found for commit {commit}"

    if name == "search_log":
        pattern = args.get("pattern", "")
        max_lines = args.get("max_lines", 50)
        log_path = DATA / "app.log"
        if not log_path.exists():
            return "ERROR: app.log not found"
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"ERROR: invalid regex: {e}"
        matches = []
        with open(log_path, "r") as f:
            for i, line in enumerate(f, 1):
                if regex.search(line):
                    matches.append(f"{i}: {line.rstrip()}")
                    if len(matches) >= max_lines:
                        break
        if not matches:
            return f"No lines matching '{pattern}'"
        return "\n".join(matches)

    return f"unknown tool: {name}"


# ── 7. Delete session ─────────────────────────────────────────────────────
def delete_session(session_id: str) -> None:
    if "bedrock_sessions" in st.session_state:
        st.session_state.bedrock_sessions.pop(session_id, None)


# ── Helper classes to mimic the event objects provided.py expects ─────────
class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _StopReason:
    def __init__(self, type_: str):
        self.type = type_


class _Event:
    def __init__(self, type_: str, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)
