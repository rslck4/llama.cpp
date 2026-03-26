#!/usr/bin/env python3
"""Format claude stream-json output for real-time terminal display.

Reads newline-delimited JSON from stdin (claude --output-format stream-json --verbose)
and prints a compact, colorized view showing tool calls and text as they happen.
"""
import json
import sys
import os

# ANSI escape codes
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"

TOOL_STYLE = {
    "Read":    f"{DIM}{CYAN}  read{RESET}",
    "Edit":    f"{YELLOW}  edit{RESET}",
    "Write":   f"{YELLOW}  write{RESET}",
    "Bash":    f"{MAGENTA}  bash{RESET}",
    "Glob":    f"{DIM}{CYAN}  glob{RESET}",
    "Grep":    f"{DIM}{CYAN}  grep{RESET}",
    "Agent":   f"{GREEN}  agent{RESET}",
    "Skill":   f"{DIM}  skill{RESET}",
    "WebFetch": f"{DIM}{CYAN}  fetch{RESET}",
    "WebSearch": f"{DIM}{CYAN}  search{RESET}",
}

in_text = False

def tool_summary(name, inp):
    """One-line summary of a tool call."""
    if name == "Read":
        p = inp.get("file_path", "")
        return os.path.basename(p) if p else ""
    if name == "Edit":
        p = inp.get("file_path", "")
        return os.path.basename(p) if p else ""
    if name == "Write":
        p = inp.get("file_path", "")
        return os.path.basename(p) if p else ""
    if name == "Bash":
        cmd = inp.get("command", "")
        # Truncate long commands
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        return cmd
    if name == "Glob":
        return inp.get("pattern", "")
    if name == "Grep":
        return inp.get("pattern", "")
    if name == "Agent":
        return inp.get("description", "")
    if name == "Skill":
        return inp.get("skill", "")
    return ""

def flush_text():
    global in_text
    if in_text:
        sys.stdout.write("\n")
        in_text = False

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        continue

    ev_type = ev.get("type", "")

    if ev_type == "assistant":
        msg = ev.get("message", {})
        content = msg.get("content", [])
        for block in content:
            btype = block.get("type", "")

            if btype == "tool_use":
                flush_text()
                name = block.get("name", "?")
                inp = block.get("input", {})
                label = TOOL_STYLE.get(name, f"  {name}")
                detail = tool_summary(name, inp)
                sys.stdout.write(f"{label} {DIM}{detail}{RESET}\n")
                sys.stdout.flush()

            elif btype == "text":
                text = block.get("text", "")
                if text:
                    in_text = True
                    sys.stdout.write(text)
                    sys.stdout.flush()

            # Skip thinking blocks silently

    elif ev_type == "result":
        flush_text()
        duration = ev.get("duration_ms", 0)
        cost = ev.get("total_cost_usd", 0)
        turns = ev.get("num_turns", 0)
        sys.stdout.write(
            f"\n{DIM}  done  {duration/1000:.0f}s  "
            f"${cost:.4f}  "
            f"{turns} turn{'s' if turns != 1 else ''}{RESET}\n"
        )
        sys.stdout.flush()

flush_text()
