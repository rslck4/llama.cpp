#!/usr/bin/env bash
# harness.sh — Task-queue agentic loop with adaptive replanning
# Usage: ./harness.sh [--task T03]       # run specific task
#        ./harness.sh                    # run all incomplete tasks
# Reads TASKS.md for the queue. Each task: generator → evaluator → planner.

set -euo pipefail

# --- Configuration -----------------------------------------------------------
MAX_RETRIES="${MAX_RETRIES:-3}"           # max gen/eval retries per task
MAX_TASK_MINUTES="${MAX_TASK_MINUTES:-9}"  # advisory limit mentioned in prompts
LOG="harness.log"
TASKS="TASKS.md"
HANDOFF="HANDOFF.md"
EVAL_RESULTS="EVAL_RESULTS.md"
CLAUDE_MD="CLAUDE.md"
EVAL_MD="EVAL.md"
MODEL="${CLAUDE_MODEL:-}"
FORMATTER="$(dirname "$0")/tools/stream-fmt.py"

# --- Helpers -----------------------------------------------------------------
ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
    local msg="[$(ts)] $*"
    echo "$msg"
    echo "$msg" >> "$LOG"
}

die() { log "FATAL: $*"; exit 1; }

notify() {
    local msg="$1" sound="$2"
    osascript -e "display notification \"$msg\" with title \"Harness\" sound name \"$sound\"" 2>/dev/null || true
    afplay "/System/Library/Sounds/${sound}.aiff" &>/dev/null &
}

kill_orphans() {
    pkill -f "llama-cli" 2>/dev/null || true
    pkill -f "llama-completion" 2>/dev/null || true
}

claude_run() {
    local prompt="$1"
    shift
    local context_files=("$@")
    local cmd=(claude --print --dangerously-skip-permissions --verbose --output-format stream-json)

    if [[ -n "$MODEL" ]]; then
        cmd+=(--model "$MODEL")
    fi

    # Prepend file contents into prompt
    local full_prompt=""
    for f in "${context_files[@]}"; do
        if [[ -f "$f" ]]; then
            full_prompt+="<file path=\"$f\">"$'\n'
            full_prompt+="$(cat "$f")"$'\n'
            full_prompt+="</file>"$'\n\n'
        fi
    done
    full_prompt+="$prompt"

    # Stream JSON events through formatter for real-time tool call + text display
    "${cmd[@]}" "$full_prompt" < /dev/null 2>&1 | tee -a "$LOG" | python3 -u "$FORMATTER"
    return "${PIPESTATUS[0]}"
}

parse_status() {
    if [[ ! -f "$EVAL_RESULTS" ]]; then
        echo "UNKNOWN"
        return
    fi
    local status
    status=$(grep -m1 '^STATUS:' "$EVAL_RESULTS" 2>/dev/null | sed 's/^STATUS:[[:space:]]*//' | tr -d '[:space:]')
    case "$status" in
        PASS|FAIL|NEEDS_HUMAN) echo "$status" ;;
        *) echo "UNKNOWN" ;;
    esac
}

# --- Task queue helpers ------------------------------------------------------

# Get the next incomplete task: returns "T03 — Description" or empty
next_task() {
    grep -m1 '^⬜\|^🔄' "$TASKS" 2>/dev/null | sed 's/^[⬜🔄] *//'
}

# Extract task ID (e.g., "T03") from a task line
task_id() {
    echo "$1" | grep -oE '^T[0-9]+'
}

# Extract task description (everything after " — ")
task_desc() {
    echo "$1" | sed 's/^T[0-9]* — //'
}

# Mark a task with a new status in TASKS.md
mark_task() {
    local tid="$1" new_status="$2"
    # Replace the status emoji at the start of the line containing this task ID
    sed -i '' "s/^[⬜🔄✅❌] *${tid} /${new_status} ${tid} /" "$TASKS"
}

# --- Validate inputs ---------------------------------------------------------
ONLY_TASK=""
if [[ "${1:-}" == "--task" ]]; then
    ONLY_TASK="${2:?--task requires a task ID (e.g., T03)}"
    shift 2
fi

if [[ ! -f "$CLAUDE_MD" ]]; then
    die "Missing $CLAUDE_MD — run from project root"
fi
if [[ ! -f "$EVAL_MD" ]]; then
    die "Missing $EVAL_MD — create evaluator mandate first"
fi
if [[ ! -f "$TASKS" ]]; then
    die "Missing $TASKS — create task queue first"
fi
if [[ ! -f "$FORMATTER" ]]; then
    die "Missing $FORMATTER — stream formatter not found"
fi

log "=== Harness started (task-queue mode) ==="
log "Tasks file: $TASKS"
[[ -n "$ONLY_TASK" ]] && log "Running only: $ONLY_TASK"

tasks_completed=0

# --- Main task loop ----------------------------------------------------------
while true; do
    kill_orphans

    # Get next task
    if [[ -n "$ONLY_TASK" ]]; then
        task_line=$(grep "$ONLY_TASK" "$TASKS" | head -1 | sed 's/^[⬜🔄✅❌] *//')
        if [[ -z "$task_line" ]]; then
            die "Task $ONLY_TASK not found in $TASKS"
        fi
        # If already done, exit
        if grep -q "^✅.*$ONLY_TASK" "$TASKS"; then
            log "Task $ONLY_TASK already completed"
            break
        fi
    else
        task_line=$(next_task)
    fi

    if [[ -z "$task_line" ]]; then
        log "=== All tasks complete ($tasks_completed this session) ==="
        notify "All tasks complete ($tasks_completed done)" "Glass"
        exit 0
    fi

    tid=$(task_id "$task_line")
    tdesc=$(task_desc "$task_line")
    log "=== Starting task $tid: $tdesc ==="
    mark_task "$tid" "🔄"

    # Reset per-task state
    rm -f "$HANDOFF" "$EVAL_RESULTS"
    retries=0
    last_failure=""
    consecutive_same_fail=0
    task_passed=false

    # --- Generator/Evaluator retry loop for this task ------------------------
    while (( retries < MAX_RETRIES )); do
        retries=$((retries + 1))
        log "--- $tid attempt $retries/$MAX_RETRIES: GENERATOR ---"

        gen_prompt="You are the GENERATOR in an agentic loop. Your current task:

TASK $tid: $tdesc

CONSTRAINTS:
- This task must complete in under 10 minutes. Stay focused.
- Do the work, then write $HANDOFF with:
  1. What you completed
  2. Exact commands to verify your work
  3. Known concerns or open questions
  4. Ready for evaluation: yes/no
- If a previous evaluation failed, $EVAL_RESULTS contains what to fix.

Work autonomously. Do not ask questions — make reasonable decisions and document them."

        gen_context=("$CLAUDE_MD" "$TASKS")
        [[ -f "$EVAL_RESULTS" ]] && gen_context+=("$EVAL_RESULTS")

        if ! claude_run "$gen_prompt" "${gen_context[@]}"; then
            log "WARNING: Generator timed out or failed for $tid"
            kill_orphans
            continue
        fi
        log "Generator $tid attempt $retries complete"
        kill_orphans

        # Check handoff exists
        if [[ ! -f "$HANDOFF" ]]; then
            log "WARNING: Generator did not produce $HANDOFF — retrying"
            continue
        fi

        log "--- $tid attempt $retries/$MAX_RETRIES: EVALUATOR ---"

        eval_prompt="You are the EVALUATOR in an agentic loop. Verify the generator's work on:

TASK $tid: $tdesc

INSTRUCTIONS:
1. Read $HANDOFF to understand what was done
2. Run every verification command listed in $HANDOFF
3. Apply every check from $EVAL_MD
4. Write $EVAL_RESULTS with:
   - Line 1: STATUS: PASS or STATUS: FAIL or STATUS: NEEDS_HUMAN
   - What you tested and how
   - Specific failures with exact evidence
   - What the generator must fix next (if FAIL)
   - Why human review is needed (if NEEDS_HUMAN)

Be strict. Only PASS if the task is genuinely complete."

        eval_context=("$EVAL_MD" "$HANDOFF")
        [[ -f "$CLAUDE_MD" ]] && eval_context+=("$CLAUDE_MD")

        if ! claude_run "$eval_prompt" "${eval_context[@]}"; then
            log "WARNING: Evaluator timed out or failed for $tid"
            kill_orphans
            continue
        fi
        log "Evaluator $tid attempt $retries complete"
        kill_orphans

        # Parse result
        status=$(parse_status)
        log "STATUS: $status"

        case "$status" in
            PASS)
                log "=== $tid PASSED on attempt $retries ==="
                task_passed=true
                break
                ;;
            NEEDS_HUMAN)
                log "=== $tid NEEDS_HUMAN on attempt $retries ==="
                mark_task "$tid" "❌"
                notify "$tid needs human review" "Ping"
                if [[ -n "$ONLY_TASK" ]]; then exit 2; fi
                break
                ;;
            FAIL)
                current_failure=$(grep -A5 '^STATUS:' "$EVAL_RESULTS" 2>/dev/null | tail -4 | head -1 || echo "")
                if [[ "$current_failure" == "$last_failure" && -n "$current_failure" ]]; then
                    consecutive_same_fail=$((consecutive_same_fail + 1))
                else
                    consecutive_same_fail=1
                    last_failure="$current_failure"
                fi

                if (( consecutive_same_fail >= MAX_RETRIES )); then
                    log "=== $tid STUCK: same failure ${MAX_RETRIES}x ==="
                    mark_task "$tid" "❌"
                    notify "$tid stuck after ${MAX_RETRIES} identical failures" "Basso"
                    if [[ -n "$ONLY_TASK" ]]; then exit 2; fi
                    break
                fi

                log "$tid FAIL (attempt $retries/$MAX_RETRIES) — retrying"
                ;;
            UNKNOWN)
                log "WARNING: Could not parse status from $EVAL_RESULTS — treating as FAIL"
                ;;
        esac
    done

    # --- Post-task: mark complete or failed ----------------------------------
    if [[ "$task_passed" == true ]]; then
        mark_task "$tid" "✅"
        tasks_completed=$((tasks_completed + 1))
        notify "$tid passed" "Glass"

        # If running single task, exit now
        if [[ -n "$ONLY_TASK" ]]; then
            log "=== Single task $tid completed ==="
            exit 0
        fi

        # --- PLANNER turn: review findings and replan TASKS.md ---------------
        log "--- $tid: PLANNER ---"

        planner_prompt="You are the PLANNER in an agentic loop. Task $tid just PASSED.

Review what was accomplished and update the task queue.

Read TASKS.md, HANDOFF.md, and EVAL_RESULTS.md. Then:

1. Check if the completed task revealed new work that should be added
2. Check if any pending tasks are now blocked, unblocked, or obsolete
3. Check if task ordering should change based on what was learned
4. Update TASKS.md:
   - Add new tasks with ⬜ status and the next available T-number
   - Mark blocked tasks with ❌ and a note
   - Reorder if dependencies changed
   - Do NOT change ✅ tasks
   - Keep the format: STATUS TASK_ID — Description

Be conservative. Only add tasks that are clearly needed. Do not split tasks unnecessarily."

        planner_context=("$TASKS" "$HANDOFF" "$EVAL_RESULTS")
        [[ -f "$CLAUDE_MD" ]] && planner_context+=("$CLAUDE_MD")

        if ! claude_run "$planner_prompt" "${planner_context[@]}"; then
            log "WARNING: Planner timed out — continuing with current TASKS.md"
        fi
        log "Planner turn complete"
        kill_orphans

        log "--- Updated TASKS.md ---"
        cat "$TASKS" >> "$LOG"
    elif (( retries >= MAX_RETRIES )) && [[ "$task_passed" == false ]]; then
        # Exhausted retries without PASS or explicit NEEDS_HUMAN
        mark_task "$tid" "❌"
        notify "$tid failed after $MAX_RETRIES attempts" "Basso"
        log "=== $tid exhausted $MAX_RETRIES retries ==="
        if [[ -n "$ONLY_TASK" ]]; then exit 1; fi
    fi
done

log "=== Harness finished ($tasks_completed tasks completed) ==="
