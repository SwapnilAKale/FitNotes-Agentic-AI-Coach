# uvicorn server:app --reload

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import os
import re
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable verbose tool call logging")
args = parser.parse_known_args()[0]
DEBUG = args.debug

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import AgentSession
from src.coordinator import Coordinator, MSG_VERIFY_RESTATE, format_verify_fail_message
from src import checkpoint as _ckpt
from src import settings, wal

DB_PATH = os.environ.get("FITNOTES_DB_PATH", "./data/FitNotes_Backup.fitnotes")
FRONTEND_SERVER = Path(__file__).parent / "frontend" / "server.py"

# Coordinator telemetry visibility: nothing configures Python logging here,
# so only WARNING+ escapes via logging.lastResort — which silently dropped
# the INFO-level [decomposition] lifecycle lines during live checks. Scoped
# fix: give ONLY the coordinator logger an INFO handler; the root logger is
# untouched, so third-party INFO noise stays suppressed. Idempotent across
# --reload re-imports.
_coord_logger = logging.getLogger("src.coordinator")
_coord_logger.setLevel(logging.INFO)
if not _coord_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    _coord_logger.addHandler(_h)

agent_ready: bool = False

# Serializes the replace-DB-file + WAL-replay critical section so two
# concurrent uploads can't interleave file writes and replays.
_upload_lock = asyncio.Lock()


def _get_db_fingerprint(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""

def _get_exercise_names(path: str) -> set:
    try:
        import sqlite3
        conn = sqlite3.connect(path)
        rows = conn.execute("SELECT name FROM exercise").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


_last_db_fingerprint: str = _get_db_fingerprint(DB_PATH)

_baseline_row_count: int = 0
try:
    import sqlite3 as _sqlite3
    _conn = _sqlite3.connect(DB_PATH)
    _baseline_row_count = _conn.execute("SELECT COUNT(*) FROM training_log").fetchone()[0]
    _conn.close()
except Exception:
    _baseline_row_count = 0


def _validate_db(path: str, min_rows: int = 0) -> dict:
    import sqlite3
    results: dict = {"valid": True, "warnings": [], "errors": []}
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row

        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            results["valid"] = False
            results["errors"].append(f"Database integrity check failed: {integrity}")
            conn.close()
            return results

        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        for required in ["training_log", "exercise"]:
            if required not in tables:
                results["valid"] = False
                results["errors"].append(
                    f"Missing required table: '{required}' — this may not be a valid FitNotes backup."
                )

        if not results["valid"]:
            conn.close()
            return results

        row_count = conn.execute("SELECT COUNT(*) FROM training_log").fetchone()[0]
        if row_count == 0:
            results["valid"] = False
            results["errors"].append("Database has no workout logs.")
        elif min_rows > 0 and row_count < min_rows:
            results["warnings"].append(
                f"New database has {row_count} sets but current database has {min_rows}. "
                f"You may be missing {min_rows - row_count} workout sets."
            )

        results["row_count"] = row_count
        conn.close()
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Cannot open database: {str(e)}")
    return results


def _is_article(text: str) -> tuple[bool, str]:
    words = text.split()
    if len(words) < 300:
        return False, f"Document too short ({len(words)} words). Articles must be at least 300 words."

    text_lower = text.lower()

    markers = ["abstract", "introduction", "methods", "results",
               "conclusion", "discussion", "references", "bibliography"]
    found_markers = [m for m in markers if m in text_lower]
    if len(found_markers) < 2:
        return False, (
            f"This doesn't look like an article — found only {len(found_markers)} structural "
            f"marker(s) ({', '.join(found_markers) if found_markers else 'none'}). "
            f"Expected at least 2 of: abstract, introduction, methods, results, "
            f"conclusion, discussion, references."
        )

    fitness_keywords = [
        # Core training — specific
        "exercise", "training load", "workout", "strength training",
        "resistance training", "cardio", "cardiovascular", "endurance training",
        "hypertrophy", "muscle mass", "muscular strength", "physical activity",
        "athletic performance", "sport performance", "exercise performance",
        # Recovery and sleep
        "recovery", "sleep quality", "sleep duration", "overtraining",
        "muscle recovery", "cortisol", "circadian rhythm", "melatonin",
        # Body composition
        "body fat", "lean mass", "body composition", "weight loss",
        "obesity", "bmi", "adipose tissue", "metabolic rate",
        # Nutrition and supplements
        "nutrition", "dietary protein", "protein intake", "carbohydrate intake",
        "caloric intake", "supplementation", "creatine", "caffeine",
        "amino acid", "macronutrient", "micronutrient", "hydration",
        "electrolyte", "pre-workout", "post-workout",
        # Physiology
        "exercise physiology", "hormonal response", "testosterone",
        "growth hormone", "lactate threshold", "vo2 max", "oxygen uptake",
        "heart rate", "blood pressure", "inflammation", "anabolic",
        "oxidative stress", "muscle fiber",
        # Injury and rehab — specific
        "sports injury", "exercise injury", "rehabilitation",
        "range of motion", "tendon", "ligament", "physical therapy",
        "muscle soreness", "delayed onset",
        # Psychology — specific
        "exercise adherence", "exercise motivation", "exercise behavior",
        "mental health fitness", "sport psychology",
    ]
    found_fitness = [k for k in fitness_keywords if k in text_lower]

    if len(found_fitness) < 3:
        return False, (
            f"This document doesn't appear to be fitness or health related. "
            f"Only {len(found_fitness)}/3 required term(s) found "
            f"({', '.join(found_fitness) if found_fitness else 'none'}). "
            f"This knowledge base accepts articles about exercise, training, "
            f"nutrition, recovery, sleep, supplements, physiology, and related topics."
        )

    return True, (
        f"Article accepted ({len(words)} words, "
        f"markers: {', '.join(found_markers)}, "
        f"fitness terms: {', '.join(found_fitness[:5])}{'...' if len(found_fitness) > 5 else ''})."
    )


def _chunk_text(text: str, max_words: int = 200) -> list[str]:
    """
    Section-aware chunking for academic papers.
    Splits on section headers first (Introduction, Methods, Results,
    Discussion, Conclusion, etc.), then word-count chunks within sections.
    Each section becomes at minimum one chunk, keeping logical flow intact.
    """
    import re

    # Common academic paper section headers
    SECTION_HEADERS = re.compile(
        r'\n(?='
        r'Abstract|Introduction|Background|Methods?|Materials?|'
        r'Results?|Discussion|Conclusions?|Limitations?|'
        r'Practical [Aa]pplications?|Data [Aa]vailability|'
        r'Ethics|Funding|Acknowledgm|References?|'
        r'Author [Cc]ontributions?|Conflict|Supplementary'
        r')',
        re.MULTILINE
    )

    # Split into sections
    sections = SECTION_HEADERS.split(text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    for section in sections:
        words = section.split()
        if len(words) <= max_words:
            # Section fits in one chunk — keep it whole
            if len(words) >= 20:
                chunks.append(section)
        else:
            # Section too long — split by paragraphs within section
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            current_chunk: list[str] = []
            current_word_count = 0
            for para in paragraphs:
                para_words = len(para.split())
                if current_word_count + para_words > max_words and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.split()) >= 20:
                        chunks.append(chunk_text)
                    current_chunk = [para]
                    current_word_count = para_words
                else:
                    current_chunk.append(para)
                    current_word_count += para_words
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= 20:
                    chunks.append(chunk_text)

    return chunks


def _ingest_article_sync(filename: str, text: str) -> dict:
    """Embed article chunks and store in user_articles ChromaDB collection."""
    import chromadb
    from src.memory import _get_embed_model

    chunks = _chunk_text(text)
    if not chunks:
        return {"success": False, "message": "No valid text chunks extracted from PDF."}

    model = _get_embed_model()
    embeddings = model.encode(chunks).tolist()

    chroma_path = os.environ.get("CHROMA_DB_PATH", "data/chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        collection = client.get_collection("user_articles")
    except Exception:
        collection = client.create_collection("user_articles")

    try:
        existing = collection.get(where={"filename": filename})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    base_id = re.sub(r'[^a-zA-Z0-9_-]', '_', filename.replace('.pdf', ''))
    ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{
            "filename": filename,
            "chunk_index": i,
            "source_type": "user_article",
        } for i in range(len(chunks))],
    )

    return {
        "success": True,
        "filename": filename,
        "chunks_added": len(chunks),
        "message": f"Added {len(chunks)} chunks from '{filename}' to knowledge base.",
    }


# Tools that actually write to the database (second phase of the staged write pattern).
# Staging tools (log_workout, set_goal, etc.) are allowed through so the MCP server
# can store the staged payload; we only gate on the execute step.
EXECUTE_TOOLS = {
    "execute_staged_workout",
    "execute_staged_goal",
    "execute_staged_goal_update",
    "execute_staged_goal_delete",
    "execute_staged_set_update",
    "execute_staged_set_delete",
}

# Mutable dict avoids `global` keyword inside async functions.
_state: dict = {
    "pending_confirmation": False,
    "confirmation_preview": "",
    "allow_execute": False,   # set True by /confirm so execute_ tools are unblocked
    "staging_preview": "",    # args from the last staging tool call, shown as preview
    # Fix 5: which staged flow is awaiting /confirm. "workout" → the server
    # executes deterministically via session.call_tool; anything else keeps the
    # sibling allow_execute + agent re-prompt path.
    "pending_execute_kind": None,
    "pending_upload_path": None,
    "pending_upload_contents": None,
    # Stage-3 #12: a decomposed turn's merged non-write answer, stashed while
    # the confirm panel is up. /confirm delivers it in the CHAT (prepended to
    # the write outcome) — the panel itself shows only the staged batch.
    "decomposed_answer": "",
}

session: AgentSession | None = None
coordinator: Coordinator | None = None
agent_lock: asyncio.Lock | None = None


async def _confirmation_handler(tool_name: str, arguments: dict) -> bool:
    if tool_name in EXECUTE_TOOLS:
        if _state["allow_execute"]:
            return True
        # Block the execute and signal the HTTP layer to return confirmation_required.
        _state["pending_confirmation"] = True
        _state["confirmation_preview"] = (
            _state["staging_preview"] or f"Confirm: {tool_name.replace('_', ' ')}"
        )
        return False
    # Staging tool — capture its args so the confirmation card can show them.
    # Fix 3: log_workout now stages a multi-exercise day as a batch (N calls before
    # one execute), so ACCUMULATE its previews instead of overwriting — otherwise the
    # confirm card shows only the last exercise. Sibling single-item ops (set_goal,
    # update/delete) keep overwrite. The accumulation is reset per /chat turn
    # (the turn handler sets staging_preview="" before routing), and the confirmation
    # gate ends the turn at execute, so a batch is scoped to one turn — no stale bleed.
    blob = json.dumps(arguments, indent=2)
    if tool_name == "log_workout":
        _state["staging_preview"] = (
            f"{_state['staging_preview']}\n\n{blob}"
            if _state["staging_preview"] else blob
        )
        # Fix 5: the agent stages and STOPS — it no longer calls execute, so the
        # blocked-execute path below can't raise the confirm panel for workouts.
        # Staging itself is now what arms confirmation_required for this turn.
        _state["pending_confirmation"] = True
        _state["confirmation_preview"] = _state["staging_preview"]
        _state["pending_execute_kind"] = "workout"
    else:
        _state["staging_preview"] = blob
    return True


async def _initialize_in_background() -> None:
    global session, agent_ready, coordinator
    try:
        session = AgentSession(DB_PATH, debug=DEBUG)
        session.confirmation_handler = _confirmation_handler
        await session.initialize()
        coordinator = Coordinator(session)
        agent_ready = True
        print("[Server] Agent ready.")
    except Exception as exc:
        print(f"[Server] Initialization failed: {exc}")
        agent_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_lock
    agent_lock = asyncio.Lock()

    # Kick off agent initialization without blocking server startup
    asyncio.create_task(_initialize_in_background())

    frontend_proc = None
    try:
        frontend_proc = await asyncio.create_subprocess_exec(
            sys.executable, str(FRONTEND_SERVER),
            env={**os.environ, "LAUNCHED_BY_MAIN": "1"},
        )
        webbrowser.open("http://localhost:3000")
    except Exception as exc:
        print(f"Warning: could not launch frontend: {exc}", file=sys.stderr)

    yield

    global agent_ready
    agent_ready = False
    if frontend_proc is not None:
        try:
            frontend_proc.terminate()
            await frontend_proc.wait()
        except Exception:
            pass
    if session is not None:
        try:
            await session.close()
        except Exception as exc:
            # close() now tears down in the owner task, so the old cross-task
            # "exit cancel scope in a different task" error should no longer fire.
            # If something still does, surface it instead of hiding it.
            print(f"[Server] Error during agent shutdown: {exc}", file=sys.stderr)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ConfirmRequest(BaseModel):
    confirmed: bool


class SettingsRequest(BaseModel):
    wal_replay_enabled: bool


def _parse_retry_seconds(msg: str) -> int | None:
    msg = msg.lower()
    # google.genai ClientError carries the delay as JSON ('retryDelay': '26s'),
    # not the api_core "try again in 26.5s" phrasing — match it explicitly.
    m = re.search(r"retrydelay['\"]?\s*:\s*['\"]?(\d+\.?\d*)s", msg)
    if m:
        return int(float(m.group(1)))
    patterns = [
        r'in (\d+)h\s*(\d+)m',       # Xh Ym
        r'in (\d+)h',                 # Xh only
        r'in (\d+)m\s*(\d+\.?\d*)s',  # XmYs (supports fractional seconds)
        r'in (\d+)m',                 # Xm only
        r'in (\d+\.?\d*)s',           # Xs only
    ]
    for pattern in patterns:
        m = re.search(pattern, msg)
        if m:
            groups = m.groups()
            if 'h' in pattern and 'm' in pattern:
                return int(groups[0]) * 3600 + int(groups[1]) * 60
            elif 'h' in pattern:
                return int(groups[0]) * 3600
            elif 'm' in pattern and 's' in pattern:
                return int(groups[0]) * 60 + int(float(groups[1]))
            elif 'm' in pattern:
                return int(groups[0]) * 60
            elif 's' in pattern:
                return int(float(groups[0]))
    return None


def _error_response(exc: Exception) -> JSONResponse:
    msg = str(exc)
    if "503" in msg or "UNAVAILABLE" in msg:
        return JSONResponse(content={
            "error": "overload",
            "text": "Gemini is under high demand right now. Wait a few minutes and try again.",
        })
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        retry_seconds = _parse_retry_seconds(msg)
        should_retry = retry_seconds is not None and retry_seconds < 120
        # QuotaInterrupted (checkpoint saved mid-pipeline) carries a status
        # message for the user — never draft content.
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit",
                "retry_after_seconds": retry_seconds,
                "should_retry": should_retry,
                "message": getattr(exc, "user_message", None) or "Rate limit reached.",
                "checkpoint_saved": bool(getattr(exc, "checkpoint_saved", False)),
            },
        )
    return JSONResponse(content={"type": "error", "text": f"An error occurred: {msg}"})


@app.get("/status")
async def status():
    return JSONResponse(content={
        "ready": agent_ready,
        "message": "Initializing fitness coach..." if not agent_ready else "Ready",
    })


@app.get("/history")
async def history():
    return JSONResponse(content={"history": session.chat_history if session else []})


@app.get("/checkpoint-status")
async def checkpoint_status():
    """Debug view of the checkpoint slot — never returns the draft text."""
    from src.checkpoint import load_checkpoint
    cp = load_checkpoint()
    if cp is None:
        return JSONResponse(content={"exists": False})
    draft = cp.get("draft")
    msgs  = cp.get("messages")
    return JSONResponse(content={
        "exists":          True,
        "created":         cp.get("created"),
        "route":           cp.get("route"),
        "question":        cp.get("question"),
        "params":          cp.get("params"),
        "completed_stage": cp.get("completed_stage"),
        "has_draft":       bool(draft),
        "draft_chars":     len(draft) if draft else 0,
        "message_count":   len(msgs) if msgs else 0,
    })


async def _process_turn(message: str) -> JSONResponse:
    """
    Shared turn handler for /chat and /resume — same guards, locking, history
    recording, and response shape. /resume passes a continue-intent message so
    the Coordinator runs its checkpoint-resume path (or the nothing-to-resume
    notice) without duplicating that logic here.
    """
    # A chat turn can reach an execute_* MCP tool and write the DB file that
    # upload+replay is mid-way through replacing. Reject with a clear message
    # instead of letting it surface as a SQLite lock error or silent loss.
    if _upload_lock.locked():
        return JSONResponse(
            status_code=503,
            content={"type": "error",
                     "text": "A database upload is in progress — chat is paused "
                             "until it finishes. Try again in a few seconds."},
        )
    if not agent_ready:
        return JSONResponse(
            status_code=503,
            content={"type": "not_ready", "text": "Still initializing. Please wait a moment."},
        )
    if agent_lock.locked():
        return JSONResponse(
            status_code=429,
            content={"error": "Agent is busy, please wait"},
        )
    async with agent_lock:
        if DEBUG:
            print(f"\n[DEBUG] Question: {message}")
        _state["pending_confirmation"] = False
        _state["allow_execute"] = False
        _state["staging_preview"] = ""
        _state["pending_execute_kind"] = None
        # A fresh turn always starts clean — an abandoned panel's stashed
        # answer must never leak into an unrelated later confirm.
        _state["decomposed_answer"] = ""
        # Clear-on-entry: wipe any staged batch left by a prior turn (reload, abandoned,
        # or cancelled) BEFORE this turn stages anything. _staged_writes lives in the MCP
        # subprocess, so the server clears it deterministically via call_tool. This MUST
        # complete before coordinator.route runs (it fires once, before any log_workout of
        # this turn, so the turn's own batch is never wiped). Defensive: a transient MCP
        # failure must not 500 the turn.
        try:
            await session.call_tool("discard_staged_writes", {})
        except Exception as exc:
            print(f"[server] discard_staged_writes (turn-start) failed: {exc}", file=sys.stderr)
        try:
            result = await coordinator.route(message)
            if DEBUG:
                print(f"[DEBUG] Result: {json.dumps({k: v for k, v in result.items() if k != 'flagged_claims'}, indent=2)}")
        except Exception as exc:
            if DEBUG:
                import traceback
                print(f"[DEBUG] Exception in turn:")
                traceback.print_exc()
            return _error_response(exc)
        # Analytical turns bypass session.answer(), so they are not recorded in
        # session.chat_history automatically.  Mirror the same shape agent.py writes.
        if result.get("route") == "analytical" and session is not None:
            _now = datetime.datetime.now().isoformat()
            session.chat_history.append({"role": "user",      "text": message,                    "timestamp": _now})
            session.chat_history.append({"role": "assistant", "text": result.get("answer", ""),   "timestamp": _now})
        # Write-path restore (resume of a quota-interrupted /log turn): the
        # coordinator restored the checkpointed staged batch into the MCP slot
        # and signals it here — the server arms the confirm panel DIRECTLY (the
        # only non-agent writer of these flags; normally _confirmation_handler
        # sets them during agent tool calls). No agent ran this turn.
        if result.get("restore_staged"):
            _state["pending_confirmation"] = True
            _state["pending_execute_kind"] = "workout"
            if not _state["confirmation_preview"]:
                _state["confirmation_preview"] = "Confirm staged workout"
        if _state["pending_confirmation"]:
            _state["pending_confirmation"] = False
            preview = _state["confirmation_preview"]
            # Observability: the args fallback renders workout-shaped text too,
            # so without a source tag a live run can't tell a working slot-read
            # from a silent fallback — a false-pass. Surfaced in the response.
            preview_source = "args_fallback"
            # Ghost-panel guard outcome: True means the pending flag was armed
            # by a log_workout CALL that staged nothing (a refusal /
            # clarification return) — fall through to the normal answer so the
            # agent's ask reaches the user instead of a panel for no batch.
            ghost_suppressed = False
            # Workout path: the panel gates the write, so it must render the
            # staged SLOT — the exact payload execute will write — formatted
            # deterministically in the MCP subprocess, not the tool args and
            # not the LLM's phrasing (either can diverge from the payload).
            # Defensive: any failure falls back to the args-based preview so
            # the panel never blanks.
            if _state["pending_execute_kind"] == "workout":
                # Slot FIRST: _confirmation_handler arms pending at tool CALL
                # time (a pre-call hook cannot see the outcome), so the slot is
                # the only structural fact that says staging actually happened.
                # PROVEN empty (successful read, empty list) ⇒ ghost — suppress
                # the panel entirely. Read failure / malformed shape is UNKNOWN
                # ⇒ fail toward the panel (execute on an empty slot is a no-op
                # error, but suppressing a real batch would lose a write).
                slot_raw = ""
                slot_list = None
                try:
                    slot_raw = await session.call_tool(
                        "read_staged_workout_slot", {})
                    slot_list = json.loads(slot_raw).get("staged_workouts")
                except Exception as exc:
                    print(f"[server] staged-slot read failed: {exc}",
                          file=sys.stderr)
                if slot_list == []:
                    _state["pending_execute_kind"] = None
                    print("[server] ghost confirmation suppressed — "
                          "log_workout was called but staged nothing",
                          file=sys.stderr)
                    ghost_suppressed = True
            if (_state["pending_execute_kind"] == "workout"
                    and not ghost_suppressed):
                try:
                    slot = json.loads(await session.call_tool(
                        "format_staged_workout_for_confirmation", {}))
                    if slot.get("preview"):
                        preview = slot["preview"]
                        preview_source = "slot"
                    else:
                        # Formatter answered but carried no preview (e.g. an
                        # {"error": ...} for an empty slot) — log it, or this
                        # fallback is indistinguishable from a fired slot-read.
                        print(f"[server] staged-workout preview empty/error: {slot!r}",
                              file=sys.stderr)
                except Exception as exc:
                    print(f"[server] staged-workout preview failed: {exc}",
                          file=sys.stderr)
                # ── Stage-2 verify-at-staging: ONE LLM diff of the assembled
                # /log-flow turns ⇄ the staged slot (+ its deterministic
                # rendering), BEFORE the panel is shown. FAIL suppresses the
                # panel and discards the slot IMMEDIATELY — a stray /confirm
                # must find pending_kind=None and an empty slot, so nothing can
                # reach execute. ERROR (verify machinery failed, incl. no
                # slot-rendered preview) fails OPEN: the panel is itself a
                # human check of the slot rendering. Defensive throughout — a
                # verify-layer crash must not 500 the turn.
                verdict = {"verdict": "ERROR", "reason": "verify unavailable"}
                try:
                    if slot_list is None and not slot_raw:
                        # The slot-first read failed — non-verifiable, never
                        # diff a good batch against an empty string and FAIL.
                        raise RuntimeError("slot read failed — verify skipped")
                    restored = result.get("restored_verify")
                    if (isinstance(restored, dict)
                            and restored.get("verdict") == "PASS"):
                        # Restored case 1: the batch was verified before the
                        # interruption and the verdict rode the checkpoint —
                        # never re-pay a completed LLM call on resume.
                        verdict = restored
                    elif preview_source == "slot":
                        # No [message] fallback: a missing flow thread must
                        # skip-verify (ERROR, fail-open), never diff the slot
                        # against the bare reply text and FAIL a good batch.
                        verdict = await coordinator.verify_log_staging(
                            result.get("log_flow_turns"),
                            slot_raw, preview)
                    else:
                        verdict = {"verdict": "ERROR",
                                   "reason": "preview unavailable — verify skipped"}
                except Exception as exc:
                    print(f"[server] staging verify errored: {exc}",
                          file=sys.stderr)
                    verdict = {"verdict": "ERROR", "reason": str(exc)}
                if verdict.get("verdict") == "FAIL":
                    # Close the execute leak BEFORE replying: kind cleared so
                    # /confirm's workout branch can't fire, slot discarded so
                    # even a direct execute is an empty-slot no-op. Discard
                    # FIRST, then record — discard clears the whole
                    # _staged_writes dict, so the FAIL verdict must be written
                    # after it to survive as the sibling key. The guarded
                    # checkpoint clear closes the resume loop (a restored slot
                    # that FAILs must not be restorable again); it never
                    # touches a non-staged (analytical) checkpoint.
                    _state["pending_execute_kind"] = None
                    try:
                        await session.call_tool("discard_staged_writes", {})
                        await session.call_tool("record_workout_verify", verdict)
                    except Exception as exc:
                        print(f"[server] staging verify FAIL cleanup failed: {exc}",
                              file=sys.stderr)
                    _ckpt.clear_staged_checkpoint()
                    # #13 defense in depth: a FAILed flow is over — a leftover
                    # carry must not consume the user's next message.
                    if coordinator is not None:
                        coordinator._pending_log_carry = False
                    print(f"[server] staging verify FAIL: {verdict.get('reason')}",
                          file=sys.stderr)
                    return JSONResponse(content={
                        "type": "answer",
                        "text": format_verify_fail_message(preview) if preview else MSG_VERIFY_RESTATE,
                    })
                # PASS / ERROR: record the verdict (stage 3's trusted-or-not
                # signal) and proceed to the panel.
                try:
                    await session.call_tool("record_workout_verify", verdict)
                except Exception as exc:
                    print(f"[server] record_workout_verify failed: {exc}",
                          file=sys.stderr)
                # Checkpoint-2 (boundary 2 of the write-path arc): a PASSed
                # batch is checkpointed BEFORE the panel wait, so an abandoned
                # panel / restart / stray 429 resumes by restoring this exact
                # verified slot (keep-until-confirm: /confirm outcomes clear
                # it). On a resume turn `message` is "continue", but the flow
                # turns and original question ride the result dict from the
                # checkpoint — the re-save never stores ["continue"]. Same on
                # a discard-confirm turn, where `message` is 'new': the
                # coordinator's resolved_question carries the stashed question
                # it actually processed, so prefer it over `message`.
                if verdict.get("verdict") == "PASS":
                    try:
                        slot_list = json.loads(slot_raw).get("staged_workouts")
                        if slot_list:
                            _ckpt.save_checkpoint(
                                route="operational",
                                question=(result.get("restored_question")
                                          or result.get("resolved_question")
                                          or message),
                                staged_slot=slot_list,
                                log_flow_turns=(result.get("log_flow_turns")
                                                or [message]),
                                verify_verdict=verdict,
                            )
                    except Exception as exc:
                        print(f"[server] staged checkpoint save failed: {exc}",
                              file=sys.stderr)
            if not ghost_suppressed:
                # Stage-3 #12: the panel is a one-time popup that gates the
                # write — it shows ONLY the staged batch. A decomposed turn's
                # merged non-write answer is stashed here and delivered in
                # the CHAT by /confirm, together with the write outcome, so
                # it survives the panel's dismissal.
                if result.get("decomposed"):
                    _state["decomposed_answer"] = result.get("answer") or ""
                return JSONResponse(content={
                    "type": "confirmation_required",
                    "preview": preview,
                    "preview_source": preview_source,
                })
            # Ghost suppressed: fall through to the normal answer return —
            # the agent's refusal/clarification ask is the turn's real output.
        # #2: the Coordinator already builds a graceful, user-facing answer for
        # BOTH the operational fallback (pipeline failure) AND the
        # DataAgentIntegrityError case. Surface THAT answer — never the raw error
        # string, which can leak internal invariant IDs ("B3: …") or exception
        # text to the user. Mirror cli.py, which always prints result['answer'].
        # Only when there is genuinely no answer do we fall back to a generic
        # message. The real error is logged server-side.
        err = result.get("error")
        if err and err != "max_iterations_reached":
            print(f"[Server] Turn completed with error (logged, not surfaced): {err}")
            if not result.get("answer"):
                return JSONResponse(content={
                    "type": "error",
                    "text": "Something went wrong while answering that. Please try again.",
                })
        return JSONResponse(content={"type": "answer", "text": result.get("answer", ""), "route": result.get("route")})


@app.post("/chat")
async def chat(body: ChatRequest):
    return await _process_turn(body.message)


@app.post("/resume")
async def resume():
    # The "Resume" button calls this. Reuse the Coordinator's continue-intent
    # path: load the slot → _resume, or the nothing-to-resume notice if empty.
    return await _process_turn("continue")


def _with_decomposed_answer(outcome_text: str) -> str:
    """Stage-3 #12: /confirm's chat reply = the stashed merged answer (the
    decomposed turn's non-write parts) + the write outcome. Clears the stash
    in every outcome branch so it can never ride into a later confirm."""
    stashed = _state["decomposed_answer"]
    _state["decomposed_answer"] = ""
    if stashed:
        return f"{stashed}\n\n{outcome_text}"
    return outcome_text


@app.post("/confirm")
async def confirm(body: ConfirmRequest):
    # /confirm is the request that actually executes staged DB writes —
    # it must never run while upload+replay is replacing the DB file.
    if _upload_lock.locked():
        return JSONResponse(
            status_code=503,
            content={"type": "error",
                     "text": "A database upload is in progress — your confirmation "
                             "was not executed. Try again in a few seconds."},
        )
    if not agent_ready:
        return JSONResponse(
            status_code=503,
            content={"type": "not_ready", "text": "Still initializing. Please wait a moment."},
        )
    if agent_lock.locked():
        return JSONResponse(
            status_code=429,
            content={"error": "Agent is busy, please wait"},
        )
    async with agent_lock:
        pending_kind = _state["pending_execute_kind"]
        _state["pending_confirmation"] = False
        _state["pending_execute_kind"] = None
        # Clear-on-cancel: a cancelled batch must be discarded immediately so it can't be
        # carried into a later execute (closes the window before the next /chat turn clears
        # it). The confirm path leaves the staged batch intact for execute. Deterministic
        # via call_tool; defensive so a clear failure doesn't break the cancel response.
        if not body.confirmed:
            try:
                await session.call_tool("discard_staged_writes", {})
            except Exception as exc:
                print(f"[server] discard_staged_writes (cancel) failed: {exc}", file=sys.stderr)
            # A cancelled batch's checkpoint must die with it, or a later
            # "continue" would restore what the user just rejected. Guarded:
            # only a staged_slot checkpoint is cleared, never an unrelated
            # interrupted-question slot.
            _ckpt.clear_staged_checkpoint()
            # #13 defense in depth: cancel ends the flow — no carry survives.
            if coordinator is not None:
                coordinator._pending_log_carry = False
        # Fix 5: a confirmed WORKOUT batch is executed by the SERVER, not the agent
        # — call_tool mirrors the deterministic discard calls above. The tool runs
        # atomic write-and-verify (rollback on mismatch), so its result IS the
        # outcome; the agent is never re-prompted and cannot misfire mid-flow.
        if body.confirmed and pending_kind == "workout":
            try:
                raw = await session.call_tool("execute_staged_workout", {})
                outcome = json.loads(raw)
            except Exception as exc:
                return _error_response(exc)
            if outcome.get("success"):
                session._staged_active = False   # slot committed+popped; keep resume coherent
                # The committed batch's checkpoint must die NOW (guarded) — a
                # later "continue" restoring an already-written batch would be
                # a double write. This is the keep-until-confirm lifecycle's
                # closing clear.
                _ckpt.clear_staged_checkpoint()
                # #13 defense in depth: the write landed — flow over, no carry.
                if coordinator is not None:
                    coordinator._pending_log_carry = False
                return JSONResponse(content={
                    "type": "answer",
                    "text": _with_decomposed_answer(
                        f"✅ {outcome.get('message', 'Workout saved and verified.')}"),
                })
            # Execute failure: the analytical half must not be lost either.
            return JSONResponse(content={
                "type": "error",
                "text": _with_decomposed_answer(
                    f"❌ {outcome.get('message') or outcome.get('error') or 'Workout write failed — nothing was saved.'}"),
            })
        # Sibling staged flows (goal / set edits) keep the agent-driven execute:
        # allow_execute unblocks their execute_* tools and the agent is re-prompted.
        _state["allow_execute"] = body.confirmed
        message = "Yes, confirmed, please execute" if body.confirmed else "Cancel that"
        try:
            # Route directly to session — /confirm is the continuation of an
            # in-flight staged write; routing through the Coordinator could
            # misclassify a bare "yes" as a new analytical or operational query.
            result = await session.answer(message)
        except Exception as exc:
            return _error_response(exc)
        finally:
            _state["allow_execute"] = False
        if result.get("error") and result["error"] != "max_iterations_reached":
            return JSONResponse(content={"type": "error", "text": result["error"]})
        # Cancelled workouts and sibling staged flows exit here — the stashed
        # decomposed answer still belongs in the chat with the outcome text.
        return JSONResponse(content={
            "type": "answer",
            "text": _with_decomposed_answer(result.get("answer", "")),
        })


async def _reinitialize_session():
    global session, agent_ready, coordinator
    agent_ready = False

    if session is not None:
        try:
            await session.close()
        except Exception as e:
            # close() shouldn't raise now (owner-task lifecycle fix). If it does,
            # that's unexpected — log it instead of silently swallowing, but stay
            # resilient (don't re-raise) so the reload still proceeds.
            print(f"[Server] Unexpected error closing old session during reinit: {e}")
        session = None

    try:
        session = AgentSession(DB_PATH, debug=DEBUG)
        session.confirmation_handler = _confirmation_handler
        await session.initialize()
        coordinator = Coordinator(session)
        agent_ready = True
        print("[Server] Agent reinitialized successfully.")
    except Exception as e:
        agent_ready = False
        print(f"[Server] Reinitialization failed: {e}")


@app.post("/reload-db")
async def reload_db(new_exercises: list = None):
    # agent_ready must be in the global statement — without it the
    # assignment below creates a dead local and /chat keeps serving
    # against a session that is about to be torn down.
    global _last_db_fingerprint, agent_ready

    current_fingerprint = _get_db_fingerprint(DB_PATH)
    if current_fingerprint == _last_db_fingerprint and current_fingerprint != "":
        print("[Server] Database unchanged — skipping reload.")
        return JSONResponse(content={"status": "ok", "message": "Database unchanged — no reload needed."})

    agent_ready = False
    asyncio.create_task(_reinitialize_session())
    _last_db_fingerprint = current_fingerprint
    payload = {"status": "reloading", "message": "Reloading database in background."}
    if new_exercises:
        payload["new_exercises"] = new_exercises
    return JSONResponse(content=payload)


async def _maybe_replay_wal() -> dict:
    """
    Replay journaled writes onto the freshly written DB file — unless the
    user turned replay off (settings: wal_replay_enabled). Skipping leaves
    every journal entry pending, so a later upload with replay re-enabled
    still applies them.
    """
    if not settings.get_setting("wal_replay_enabled", True):
        print("[Server] WAL replay skipped — disabled in settings.")
        return {"replayed": 0, "conflicts": 0, "skipped": True}
    wal_result = await asyncio.to_thread(wal.replay_writes, DB_PATH)
    print(f"[Server] WAL replay: {wal_result['replayed']} replayed, "
          f"{wal_result['conflicts']} conflicts.")
    return wal_result


@app.post("/upload")
async def upload_db(file: UploadFile):
    import tempfile
    global _baseline_row_count

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".fitnotes") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    async with _upload_lock:
        try:
            validation = _validate_db(tmp_path, min_rows=_baseline_row_count)

            if not validation["valid"]:
                os.unlink(tmp_path)
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "errors": validation["errors"]},
                )

            if validation.get("warnings"):
                _state["pending_upload_path"] = tmp_path
                _state["pending_upload_contents"] = contents
                return JSONResponse(content={
                    "status": "warning",
                    "warnings": validation["warnings"],
                    "message": "Upload has warnings. Proceed anyway?",
                })

            # agent_lock drains any in-flight chat/confirm turn (which can be
            # mid-DB-write in the MCP subprocess) before the file is replaced.
            # New chat/confirm requests are rejected while _upload_lock is held.
            # Lock order is always _upload_lock -> agent_lock; nothing acquires
            # _upload_lock while holding agent_lock, so this cannot deadlock.
            async with agent_lock:
                old_exercises = _get_exercise_names(DB_PATH)
                os.unlink(tmp_path)
                with open(DB_PATH, "wb") as f:
                    f.write(contents)
                _baseline_row_count = validation.get("row_count", _baseline_row_count)
                new_exercises = sorted(_get_exercise_names(DB_PATH) - old_exercises)

                # The fresh backup just wiped any agent-written rows — replay
                # the journaled writes onto the new file before the agent
                # reinitializes against it (unless the user disabled replay).
                wal_result = await _maybe_replay_wal()

        except Exception as e:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return JSONResponse(status_code=500, content={"status": "error", "errors": [str(e)]})

    reload_response = await reload_db(new_exercises=new_exercises)
    payload = json.loads(reload_response.body)
    payload["wal_replay"] = wal_result
    return JSONResponse(content=payload)


@app.post("/upload/confirm")
async def upload_confirm():
    global _baseline_row_count

    contents = _state.get("pending_upload_contents")
    tmp_path = _state.get("pending_upload_path")

    if not contents:
        return JSONResponse(status_code=400, content={"status": "error", "errors": ["No pending upload."]})

    if tmp_path:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    async with _upload_lock:
        # Same drain-then-replace discipline as /upload (see comment there).
        async with agent_lock:
            with open(DB_PATH, "wb") as f:
                f.write(contents)

            _state["pending_upload_path"] = None
            _state["pending_upload_contents"] = None

            # Same replay gate as /upload — this path also replaces the DB file.
            wal_result = await _maybe_replay_wal()

    reload_response = await reload_db()
    payload = json.loads(reload_response.body)
    payload["wal_replay"] = wal_result
    return JSONResponse(content=payload)


@app.get("/wal-status")
async def wal_status():
    """Current write-ahead log contents — debugging aid for upload replay."""
    records = await asyncio.to_thread(wal.get_records)
    by_status = {"pending": 0, "replayed": 0, "conflict": 0}
    for r in records:
        status = r.get("status", "pending")
        by_status[status] = by_status.get(status, 0) + 1
    return JSONResponse(content={
        "total":     len(records),
        "pending":   by_status["pending"],
        "replayed":  by_status["replayed"],
        "conflicts": by_status["conflict"],
        "records":   records,
    })


@app.get("/settings")
async def get_settings():
    records = await asyncio.to_thread(wal.get_records)
    pending = sum(1 for r in records if r.get("status", "pending") == "pending")
    return JSONResponse(content={
        "wal_replay_enabled": bool(settings.get_setting("wal_replay_enabled", True)),
        "wal_pending": pending,
    })


@app.post("/settings")
async def post_settings(body: SettingsRequest):
    value = bool(body.wal_replay_enabled)
    await asyncio.to_thread(settings.set_setting, "wal_replay_enabled", value)
    return JSONResponse(content={"wal_replay_enabled": value})


@app.post("/wal-wipe")
async def wal_wipe():
    """Archive-then-empty the write-ahead log (user 'clear saved chat logs')."""
    result = await asyncio.to_thread(wal.wipe)
    print(f"[Server] WAL wiped: {result['wiped']} records "
          f"(archive: {result['archive']}).")
    return JSONResponse(content=result)


@app.post("/upload/article")
async def upload_article(file: UploadFile):
    import pypdf
    import tempfile

    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"status": "error",
                     "message": "Only PDF files are supported. Please upload a .pdf file."},
        )

    contents = await file.read()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        reader = pypdf.PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        os.unlink(tmp_path)

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Could not read PDF: {str(e)}"},
        )

    if not text.strip():
        return JSONResponse(
            status_code=400,
            content={"status": "error",
                     "message": "PDF appears to be scanned or image-based — "
                                "no text could be extracted. Only text-based PDFs are supported."},
        )

    is_article, reason = _is_article(text)
    if not is_article:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": reason},
        )

    articles_dir = Path("data/user_articles")
    articles_dir.mkdir(parents=True, exist_ok=True)
    (articles_dir / file.filename).write_bytes(contents)

    result = await asyncio.to_thread(_ingest_article_sync, file.filename, text)

    if result["success"]:
        return JSONResponse(content={
            "status": "ok",
            "message": result["message"],
            "chunks": result["chunks_added"],
        })
    else:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": result["message"]},
        )


@app.get("/articles")
async def list_articles():
    try:
        import chromadb
        from collections import Counter
        chroma_path = os.environ.get("CHROMA_DB_PATH", "data/chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection("user_articles")
        results = collection.get(include=["metadatas"])
        filenames = Counter(m["filename"] for m in results["metadatas"])
        articles = [{"filename": f, "chunks": c} for f, c in filenames.items()]
        return JSONResponse(content={"articles": articles})
    except Exception:
        return JSONResponse(content={"articles": []})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
