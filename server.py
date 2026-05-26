# uvicorn server:app --reload

import argparse
import asyncio
import hashlib
import json
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

DB_PATH = os.environ.get("FITNOTES_DB_PATH", "./data/FitNotes_Backup.fitnotes")
FRONTEND_SERVER = Path(__file__).parent / "frontend" / "server.py"

agent_ready: bool = False


def _get_db_fingerprint(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""

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
    "pending_upload_path": None,
    "pending_upload_contents": None,
}

session: AgentSession | None = None
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
    _state["staging_preview"] = json.dumps(arguments, indent=2)
    return True


async def _initialize_in_background() -> None:
    global session, agent_ready
    try:
        session = AgentSession(DB_PATH, debug=DEBUG)
        session.confirmation_handler = _confirmation_handler
        await session.initialize()
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
        except Exception:
            pass


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


def _parse_retry_seconds(msg: str) -> int | None:
    msg = msg.lower()
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
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit",
                "retry_after_seconds": retry_seconds,
                "should_retry": should_retry,
                "message": "Rate limit reached.",
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


@app.post("/chat")
async def chat(body: ChatRequest):
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
            print(f"\n[DEBUG] Question: {body.message}")
        _state["pending_confirmation"] = False
        _state["allow_execute"] = False
        _state["staging_preview"] = ""
        try:
            result = await session.answer(body.message)
            if DEBUG:
                print(f"[DEBUG] Result: {json.dumps(result, indent=2)}")
        except Exception as exc:
            if DEBUG:
                import traceback
                print(f"[DEBUG] Exception in /chat:")
                traceback.print_exc()
            return _error_response(exc)
        if _state["pending_confirmation"]:
            _state["pending_confirmation"] = False
            return JSONResponse(content={
                "type": "confirmation_required",
                "preview": _state["confirmation_preview"],
            })
        if result.get("error") and result["error"] != "max_iterations_reached":
            return JSONResponse(content={"type": "error", "text": result["error"]})
        return JSONResponse(content={"type": "answer", "text": result.get("answer", "")})


@app.post("/confirm")
async def confirm(body: ConfirmRequest):
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
        _state["allow_execute"] = body.confirmed
        _state["pending_confirmation"] = False
        message = "Yes, confirmed, please execute" if body.confirmed else "Cancel that"
        try:
            result = await session.answer(message)
        except Exception as exc:
            return _error_response(exc)
        finally:
            _state["allow_execute"] = False
        if result.get("error") and result["error"] != "max_iterations_reached":
            return JSONResponse(content={"type": "error", "text": result["error"]})
        return JSONResponse(content={"type": "answer", "text": result.get("answer", "")})


async def _reinitialize_session():
    global session, agent_ready
    agent_ready = False

    if session is not None:
        try:
            await session.close()
        except Exception:
            pass
        session = None

    try:
        session = AgentSession(DB_PATH, debug=DEBUG)
        session.confirmation_handler = _confirmation_handler
        await session.initialize()
        agent_ready = True
        print("[Server] Agent reinitialized successfully.")
    except Exception as e:
        agent_ready = False
        print(f"[Server] Reinitialization failed: {e}")


@app.post("/reload-db")
async def reload_db():
    global _last_db_fingerprint

    current_fingerprint = _get_db_fingerprint(DB_PATH)
    if current_fingerprint == _last_db_fingerprint and current_fingerprint != "":
        print("[Server] Database unchanged — skipping reload.")
        return JSONResponse(content={"status": "ok", "message": "Database unchanged — no reload needed."})

    agent_ready = False
    asyncio.create_task(_reinitialize_session())
    _last_db_fingerprint = current_fingerprint
    return JSONResponse(content={"status": "reloading", "message": "Reloading database in background."})


@app.post("/upload")
async def upload_db(file: UploadFile):
    import tempfile
    global _baseline_row_count

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".fitnotes") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

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

        os.unlink(tmp_path)
        with open(DB_PATH, "wb") as f:
            f.write(contents)
        _baseline_row_count = validation.get("row_count", _baseline_row_count)

    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"status": "error", "errors": [str(e)]})

    return await reload_db()


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

    with open(DB_PATH, "wb") as f:
        f.write(contents)

    _state["pending_upload_path"] = None
    _state["pending_upload_contents"] = None

    return await reload_db()


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
