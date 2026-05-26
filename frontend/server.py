# python server.py
# Serves the UI at http://localhost:3000

import asyncio
import os
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

FRONTEND_DIR = Path(__file__).parent
DB_PATH = Path(__file__).parent.parent / "data" / "FitNotes_Backup.fitnotes"

app = FastAPI()


@app.on_event("startup")
async def open_browser():
    if os.environ.get("LAUNCHED_BY_MAIN"):
        return  # Main server handles browser open
    await asyncio.sleep(0.5)
    webbrowser.open("http://localhost:3000")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"), media_type="text/html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").endswith(".fitnotes"):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "Only .fitnotes files are accepted"},
        )
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        DB_PATH.write_bytes(await file.read())
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(exc)},
        )

    async def _notify_reload() -> None:
        def _call():
            urllib.request.urlopen(
                urllib.request.Request("http://localhost:8000/reload-db", method="POST"),
                timeout=3,
            )
        try:
            await asyncio.to_thread(_call)
        except Exception:
            pass

    await _notify_reload()
    return JSONResponse(content={"status": "success"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
