from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import settings

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
frontend_dist_dir = settings.frontend_dist_dir


@app.get("/")
def home():
    if frontend_dist_dir.exists():
        return FileResponse(frontend_dist_dir / "index.html")
    return {
        "message": "Frontend build not found. Run Vite frontend separately (`npm run dev`) or build it (`npm run build`)."
    }


if frontend_dist_dir.exists():
    assets_dir = frontend_dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        candidate = frontend_dist_dir / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(frontend_dist_dir / "index.html")
