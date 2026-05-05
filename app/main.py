from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.db import create_db_and_tables
from app.api.routes.intelligence import router as intelligence_router

app = FastAPI(title="DBU AI Policy Assistant")

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

app.include_router(intelligence_router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root() -> FileResponse:
    return FileResponse("app/static/index.html")