from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlmodel import Session
from sqlalchemy import text

from app.db.database import engine
from app.core.config import settings

router = APIRouter(tags=["System"])


@router.get("/version")
def version():
    return {"version": settings.VERSION}


@router.get("/health")
def healthcheck():
    db_status = "ok"
    db_details = None

    try:
        with Session(engine) as session:
            session.exec(text("SELECT 1"))
    except Exception as e:
        db_status = "error"
        db_details = str(e)

    return JSONResponse(content={"database": {"status": db_status, "details": db_details}})
