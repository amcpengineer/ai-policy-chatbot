from sqlmodel import create_engine, SQLModel, Session
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)