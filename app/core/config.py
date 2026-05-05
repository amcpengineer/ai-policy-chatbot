from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg://postgres:changethis@localhost:5432/dbu_ai"

    class Config:
        env_file = ".env"

settings = Settings()