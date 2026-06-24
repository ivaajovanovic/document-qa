from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Document QA API")
app.include_router(router)