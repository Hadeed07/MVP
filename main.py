from fastapi import FastAPI
from routers import scan
from fastapi.middleware.cors import CORSMiddleware

from database import init_database
from routers import recommendation


app = FastAPI()
init_database()

app.include_router(scan.router)
app.include_router(recommendation.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# uvicorn main:app --reload