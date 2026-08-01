from fastapi import FastAPI
from routers import scan
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.include_router(scan.router)

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