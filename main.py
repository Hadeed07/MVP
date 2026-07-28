from fastapi import FastAPI
from routers import scan

app = FastAPI()
app.include_router(scan.router)



# uvicorn main:app --reload