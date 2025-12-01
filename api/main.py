from fastapi import FastAPI

from api.routers import health, sign


app = FastAPI(title="SignBridge API", version="0.1.0")


@app.get("/")
def read_root():
    return {"message": "SignBridge API is running", "docs_url": "/docs"}


app.include_router(health.router)
app.include_router(sign.router)
