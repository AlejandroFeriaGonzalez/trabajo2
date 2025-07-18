from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import random
from model_service import initialize_model, predict_credit_risk
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the model when the application starts
    if initialize_model("./models"):
        print("Modelo inicializado correctamente")
    else:
        print("Error al inicializar el modelo")
    yield
    # Cleanup can be done here if needed

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    with open(os.path.join("static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(data: dict):
    return predict_credit_risk(data)