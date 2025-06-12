from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Usa la carpeta 'static' como templates (o cambia si mueves el html a otra carpeta)
templates = Jinja2Templates(directory="static")


@app.get("/")
async def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(
    request: Request,
    age: int = Form(...),
    income: int = Form(...),
    loan_amount: int = Form(...),
    credit_score: int = Form(...),
):
    print(f"Received data: Age={age}, Income={income}, Loan Amount={loan_amount}, Credit Score={credit_score}")
    # Dummy logic for demonstration
    risk = "High" if credit_score < 600 or income < loan_amount else "Low"
    result = f"Credit Risk Result: {risk}"
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
