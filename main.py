from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_form():
    with open(os.path.join("static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    age: int = Form(...),
    income: int = Form(...),
    loan_amount: int = Form(...),
    credit_score: int = Form(...),
):
    # Dummy logic for demonstration
    risk = "High" if credit_score < 600 or income < loan_amount else "Low"
    return f"<h2>Credit Risk Result: {risk}</h2><a href='/'>Back</a>"
