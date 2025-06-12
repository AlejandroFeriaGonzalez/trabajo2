from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import random

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    with open(os.path.join("static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    age: int = Form(...),
    income: int = Form(...),
    loan_amount: int = Form(...),
    credit_score: int = Form(...),
):
    # Generate random score for demonstration
    score = random.randint(1, 100)
    
    # Return only the HTML fragment that will be inserted into the page
    risk_level = "High risk - Approval unlikely" if score < 40 else "Medium risk - Further review needed" if score < 70 else "Low risk - Approval likely"
    color_class = "bg-red-600" if score < 40 else "bg-yellow-400" if score < 70 else "bg-green-600"
    
    return f"""
    <div class="bg-blue-50 p-4 rounded-lg">
      <h3 class="font-bold text-blue-800 text-lg mb-2">Risk Analysis Result</h3>
      <p class="text-blue-700">Score: {score}</p>
      <div class="mt-3 flex justify-center">
        <div class="w-full bg-gray-200 rounded-full h-2.5">
          <div class="{color_class} h-2.5 rounded-full" style="width: {score}%"></div>
        </div>
      </div>
      <p class="text-sm text-gray-600 mt-2 text-center">
        {risk_level}
      </p>
    </div>
    """
