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


@app.post("/analyze")
async def analyze_risk(
    pct_principal_paid: float = Form(...),
    total_rec_prncp: float = Form(...),
    last_pymnt_amnt: float = Form(...),
    recoveries: float = Form(...),
    total_pymnt: float = Form(...),
    total_pymnt_inv: float = Form(...),
    out_prncp: float = Form(...),
    out_prncp_inv: float = Form(...),
    int_rate: float = Form(...),
    pct_term_paid: float = Form(...),
):
    """Endpoint to analyze credit risk from form data"""
    input_data = {
        "pct_principal_paid": pct_principal_paid,
        "total_rec_prncp": total_rec_prncp,
        "last_pymnt_amnt": last_pymnt_amnt,
        "recoveries": recoveries,
        "total_pymnt": total_pymnt,
        "total_pymnt_inv": total_pymnt_inv,
        "out_prncp": out_prncp,
        "out_prncp_inv": out_prncp_inv,
        "int_rate": int_rate,
        "pct_term_paid": pct_term_paid,
    }

    try:
        result = predict_credit_risk(input_data)

        # Return HTML for HTMX to insert
        risk_color = (
            "text-green-400"
            if result["default_probability"] < 0.25
            else (
                "text-yellow-400"
                if result["default_probability"] < 0.5
                else "text-red-400"
            )
        )

        html_response = f"""
        <div class="space-y-4">
            <h3 class="text-lg font-semibold text-gray-100">Analysis Results</h3>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="bg-slate-700 p-4 rounded-lg">
                    <h4 class="font-medium text-gray-300 mb-2">Credit Score</h4>
                    <p class="text-2xl font-bold text-sky-400">{result['credit_score']:.0f}</p>
                </div>
                
                <div class="bg-slate-700 p-4 rounded-lg">
                    <h4 class="font-medium text-gray-300 mb-2">Risk Level</h4>
                    <p class="text-xl font-semibold {risk_color}">{result['risk_level']}</p>
                </div>
            </div>
            
            <div class="bg-slate-700 p-4 rounded-lg">
                <h4 class="font-medium text-gray-300 mb-2">Default Probability</h4>
                <div class="flex items-center space-x-2">
                    <div class="flex-1 bg-slate-600 rounded-full h-3">
                        <div class="bg-gradient-to-r from-green-500 to-red-500 h-3 rounded-full" 
                             style="width: {result['default_probability'] * 100:.1f}%"></div>
                    </div>
                    <span class="text-gray-200 font-medium">{result['default_probability']:.1%}</span>
                </div>
            </div>
            
            <div class="bg-slate-700 p-4 rounded-lg">
                <h4 class="font-medium text-gray-300 mb-2">Recommendation</h4>
                <p class="text-gray-200">{result['recommendation']}</p>
            </div>
        </div>
        """

        return HTMLResponse(content=html_response)

    except Exception as e:
        error_html = f"""
        <div class="bg-red-600 bg-opacity-20 border border-red-500 rounded-lg p-4">
            <h3 class="text-lg font-semibold text-red-400 mb-2">Error</h3>
            <p class="text-red-300">An error occurred while analyzing your credit risk: {str(e)}</p>
        </div>
        """
        return HTMLResponse(content=error_html)
