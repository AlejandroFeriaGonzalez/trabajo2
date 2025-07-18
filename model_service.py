import os
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionInput:
    """Estructura de datos para entrada de predicción simplificada"""

    loan_amnt: float = 0.0
    annual_inc: float = 0.0
    int_rate: float = 0.0
    installment: float = 0.0
    dti: float = 0.0
    fico_range_low: float = 0.0
    fico_range_high: float = 0.0
    inq_last_6mths: float = 0.0
    open_acc: float = 0.0
    pub_rec: float = 0.0
    revol_bal: float = 0.0
    revol_util: float = 0.0
    total_acc: float = 0.0


class ModelService:
    """Servicio para carga y predicción del modelo de riesgo crediticio"""

    # Features principales del modelo
    TOP_FEATURES = [
        "pct_principal_paid",
        "total_rec_prncp",
        "last_pymnt_amnt",
        "recoveries",
        "total_pymnt",
        "total_pymnt_inv",
        "out_prncp",
        "out_prncp_inv",
        "int_rate",
        "pct_term_paid",
    ]

    def __init__(self, model_dir: str = "./models"):
        """
        Inicializa el servicio del modelo

        Args:
            model_dir: Directorio donde se encuentran los archivos del modelo
        """
        # Normaliza la ruta para compatibilidad multiplataforma
        self.model_dir = Path(os.path.abspath(os.path.expanduser(model_dir)))
        self.model = None
        self.scaler = None
        self.features = None
        self._is_loaded = False

    def load_model(self) -> bool:
        """
        Carga el modelo y todos los artefactos necesarios

        Returns:
            bool: True si la carga fue exitosa
        """
        try:
            model_path = self.model_dir / "credit_risk_model.h5"
            scaler_path = self.model_dir / "scaler.joblib"
            features_path = self.model_dir / "features.pkl"

            # Verificar que existan los archivos necesarios
            if not model_path.exists():
                logger.error(f"Modelo no encontrado en {model_path}")
                return False

            # Cargar modelo
            self.model = keras.models.load_model(str(model_path))
            logger.info("Modelo cargado exitosamente")

            # Cargar scaler si existe
            if scaler_path.exists():
                self.scaler = joblib.load(str(scaler_path))
                logger.info("Scaler cargado exitosamente")

            # Cargar features si existe
            if features_path.exists():
                self.features = joblib.load(str(features_path))
                logger.info("Features cargadas exitosamente")
            else:
                self.features = self.TOP_FEATURES

            self._is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            return False

    def _ensure_model_loaded(self):
        """Asegura que el modelo esté cargado"""
        if not self._is_loaded:
            if not self.load_model():
                raise RuntimeError("No se pudo cargar el modelo")

    def _prepare_features(self, input_data: Dict) -> np.ndarray:
        """
        Prepara las features para el modelo basándose en datos de entrada simplificados

        Args:
            input_data: Diccionario con datos de entrada

        Returns:
            Array numpy con features preparadas
        """
        # Crear un DataFrame con las features requeridas
        features_dict = {}

        # Para este ejemplo, vamos a simular las features principales
        # En un caso real, estas se calcularían basándose en los datos de entrada

        # Simular features basándose en datos disponibles
        loan_amnt = input_data.get("loan_amnt", 0)
        int_rate = input_data.get("int_rate", 0)

        # Simular features de comportamiento (en producción vendrían de histórico)
        features_dict["pct_principal_paid"] = 0.8  # Simulado
        features_dict["total_rec_prncp"] = loan_amnt * 0.8  # Simulado
        features_dict["last_pymnt_amnt"] = loan_amnt * 0.05  # Simulado
        features_dict["recoveries"] = 0  # Generalmente 0 para nuevos préstamos
        features_dict["total_pymnt"] = loan_amnt * 1.1  # Simulado
        features_dict["total_pymnt_inv"] = loan_amnt * 1.1  # Simulado
        features_dict["out_prncp"] = loan_amnt * 0.2  # Simulado
        features_dict["out_prncp_inv"] = loan_amnt * 0.2  # Simulado
        features_dict["int_rate"] = int_rate
        features_dict["pct_term_paid"] = 0.6  # Simulado

        # Crear DataFrame
        df = pd.DataFrame([features_dict])

        # Asegurar que todas las features estén presentes
        for feature in self.TOP_FEATURES:
            if feature not in df.columns:
                df[feature] = 0.0

        # Seleccionar solo las features necesarias
        X = df[self.TOP_FEATURES].astype("float32")

        # Aplicar escalamiento si existe
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            return X_scaled

        return X.values

    def predict_risk(self, input_data: Dict) -> Dict[str, float]:
        """
        Predice el riesgo de incumplimiento

        Args:
            input_data: Diccionario con datos del solicitante

        Returns:
            Diccionario con probabilidad de incumplimiento y score crediticio
        """
        self._ensure_model_loaded()

        try:
            # Preparar features
            X = self._prepare_features(input_data)

            # Hacer predicción
            probability = self.model.predict(X, verbose=0)[0][0]

            # Convertir a score crediticio (300-850)
            score = self._convert_prob_to_score(probability)

            # Determinar nivel de riesgo
            risk_level = self._get_risk_level(probability)

            return {
                "default_probability": float(probability),
                "credit_score": float(score),
                "risk_level": risk_level,
                "recommendation": self._get_recommendation(probability),
            }

        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            raise

    def _convert_prob_to_score(
        self, probability: float, min_score: int = 300, max_score: int = 850
    ) -> float:
        """
        Convierte probabilidad de incumplimiento a score crediticio

        Args:
            probability: Probabilidad de incumplimiento (0-1)
            min_score: Score mínimo
            max_score: Score máximo

        Returns:
            Score crediticio
        """
        return min_score + (max_score - min_score) * (1 - probability)

    def _get_risk_level(self, probability: float) -> str:
        """
        Determina el nivel de riesgo basado en la probabilidad

        Args:
            probability: Probabilidad de incumplimiento

        Returns:
            Nivel de riesgo como string
        """
        if probability < 0.1:
            return "Muy Bajo"
        elif probability < 0.25:
            return "Bajo"
        elif probability < 0.5:
            return "Medio"
        elif probability < 0.75:
            return "Alto"
        else:
            return "Muy Alto"

    def _get_recommendation(self, probability: float) -> str:
        """
        Proporciona recomendación basada en la probabilidad

        Args:
            probability: Probabilidad de incumplimiento

        Returns:
            Recomendación como string
        """
        if probability < 0.1:
            return "Aprobación recomendada con condiciones estándar"
        elif probability < 0.25:
            return "Aprobación recomendada con seguimiento regular"
        elif probability < 0.5:
            return "Evaluación adicional requerida, considerar garantías"
        elif probability < 0.75:
            return "Rechazo recomendado o condiciones muy estrictas"
        else:
            return "Rechazo fuertemente recomendado"

    def health_check(self) -> Dict[str, bool]:
        """
        Verifica el estado del servicio

        Returns:
            Diccionario con estado de componentes
        """
        return {
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "features_loaded": self.features is not None,
            "service_ready": self._is_loaded,
        }


# Instancia global del servicio
model_service = ModelService()


def get_model_service() -> ModelService:
    """
    Función factory para obtener instancia del servicio del modelo

    Returns:
        Instancia del ModelService
    """
    return model_service


def predict_credit_risk(input_data: Dict) -> Dict[str, float]:
    """
    Función simple para predicción de riesgo crediticio

    Args:
        input_data: Datos del solicitante

    Returns:
        Resultado de la predicción
    """
    service = get_model_service()
    return service.predict_risk(input_data)


def initialize_model(model_dir: str = ".") -> bool:
    """
    Inicializa el modelo al arrancar la aplicación

    Args:
        model_dir: Directorio del modelo

    Returns:
        True si la inicialización fue exitosa
    """
    global model_service
    model_service = ModelService(model_dir)
    return model_service.load_model()


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar modelo
    if initialize_model("./models"):
        print("Modelo inicializado correctamente")

        # Ejemplo de predicción
        test_data = {
            "loan_amnt": 10000,
            "annual_inc": 50000,
            "int_rate": 12.5,
            "installment": 250,
            "dti": 15.2,
            "fico_range_low": 680,
            "fico_range_high": 684,
            "inq_last_6mths": 1,
            "open_acc": 8,
            "pub_rec": 0,
            "revol_bal": 5000,
            "revol_util": 65.2,
            "total_acc": 15,
        }

        result = predict_credit_risk(test_data)
        print("Resultado de predicción:")
        print(f"Probabilidad de incumplimiento: {result['default_probability']:.2%}")
        print(f"Score crediticio: {result['credit_score']:.0f}")
        print(f"Nivel de riesgo: {result['risk_level']}")
        print(f"Recomendación: {result['recommendation']}")

        # Verificar estado del servicio
        health = model_service.health_check()
        print(f"\nEstado del servicio: {health}")
    else:
        print("Error al inicializar el modelo")
