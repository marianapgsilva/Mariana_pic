import os
print("API ATIVA A PARTIR DE:", __file__)
from functools import lru_cache
from typing import Any
 
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# CAMINHOS
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELS = os.path.join(PROJECT_DIR, "models")
 
CAMINHO_MODELO   = os.path.join(PASTA_MODELS, "modelo_lstm_curva4_pesoproposta_pesomax.keras")
CAMINHO_SCALER_X = os.path.join(PASTA_MODELS, "scaler_x_lstm_curva4_pesoproposta_pesomax.pkl")
CAMINHO_SCALER_Y = os.path.join(PASTA_MODELS, "scaler_y_lstm_curva4_pesoproposta_pesomax.pkl")
CAMINHO_METADATA = os.path.join(PASTA_MODELS, "metadata_lstm_curva4_pesoproposta_pesomax.pkl")
 
# ============================================================
# CONSTANTES (têm de coincidir com o script de treino)
# ============================================================
MARCOS_INTERFACE = [
    {"nome": "0",   "meses": 0,  "dia_nominal": 0},
    {"nome": "3m",  "meses": 3,  "dia_nominal": 90},
    {"nome": "6m",  "meses": 6,  "dia_nominal": 180},
    {"nome": "12m", "meses": 12, "dia_nominal": 365},
    {"nome": "24m", "meses": 24, "dia_nominal": 730},
    {"nome": "36m", "meses": 36, "dia_nominal": 1095},
    {"nome": "48m", "meses": 48, "dia_nominal": 1460},
    {"nome": "60m", "meses": 60, "dia_nominal": 1825},
]
 
MAX_TIMESTEPS    = 9
NUM_SEQ_FEATURES = 13
 
# ============================================================
# APP
# ============================================================
app = FastAPI(title="BariCurve API")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ============================================================
# SCHEMAS
# ============================================================
class WeightEntry(BaseModel):
    data: str | None = None
    peso: float
    dia: float | None = None
    mes: float | None = None
 
 
class PatientData(BaseModel):
    sexo: int
    idade: int
    dislipidemia: int
    altura: float
    peso_inicial: float
    fumador: int | None = None
    peso_max_kg: float | None = None
    pesos: list[WeightEntry] = []
 
 
# ============================================================
# CARREGAMENTO DO MODELO (só uma vez ao arrancar)
# ============================================================
@lru_cache(maxsize=1)
def load_model_bundle():
    from tensorflow.keras.models import load_model
    modelo   = load_model(CAMINHO_MODELO, compile=False)
    print("MODELO CARREGADO COM SUCESSO")
    scaler_x = joblib.load(CAMINHO_SCALER_X)
    scaler_y = joblib.load(CAMINHO_SCALER_Y)
    metadata = joblib.load(CAMINHO_METADATA)
    return modelo, scaler_x, scaler_y, metadata
 
 
# ============================================================
# LÓGICA DE PREVISÃO (replica o script de treino)
# ============================================================
def calculate_bmi(weight: float, height: float) -> float:
    return weight / (height ** 2)
 
 
def construir_sequencia(static_values: dict, historico: list, dia_alvo: float) -> np.ndarray:
    historico_valido = sorted(
        [m for m in historico if m["dia"] < dia_alvo],
        key=lambda m: m["dia"]
    )[-MAX_TIMESTEPS:]
 
    seq = np.zeros((MAX_TIMESTEPS, NUM_SEQ_FEATURES), dtype=float)
 
    for t, m in enumerate(historico_valido):
        seq[t] = [
            static_values["genero_num"],
            static_values["idade"],
            static_values["dislipidemia"],
            static_values["peso_max_kg"],
            m["dia"],
            m["imc"],
            dia_alvo - m["dia"],       # delta_dias_ate_alvo
            m.get("dia_estimado", 0),
            dia_alvo,
            0.0,                       # dia_alvo_estimado
            m.get("peso_minimo", 0),
            m.get("peso_proposta", 0),
            1.0,                       # medicao_valida
        ]
    return seq
 
 
def prever_imc(modelo, scaler_x, scaler_y, static_values, historico, dia_alvo) -> float:
    seq = construir_sequencia(static_values, historico, dia_alvo)
    X = seq.reshape(1, MAX_TIMESTEPS, NUM_SEQ_FEATURES)
    X_flat = X.reshape(1, -1)
    X_sc = scaler_x.transform(X_flat).reshape(1, MAX_TIMESTEPS, NUM_SEQ_FEATURES)
    pred_sc = modelo.predict(X_sc, verbose=0)
    return float(scaler_y.inverse_transform(pred_sc)[0][0])
 
 
def prever_trajetoria(modelo, scaler_x, scaler_y, static_values, medicoes_iniciais, altura) -> list:
    historico = sorted(medicoes_iniciais, key=lambda m: m["dia"])
    resultados = []
 
    # Marco 0 — valor real
    imc_inicial = next((m["imc"] for m in historico if m["dia"] == 0), None)
    if imc_inicial is not None:
        resultados.append({
            "mes":    0,
            "dia":    0,
            "imc":    round(imc_inicial, 2),
            "peso":   round(imc_inicial * altura ** 2, 1),
            "origem": "real",
        })
 
    for marco in MARCOS_INTERFACE[1:]:
        dia_alvo = float(marco["dia_nominal"])
 
        medicao_real = next(
            (m for m in historico if abs(m["dia"] - dia_alvo) <= 0.5), None
        )
 
        if medicao_real is not None:
            imc_previsto = float(medicao_real["imc"])
            origem = "real"
        else:
            imc_previsto = prever_imc(modelo, scaler_x, scaler_y, static_values, historico, dia_alvo)
            origem = "previsto"
            historico.append({
                "nome": marco["nome"],
                "dia": dia_alvo,
                "imc": imc_previsto,
                "dia_estimado": 1.0,
                "peso_minimo": 0.0,
                "peso_proposta": 0.0,
            })
            historico.sort(key=lambda m: m["dia"])
 
        resultados.append({
            "mes":    marco["meses"],
            "dia":    int(dia_alvo),
            "imc":    round(imc_previsto, 2),
            "peso":   round(imc_previsto * altura ** 2, 1),
            "origem": origem,
        })
 
    return resultados
 
 
def entry_day(entry: WeightEntry) -> float | None:
    if entry.dia is not None:
        return float(entry.dia)
    if entry.mes is not None:
        return float(entry.mes) * 30.4375
    return None
 
 
def build_initial_measurements(data: PatientData) -> list[dict[str, Any]]:
    measurements = [{
        "nome": "0",
        "dia": 0.0,
        "imc": calculate_bmi(data.peso_inicial, data.altura),
        "dia_estimado": 0.0,
        "peso_minimo": 0.0,
        "peso_proposta": 0.0,
    }]
 
    for i, entry in enumerate(data.pesos):
        day = entry_day(entry)
        if day is None or day <= 0:
            continue
        measurements.append({
            "nome": f"registo_{i+1}",
            "dia": day,
            "imc": calculate_bmi(entry.peso, data.altura),
            "dia_estimado": 0.0,
            "peso_minimo": 0.0,
            "peso_proposta": 0.0,
        })
 
    return sorted(measurements, key=lambda m: m["dia"])
 
 
# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def home():
    return {"message": "BariCurve API com modelo LSTM real"}
 
 
@app.get("/health")
def health():
    try:
        modelo, _, _, metadata = load_model_bundle()
        return {"status": "ok", "modelo": "carregado", "mae_teste": metadata.get("mae_teste")}
    except Exception as e:
        return {"status": "erro", "detalhe": str(e)}
 
 
@app.post("/predict")
def predict(data: PatientData):
    try:
        modelo, scaler_x, scaler_y, _ = load_model_bundle()
 
        all_weights = [data.peso_inicial] + [e.peso for e in data.pesos]
        static_values = {
            "genero_num":   float(data.sexo),
            "idade":        float(data.idade),
            "dislipidemia": float(data.dislipidemia),
            "peso_max_kg":  float(data.peso_max_kg or max(all_weights)),
        }
 
        measurements = build_initial_measurements(data)
        evolucao = prever_trajetoria(modelo, scaler_x, scaler_y, static_values, measurements, data.altura)
 
        return {
            "paciente": data.dict(),
            "evolucao": evolucao,
        }
 
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
 