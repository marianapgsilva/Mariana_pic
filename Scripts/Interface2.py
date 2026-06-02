import os
import sys
from datetime import date, timedelta

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from LSTM_Curva import (
    MARCOS_INTERFACE,
    calcular_imc,
    carregar_modelo_temporal,
    prever_trajetoria_temporal,
)

# Configuracao da pagina
st.set_page_config(page_title="BariCurve - Simulador Clinico", layout="wide")

PROJETO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PASTA_MODELS = os.path.join(PROJETO_ROOT, "models")

MEDICOES_REAIS_CONFIG = [
    {
        "nome": "3m_real",
        "label": "3 meses",
        "checkbox": "Inserir peso real medido aos 3 meses?",
        "data_label": "Data em que o peso dos 3 meses foi medido",
        "dia_nominal": 90,
        "peso_default_factor": 0.85,
    },
    {
        "nome": "6m_real",
        "label": "6 meses",
        "checkbox": "Inserir peso real medido aos 6 meses?",
        "data_label": "Data em que o peso dos 6 meses foi medido",
        "dia_nominal": 180,
        "peso_default_factor": 0.75,
    },
    {
        "nome": "12m_real",
        "label": "12 meses",
        "checkbox": "Inserir peso real medido aos 12 meses?",
        "data_label": "Data em que o peso dos 12 meses foi medido",
        "dia_nominal": 365,
        "peso_default_factor": 0.68,
    },
    {
        "nome": "24m_real",
        "label": "2 anos",
        "checkbox": "Inserir peso real medido aos 2 anos?",
        "data_label": "Data em que o peso dos 2 anos foi medido",
        "dia_nominal": 730,
        "peso_default_factor": 0.65,
    },
    {
        "nome": "36m_real",
        "label": "3 anos",
        "checkbox": "Inserir peso real medido aos 3 anos?",
        "data_label": "Data em que o peso dos 3 anos foi medido",
        "dia_nominal": 1095,
        "peso_default_factor": 0.65,
    },
    {
        "nome": "48m_real",
        "label": "4 anos",
        "checkbox": "Inserir peso real medido aos 4 anos?",
        "data_label": "Data em que o peso dos 4 anos foi medido",
        "dia_nominal": 1460,
        "peso_default_factor": 0.66,
    },
]


def peso_default(peso_inicial, factor):
    return float(np.clip(round(peso_inicial * factor, 1), 30.0, 250.0))

st.title("BariCurve: Predicao de Trajetoria Pos-Bariatrica")
st.markdown(
    "Este simulador usa uma LSTM temporal: cada medicao entra como IMC associado "
    "ao dia real desde a cirurgia. A projecao clinica e feita ate aos 5 anos."
)

# ============================================================
# SIDEBAR - INPUTS DO MEDICO
# ============================================================
with st.sidebar:
    st.header("Dados do Doente")
    genero = st.radio("Genero", options=["Feminino", "Masculino"], index=0)
    genero_num = 0 if genero == "Feminino" else 1

    idade = st.slider("Idade (anos)", 18, 85, 40)
    dislipidemia = st.selectbox(
        "Dislipidemia?",
        options=[0, 1],
        format_func=lambda x: "Sim" if x == 1 else "Nao",
    )

    st.divider()
    st.header("Cirurgia")
    data_cirurgia = st.date_input("Data em que foi feita a cirurgia", value=date.today())
    peso_0 = st.number_input(
        "Peso no dia da cirurgia (kg)",
        min_value=35.0,
        max_value=260.0,
        value=120.0,
        step=0.1,
    )
    altura = st.number_input(
        "Altura (m)",
        min_value=1.20,
        max_value=2.20,
        value=1.65,
        step=0.01,
        format="%.2f",
    )

    imc_0 = calcular_imc(peso_0, altura)
    st.caption(f"IMC inicial calculado: {imc_0:.2f} kg/m2")

    st.divider()
    st.header("Medicoes reais pos-operatorias")
    medicoes_reais_extra = []

    for config in MEDICOES_REAIS_CONFIG:
        usa_medicao = st.checkbox(config["checkbox"], key=f"usa_{config['nome']}")

        if usa_medicao:
            data_medicao = st.date_input(
                config["data_label"],
                value=data_cirurgia + timedelta(days=config["dia_nominal"]),
                key=f"data_{config['nome']}",
            )
            peso_real = st.number_input(
                f"Peso medido aos {config['label']} (kg)",
                min_value=30.0,
                max_value=250.0,
                value=peso_default(peso_0, config["peso_default_factor"]),
                step=0.1,
                key=f"peso_{config['nome']}",
            )

            dia_real = (data_medicao - data_cirurgia).days
            imc_real = calcular_imc(peso_real, altura)
            medicoes_reais_extra.append(
                {
                    "nome": config["nome"],
                    "label": config["label"],
                    "dia": float(dia_real),
                    "imc": float(imc_real),
                    "peso": float(peso_real),
                    "dia_estimado": 0.0,
                    "origem": "real",
                }
            )

            st.caption(f"Medicao real: D{dia_real} | IMC calculado: {imc_real:.2f} kg/m2")


# ============================================================
# LOGICA DE PREVISAO
# ============================================================
if st.button("Gerar Projecao Clinica", type="primary"):
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    if not np.isfinite(imc_0):
        st.error("Verifica o peso e a altura. Nao foi possivel calcular o IMC inicial.")
        st.stop()

    medicoes_invalidas = [m for m in medicoes_reais_extra if m["dia"] <= 0]
    if medicoes_invalidas:
        labels_invalidas = ", ".join(m["label"] for m in medicoes_invalidas)
        st.error(
            "As datas das medicoes reais tem de ser posteriores a data da cirurgia: "
            f"{labels_invalidas}."
        )
        st.stop()

    static_values = {
        "genero_num": float(genero_num),
        "idade": float(idade),
        "dislipidemia": float(dislipidemia),
    }

    medicoes_iniciais = [
        {
            "nome": "0",
            "dia": 0.0,
            "imc": float(imc_0),
            "dia_estimado": 0.0,
            "origem": "real",
        }
    ]

    pontos_reais = [
        {
            "descricao": "Cirurgia",
            "dia": 0.0,
            "mes": 0.0,
            "imc": float(imc_0),
            "peso": float(peso_0),
        }
    ]

    for medicao in medicoes_reais_extra:
        medicoes_iniciais.append(
            {
                "nome": medicao["nome"],
                "dia": float(medicao["dia"]),
                "imc": float(medicao["imc"]),
                "dia_estimado": 0.0,
                "origem": "real",
            }
        )
        pontos_reais.append(
            {
                "descricao": f"Peso {medicao['label']} medido no D{int(round(medicao['dia']))}",
                "dia": float(medicao["dia"]),
                "mes": float(medicao["dia"]) / 30.4375,
                "imc": float(medicao["imc"]),
                "peso": float(medicao["peso"]),
            }
        )

    with st.spinner("A processar trajetoria temporal..."):
        try:
            modelo, sc_x, sc_y, metadata = carregar_modelo_temporal()
            resultados = prever_trajetoria_temporal(
                modelo, sc_x, sc_y, static_values, medicoes_iniciais
            )
        except Exception as e:
            st.error(f"Erro na execucao: {e}")
            st.stop()

    st.success("Trajetoria calculada com sucesso.")

    tabela = pd.DataFrame(
        [
            {
                "Marco": r["nome"],
                "Dia alvo": int(round(r["dia"])),
                "Mes aproximado": r["meses"],
                "IMC": round(r["imc"], 2),
                "Peso estimado (kg)": round(r["imc"] * (altura**2), 1),
                "Origem": r["origem"],
            }
            for r in resultados
        ]
    )

    st.dataframe(tabela, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    dias_prev = np.array([r["dia"] for r in resultados], dtype=float)
    meses_prev = dias_prev / 30.4375
    imcs_prev = np.array([r["imc"] for r in resultados], dtype=float)

    ax.plot(
        meses_prev,
        imcs_prev,
        "sr--",
        label="Projecao BariCurve ate 5 anos",
        linewidth=2,
    )
    ax.fill_between(
        meses_prev,
        imcs_prev - 2,
        imcs_prev + 2,
        color="red",
        alpha=0.1,
        label="Margem +/-2 IMC",
    )

    pontos_reais = sorted(pontos_reais, key=lambda item: item["dia"])
    meses_reais = [p["mes"] for p in pontos_reais]
    imcs_reais = [p["imc"] for p in pontos_reais]
    ax.plot(meses_reais, imcs_reais, "ok-", label="Dados reais inseridos", markersize=7)

    for ponto in pontos_reais:
        ax.annotate(
            f"D{int(round(ponto['dia']))}",
            (ponto["mes"], ponto["imc"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    xticks = [m["dia_nominal"] / 30.4375 for m in MARCOS_INTERFACE]
    xticklabels = [m["nome"] for m in MARCOS_INTERFACE]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_title("Projecao de evolucao do IMC", fontsize=14)
    ax.set_xlabel("Tempo apos cirurgia")
    ax.set_ylabel("IMC (kg/m2)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    st.pyplot(fig)

    fig_peso, ax_peso = plt.subplots(figsize=(11, 5.5))

    pesos_prev = imcs_prev * (altura**2)
    pesos_prev_min = (imcs_prev - 2) * (altura**2)
    pesos_prev_max = (imcs_prev + 2) * (altura**2)

    ax_peso.plot(
        meses_prev,
        pesos_prev,
        "sb--",
        label="Projecao BariCurve ate 5 anos",
        linewidth=2,
    )
    ax_peso.fill_between(
        meses_prev,
        pesos_prev_min,
        pesos_prev_max,
        color="blue",
        alpha=0.1,
        label="Margem equivalente a +/-2 IMC",
    )

    pesos_reais = [p["imc"] * (altura**2) for p in pontos_reais]
    ax_peso.plot(
        meses_reais,
        pesos_reais,
        "ok-",
        label="Dados reais inseridos",
        markersize=7,
    )

    for ponto in pontos_reais:
        ax_peso.annotate(
            f"D{int(round(ponto['dia']))}",
            (ponto["mes"], ponto["imc"] * (altura**2)),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    ax_peso.set_xticks(xticks)
    ax_peso.set_xticklabels(xticklabels)
    ax_peso.set_title("Projecao de evolucao do peso", fontsize=14)
    ax_peso.set_xlabel("Tempo apos cirurgia")
    ax_peso.set_ylabel("Peso (kg)")
    ax_peso.grid(True, linestyle=":", alpha=0.6)
    ax_peso.legend()

    st.pyplot(fig_peso)

    if metadata and "mae_teste" in metadata:
        st.caption(f"MAE de teste do modelo temporal: {metadata['mae_teste']:.2f} IMC")
