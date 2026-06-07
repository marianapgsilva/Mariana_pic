"""
BariCurve - LSTM temporal com peso minimo, peso proposta e peso maximo
======================================================================
Este script treina um unico modelo LSTM capaz de prever o IMC em diferentes
dias alvo. Cada medicao entra como par temporal: dia desde cirurgia + IMC.
A versao 4 acrescenta peso maximo como variavel estatica e peso proposta
cirurgica como medicao temporal pre-operatoria, alem do peso minimo
pos-operatorio da versao 3.

O treino usa medicoes disponiveis ate 84 meses, mas a interface clinica
preve apenas ate 60 meses.
"""

import os
import warnings
import unicodedata

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")

# ============================================================
# CAMINHOS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJETO_ROOT = os.path.join(BASE_DIR, "..")

PASTA_DADOS = os.path.join(PROJETO_ROOT, "Dados")
PASTA_MODELS = os.path.join(PROJETO_ROOT, "models")
CAMINHO_CSV = os.path.join(PASTA_DADOS, "Base_dados_v1.csv")

os.makedirs(PASTA_MODELS, exist_ok=True)

CAMINHO_MODELO_TEMPORAL = os.path.join(
    PASTA_MODELS, "modelo_lstm_curva4_pesoproposta_pesomax.keras"
)
CAMINHO_SCALER_X_TEMPORAL = os.path.join(
    PASTA_MODELS, "scaler_x_lstm_curva4_pesoproposta_pesomax.pkl"
)
CAMINHO_SCALER_Y_TEMPORAL = os.path.join(
    PASTA_MODELS, "scaler_y_lstm_curva4_pesoproposta_pesomax.pkl"
)
CAMINHO_METADATA_TEMPORAL = os.path.join(
    PASTA_MODELS, "metadata_lstm_curva4_pesoproposta_pesomax.pkl"
)

# ============================================================
# DEFINICAO DOS MARCOS
# ============================================================
MARCOS_TREINO = [
    {"nome": "0", "meses": 0, "dia_nominal": 0},
    {"nome": "3m", "meses": 3, "dia_nominal": 90},
    {"nome": "6m", "meses": 6, "dia_nominal": 180},
    {"nome": "12m", "meses": 12, "dia_nominal": 365},
    {"nome": "24m", "meses": 24, "dia_nominal": 730},
    {"nome": "36m", "meses": 36, "dia_nominal": 1095},
    {"nome": "48m", "meses": 48, "dia_nominal": 1460},
    {"nome": "60m", "meses": 60, "dia_nominal": 1825},
    {"nome": "72m", "meses": 72, "dia_nominal": 2190},
    {"nome": "84m", "meses": 84, "dia_nominal": 2555},
]

MARCOS_INTERFACE = [m for m in MARCOS_TREINO if m["meses"] <= 60]
MAX_TIMESTEPS = len(MARCOS_TREINO) - 1

SEQ_FEATURE_NAMES = [
    "genero_num",
    "idade",
    "dislipidemia",
    "diabetes",
    "peso_max_kg",
    "dia_medicao",
    "imc_medicao",
    "delta_dias_ate_alvo",
    "dia_medicao_estimado",
    "dia_alvo",
    "dia_alvo_estimado",
    "medicao_peso_minimo",
    "medicao_peso_proposta",
    "medicao_valida",
]

NUM_SEQ_FEATURES = len(SEQ_FEATURE_NAMES)


# ============================================================
# UTILITARIOS DE COLUNAS E NUMEROS
# ============================================================
def normalizar_texto(valor):
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(texto.split())


def encontrar_coluna(df, *partes):
    partes_norm = [normalizar_texto(p) for p in partes]
    for coluna in df.columns:
        nome_norm = normalizar_texto(coluna)
        if all(parte in nome_norm for parte in partes_norm):
            return coluna
    return None


def serie_numerica(serie):
    valores = (
        serie.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace(
            {
                "": np.nan,
                "-": np.nan,
                "--": np.nan,
                "#VALUE!": np.nan,
                "sem registo": np.nan,
                "sem timing": np.nan,
            }
        )
    )
    return pd.to_numeric(valores, errors="coerce")


def coluna_numerica(df_raw, *partes):
    coluna = encontrar_coluna(df_raw, *partes)
    if coluna is None:
        return pd.Series(np.nan, index=df_raw.index)
    return serie_numerica(df_raw[coluna])


def coluna_texto(df_raw, *partes):
    coluna = encontrar_coluna(df_raw, *partes)
    if coluna is None:
        return pd.Series(np.nan, index=df_raw.index)
    return df_raw[coluna]


def calcular_imc(peso, altura):
    if altura is None or altura <= 0:
        return np.nan
    return peso / (altura**2)


def escolher_imc(peso, altura, imc_original):
    imc_calculado = peso / (altura**2)
    imc_final = imc_calculado.where(imc_calculado.between(10, 80), imc_original)
    return imc_final.where(imc_final.between(10, 80), np.nan)


# ============================================================
# 1. CARREGAMENTO E LIMPEZA
# ============================================================
def carregar_dados(caminho):
    df_raw = pd.read_csv(caminho)
    df = pd.DataFrame(index=df_raw.index)

    df["genero"] = coluna_texto(df_raw, "genero")
    df["idade"] = coluna_numerica(df_raw, "idade_anos")
    df["altura"] = coluna_numerica(df_raw, "altura_m")
    df["dislipidemia"] = coluna_numerica(df_raw, "dislipidemia").fillna(0).astype(int)
    df["diabetes"] = coluna_numerica(df_raw, "diabetes").fillna(0).astype(int)

    df["genero_num"] = (
        df["genero"].astype(str).str.strip().str[0].str.upper().map({"F": 0, "M": 1})
    )
    df.loc[~df["altura"].between(1.0, 2.3), "altura"] = np.nan

    df["peso_0"] = coluna_numerica(df_raw, "peso dia da cirurgia")
    df["imc_original_0"] = coluna_numerica(df_raw, "imc_kg/m2")
    df["peso_max_kg"] = coluna_numerica(df_raw, "peso max")
    df["peso_max_kg"] = df["peso_max_kg"].where(df["peso_max_kg"].between(30, 300), np.nan)
    df["peso_proposta_cirurgica"] = coluna_numerica(df_raw, "peso proposta cirurgica")
    df["dia_original_peso_proposta"] = coluna_numerica(
        df_raw, "numero de dias_peso proposta", "cirurgia"
    )
    df["peso_minimo_pos"] = coluna_numerica(df_raw, "peso minimo", "pos-operatorio")
    df["dia_original_peso_minimo"] = coluna_numerica(
        df_raw, "numero de dias_peso minimo", "cirurgia"
    )

    colunas_marcos = {
        "3m": {
            "dia": ("numero de dias_3meses", "cirurgia"),
            "peso": ("peso_3_mes_kg",),
            "imc": ("imc_3_mes",),
        },
        "6m": {
            "dia": ("numero de dias_6meses", "cirurgia"),
            "peso": ("peso_6_mes_kg",),
            "imc": ("imc_6_mes",),
        },
        "12m": {
            "dia": ("numero de dias_1ano", "cirurgia"),
            "peso": ("peso_12_mes_kg",),
            "imc": ("imc_12_mes",),
        },
        "24m": {
            "dia": ("numero de dias_2anos", "cirurgia"),
            "peso": ("peso_24_mes_kg",),
            "imc": ("imc_24_mes",),
        },
        "36m": {
            "dia": ("numero de dias_3anos", "cirurgia"),
            "peso": ("peso_36_mes_kg",),
            "imc": ("imc_36_mes",),
        },
        "48m": {
            "dia": ("numero de dias_4anos", "cirurgia"),
            "peso": ("peso_48_meses",),
            "imc": ("imc_48_mes",),
        },
        "60m": {
            "dia": ("numero de dias_5anos", "cirurgia"),
            "peso": ("peso_60_meses",),
            "imc": ("imc_60_mes",),
        },
        "72m": {
            "dia": ("numero de dias_6anos", "cirurgia"),
            "peso": ("peso_72_meses",),
            "imc": ("imc_72_mes",),
        },
        "84m": {
            "dia": ("numero de dias_7anos", "cirurgia"),
            "peso": ("peso_84_meses",),
            "imc": ("imc_84_mes",),
        },
    }

    for nome, config in colunas_marcos.items():
        df[f"dia_original_{nome}"] = coluna_numerica(df_raw, *config["dia"])
        df[f"peso_{nome}"] = coluna_numerica(df_raw, *config["peso"])
        df[f"imc_original_{nome}"] = coluna_numerica(df_raw, *config["imc"])

    df["imc_0"] = escolher_imc(df["peso_0"], df["altura"], df["imc_original_0"])
    df["dia_0"] = 0.0
    df["dia_0_estimado"] = 0

    df["imc_peso_proposta"] = escolher_imc(
        df["peso_proposta_cirurgica"],
        df["altura"],
        pd.Series(np.nan, index=df.index),
    )
    dia_peso_proposta = df["dia_original_peso_proposta"].where(
        df["dia_original_peso_proposta"].between(-3650, 3650)
        & (df["dia_original_peso_proposta"] != 0),
        np.nan,
    )
    tem_peso_proposta = df["imc_peso_proposta"].notna() & dia_peso_proposta.notna()
    df["dia_peso_proposta"] = dia_peso_proposta.where(tem_peso_proposta, np.nan)
    df["dia_peso_proposta_estimado"] = 0

    df["imc_peso_minimo"] = escolher_imc(
        df["peso_minimo_pos"],
        df["altura"],
        pd.Series(np.nan, index=df.index),
    )
    dia_peso_minimo = df["dia_original_peso_minimo"].where(
        df["dia_original_peso_minimo"].between(1, 3650), np.nan
    )
    tem_peso_minimo = df["imc_peso_minimo"].notna() & dia_peso_minimo.notna()
    df["dia_peso_minimo"] = dia_peso_minimo.where(tem_peso_minimo, np.nan)
    df["dia_peso_minimo_estimado"] = 0

    for marco in MARCOS_TREINO[1:]:
        nome = marco["nome"]
        dia_col = f"dia_original_{nome}"

        df[f"imc_{nome}"] = escolher_imc(
            df[f"peso_{nome}"], df["altura"], df[f"imc_original_{nome}"]
        )

        dia_real = df[dia_col].where(df[dia_col].between(1, 3650), np.nan)
        tem_medicao = df[f"imc_{nome}"].notna()
        dia_estimado = dia_real.isna() & tem_medicao

        df[f"dia_{nome}"] = dia_real.where(
            dia_real.notna(), np.where(tem_medicao, marco["dia_nominal"], np.nan)
        )
        df[f"dia_{nome}_estimado"] = dia_estimado.astype(int)

    return df


# ============================================================
# 2. CONSTRUCAO DAS AMOSTRAS TEMPORAIS
# ============================================================
def obter_variaveis_estaticas(row):
    valores = {
        "genero_num": row.get("genero_num"),
        "idade": row.get("idade"),
        "dislipidemia": row.get("dislipidemia"),
        "diabetes": row.get("diabetes"),
        "peso_max_kg": row.get("peso_max_kg"),
    }
    if any(pd.isna(v) for v in valores.values()):
        return None
    return {k: float(v) for k, v in valores.items()}


def obter_medicoes_linha(row):
    medicoes = []

    for marco in MARCOS_TREINO:
        nome = marco["nome"]
        dia = row.get(f"dia_{nome}")
        imc = row.get(f"imc_{nome}")
        dia_estimado = row.get(f"dia_{nome}_estimado", 0)

        if pd.isna(dia) or pd.isna(imc):
            continue

        medicoes.append(
            {
                "nome": nome,
                "meses": marco["meses"],
                "dia": float(dia),
                "imc": float(imc),
                "dia_estimado": float(dia_estimado),
                "peso_minimo": 0.0,
                "peso_proposta": 0.0,
                "usar_como_alvo": True,
            }
        )

    dia_peso_proposta = row.get("dia_peso_proposta")
    imc_peso_proposta = row.get("imc_peso_proposta")
    if not pd.isna(dia_peso_proposta) and not pd.isna(imc_peso_proposta):
        medicoes.append(
            {
                "nome": "peso_proposta",
                "meses": float(dia_peso_proposta) / 30.4375,
                "dia": float(dia_peso_proposta),
                "imc": float(imc_peso_proposta),
                "dia_estimado": float(row.get("dia_peso_proposta_estimado", 0)),
                "peso_minimo": 0.0,
                "peso_proposta": 1.0,
                "usar_como_alvo": False,
            }
        )

    dia_peso_minimo = row.get("dia_peso_minimo")
    imc_peso_minimo = row.get("imc_peso_minimo")
    if not pd.isna(dia_peso_minimo) and not pd.isna(imc_peso_minimo):
        medicoes.append(
            {
                "nome": "peso_min",
                "meses": float(dia_peso_minimo) / 30.4375,
                "dia": float(dia_peso_minimo),
                "imc": float(imc_peso_minimo),
                "dia_estimado": float(row.get("dia_peso_minimo_estimado", 0)),
                "peso_minimo": 1.0,
                "peso_proposta": 0.0,
                "usar_como_alvo": False,
            }
        )

    medicoes.sort(key=lambda item: item["dia"])
    return medicoes


def construir_sequencia(static_values, historico, dia_alvo, dia_alvo_estimado=0):
    historico_valido = [
        m for m in historico if np.isfinite(m["dia"]) and np.isfinite(m["imc"]) and m["dia"] < dia_alvo
    ]
    historico_valido = sorted(historico_valido, key=lambda item: item["dia"])[-MAX_TIMESTEPS:]

    seq = np.zeros((MAX_TIMESTEPS, NUM_SEQ_FEATURES), dtype=float)

    for t, medicao in enumerate(historico_valido):
        seq[t] = [
            static_values["genero_num"],
            static_values["idade"],
            static_values["dislipidemia"],
            static_values["diabetes"],
            static_values["peso_max_kg"],
            medicao["dia"],
            medicao["imc"],
            dia_alvo - medicao["dia"],
            medicao.get("dia_estimado", 0),
            dia_alvo,
            dia_alvo_estimado,
            medicao.get("peso_minimo", 0),
            medicao.get("peso_proposta", 0),
            1.0,
        ]

    return seq


def criar_amostras_treino(df):
    X, y, nomes_alvo = [], [], []

    for _, row in df.iterrows():
        static_values = obter_variaveis_estaticas(row)
        if static_values is None:
            continue

        medicoes = obter_medicoes_linha(row)
        if len(medicoes) < 2:
            continue

        for alvo in medicoes:
            if alvo["nome"] == "0" or not alvo.get("usar_como_alvo", True):
                continue

            historico = [m for m in medicoes if m["dia"] < alvo["dia"]]
            if not historico:
                continue

            X.append(
                construir_sequencia(
                    static_values,
                    historico,
                    alvo["dia"],
                    dia_alvo_estimado=alvo.get("dia_estimado", 0),
                )
            )
            y.append(alvo["imc"])
            nomes_alvo.append(alvo["nome"])

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), nomes_alvo


def aplicar_scaler_x(X_seq, scaler):
    shape = X_seq.shape
    X_flat = X_seq.reshape(shape[0], -1)
    X_flat_sc = scaler.transform(X_flat)
    return X_flat_sc.reshape(shape)


# ============================================================
# 3. TREINO DO MODELO TEMPORAL
# ============================================================
def treinar_modelo_temporal(df):
    X, y, nomes_alvo = criar_amostras_treino(df)

    if len(X) < 30:
        raise ValueError("Dados insuficientes para treinar o modelo temporal.")

    print(f"\nAmostras temporais criadas: {len(X)}")
    print("Distribuicao dos alvos usados no treino:")
    print(pd.Series(nomes_alvo).value_counts().sort_index().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    sc_x = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_sc = sc_x.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_sc = sc_x.transform(X_test_flat).reshape(X_test.shape)

    sc_y = StandardScaler()
    y_train_sc = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    modelo = Sequential(
        [
            Input(shape=(MAX_TIMESTEPS, NUM_SEQ_FEATURES)),
            LSTM(32, activation="tanh", recurrent_dropout=0.2, kernel_regularizer=l2(0.001)),
            Dropout(0.25),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(delta=1.0))

    modelo.fit(
        X_train_sc,
        y_train_sc,
        epochs=150,
        batch_size=16,
        validation_split=0.15,
        callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
        verbose=0,
    )

    pred_sc = modelo.predict(X_test_sc, verbose=0)
    pred = sc_y.inverse_transform(pred_sc).ravel()
    mae = mean_absolute_error(y_test, pred)

    print(f"\nMAE de teste do modelo temporal: {mae:.3f} IMC")

    metadata = {
        "max_timesteps": MAX_TIMESTEPS,
        "seq_feature_names": SEQ_FEATURE_NAMES,
        "marcos_treino": MARCOS_TREINO,
        "marcos_interface": MARCOS_INTERFACE,
        "mae_teste": float(mae),
        "usa_peso_minimo_pos_operatorio": True,
        "usa_peso_maximo": True,
        "usa_peso_proposta_cirurgica": True,
        "colunas_peso_minimo": {
            "peso": "Peso minimo Pos-operatorio",
            "dia": "Numero de Dias_peso minimo desde Cirurgia (D0)",
        },
        "coluna_peso_maximo": "Peso Max_kg",
        "colunas_peso_proposta": {
            "peso": "Peso PROPOSTA CIRURGICA",
            "dia": "Numero de Dias_peso proposta desde Cirurgia (D0)",
        },
    }

    modelo.save(CAMINHO_MODELO_TEMPORAL)
    joblib.dump(sc_x, CAMINHO_SCALER_X_TEMPORAL)
    joblib.dump(sc_y, CAMINHO_SCALER_Y_TEMPORAL)
    joblib.dump(metadata, CAMINHO_METADATA_TEMPORAL)

    print(f"Modelo salvo em: {CAMINHO_MODELO_TEMPORAL}")
    return modelo, sc_x, sc_y, metadata


# ============================================================
# 4. PREVISAO CLINICA
# ============================================================
def carregar_modelo_temporal():
    if not os.path.exists(CAMINHO_MODELO_TEMPORAL):
        raise FileNotFoundError(
            "Falta o modelo temporal. Corre primeiro: python "
            "previsao_peso_2026_versao2.0/Scripts/LSTM_Curva4_pesoproposta_pesomax.py"
        )

    modelo = load_model(CAMINHO_MODELO_TEMPORAL, compile=False)
    sc_x = joblib.load(CAMINHO_SCALER_X_TEMPORAL)
    sc_y = joblib.load(CAMINHO_SCALER_Y_TEMPORAL)
    metadata = joblib.load(CAMINHO_METADATA_TEMPORAL)
    return modelo, sc_x, sc_y, metadata


def prever_imc_no_dia(modelo, sc_x, sc_y, static_values, historico, dia_alvo):
    seq = construir_sequencia(static_values, historico, dia_alvo, dia_alvo_estimado=0)
    X = seq.reshape(1, MAX_TIMESTEPS, NUM_SEQ_FEATURES)
    X_sc = aplicar_scaler_x(X, sc_x)
    pred_sc = modelo.predict(X_sc, verbose=0)
    return float(sc_y.inverse_transform(pred_sc)[0][0])


def prever_trajetoria_temporal(modelo, sc_x, sc_y, static_values, medicoes_iniciais):
    historico = sorted(medicoes_iniciais, key=lambda item: item["dia"])
    resultados = []

    imc_inicial = next((m["imc"] for m in historico if m["dia"] == 0), None)
    if imc_inicial is not None:
        resultados.append(
            {"nome": "0", "meses": 0, "dia": 0.0, "imc": float(imc_inicial), "origem": "real"}
        )

    for marco in MARCOS_INTERFACE[1:]:
        dia_alvo = float(marco["dia_nominal"])
        medicao_real_no_alvo = next(
            (m for m in historico if abs(m["dia"] - dia_alvo) <= 0.5), None
        )

        if medicao_real_no_alvo is not None:
            pred_final = float(medicao_real_no_alvo["imc"])
            origem = "real"
        else:
            pred_final = prever_imc_no_dia(modelo, sc_x, sc_y, static_values, historico, dia_alvo)
            origem = "previsto"
            historico.append(
                {
                    "nome": marco["nome"],
                    "dia": dia_alvo,
                    "imc": pred_final,
                    "dia_estimado": 1.0,
                    "peso_minimo": 0.0,
                    "peso_proposta": 0.0,
                    "usar_como_alvo": True,
                    "origem": origem,
                }
            )
            historico.sort(key=lambda item: item["dia"])

        resultados.append(
            {
                "nome": marco["nome"],
                "meses": marco["meses"],
                "dia": dia_alvo,
                "imc": pred_final,
                "origem": origem,
            }
        )

    return resultados


# ============================================================
# 5. VISUALIZACAO DE TESTE
# ============================================================
def visualizar_trajetoria_doente(df, doente_idx=0):
    import matplotlib.pyplot as plt

    modelo, sc_x, sc_y, _ = treinar_modelo_temporal(df)

    row = df.iloc[doente_idx]
    static_values = obter_variaveis_estaticas(row)
    medicoes_reais = obter_medicoes_linha(row)

    if static_values is None or not medicoes_reais:
        print("Doente sem dados suficientes para visualizar.")
        return

    medicoes_iniciais = [m for m in medicoes_reais if m["nome"] == "0"]
    previsoes = prever_trajetoria_temporal(modelo, sc_x, sc_y, static_values, medicoes_iniciais)

    plt.figure(figsize=(10, 6))

    dias_reais = [m["dia"] for m in medicoes_reais if m["meses"] <= 84]
    imcs_reais = [m["imc"] for m in medicoes_reais if m["meses"] <= 84]
    plt.plot(np.array(dias_reais) / 30.4375, imcs_reais, "ok-", label="Dados reais", alpha=0.45)

    dias_prev = [p["dia"] for p in previsoes]
    imcs_prev = [p["imc"] for p in previsoes]
    plt.plot(np.array(dias_prev) / 30.4375, imcs_prev, "sr--", label="Previsao ate 60m")

    plt.title(f"BariCurve temporal - doente indice {doente_idx}")
    plt.xlabel("Meses apos cirurgia")
    plt.ylabel("IMC (kg/m2)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.show()


# ============================================================
# EXECUCAO
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(CAMINHO_CSV):
        print(f"ERRO: Ficheiro nao encontrado em: {CAMINHO_CSV}")
    else:
        base_dados = carregar_dados(CAMINHO_CSV)
        treinar_modelo_temporal(base_dados)
