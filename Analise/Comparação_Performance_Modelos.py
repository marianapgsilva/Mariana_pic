"""
Comparacao temporal dos modelos BariCurve
=========================================
Esta versao usa a mesma representacao temporal da versao 2.0:
cada medicao entra como dia real desde cirurgia + IMC + delta ate ao alvo.

Modelos comparados:
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
  - MLP
  - LSTM

A validacao usa GroupKFold para que medicoes do mesmo doente nao aparecam ao
mesmo tempo em treino e teste.
"""

import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJETO_ROOT = os.path.join(BASE_DIR, "..")
SCRIPTS_DIR = os.path.join(PROJETO_ROOT, "Scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from LSTM_Curva import (  # noqa: E402
    CAMINHO_CSV,
    MAX_TIMESTEPS,
    NUM_SEQ_FEATURES,
    construir_sequencia,
    carregar_dados,
    obter_medicoes_linha,
    obter_variaveis_estaticas,
)

INTERVALO_ACERTO = 2.0
NUM_FOLDS = 5
TARGETS_CLINICOS = {"3m", "6m", "12m", "24m", "36m", "48m", "60m"}


def criar_amostras_temporais(df):
    X, y, alvo_nome, grupo_doente = [], [], [], []

    for idx, row in df.iterrows():
        static_values = obter_variaveis_estaticas(row)
        if static_values is None:
            continue

        medicoes = obter_medicoes_linha(row)
        if len(medicoes) < 2:
            continue

        for alvo in medicoes:
            if alvo["nome"] == "0":
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
            alvo_nome.append(alvo["nome"])
            grupo_doente.append(idx)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(alvo_nome),
        np.asarray(grupo_doente),
    )


def criar_modelo_lstm():
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
    return modelo


def avaliar_modelos(X_seq, y, alvo_nome, grupos):
    modelos = [
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "MLP",
        "LSTM",
    ]
    preds = {nome: np.full(len(y), np.nan) for nome in modelos}
    X_flat = X_seq.reshape(X_seq.shape[0], -1)

    cv = GroupKFold(n_splits=NUM_FOLDS)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_flat, y, groups=grupos), start=1):
        print(f"\nFold {fold_idx}/{NUM_FOLDS}")

        X_train_flat, X_test_flat = X_flat[train_idx], X_flat[test_idx]
        X_train_seq, X_test_seq = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        sc_x = StandardScaler()
        X_train_flat_sc = sc_x.fit_transform(X_train_flat)
        X_test_flat_sc = sc_x.transform(X_test_flat)
        X_train_seq_sc = X_train_flat_sc.reshape(X_train_seq.shape)
        X_test_seq_sc = X_test_flat_sc.reshape(X_test_seq.shape)

        sc_y = StandardScaler()
        y_train_sc = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train_flat, y_train)
        preds["Random Forest"][test_idx] = rf.predict(X_test_flat)

        xgb = XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb.fit(X_train_flat, y_train)
        preds["XGBoost"][test_idx] = xgb.predict(X_test_flat)

        lgbm = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=16,
            min_child_samples=10,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(X_train_flat, y_train)
        preds["LightGBM"][test_idx] = lgbm.predict(X_test_flat)

        cat = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=4,
            loss_function="RMSE",
            eval_metric="MAE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        cat.fit(X_train_flat, y_train)
        preds["CatBoost"][test_idx] = cat.predict(X_test_flat)

        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=700,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
        mlp.fit(X_train_flat_sc, y_train_sc)
        pred_mlp_sc = mlp.predict(X_test_flat_sc)
        preds["MLP"][test_idx] = sc_y.inverse_transform(pred_mlp_sc.reshape(-1, 1)).ravel()

        lstm = criar_modelo_lstm()
        lstm.fit(
            X_train_seq_sc,
            y_train_sc,
            epochs=150,
            batch_size=16,
            validation_split=0.15,
            callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
            verbose=0,
        )
        pred_lstm_sc = lstm.predict(X_test_seq_sc, verbose=0).ravel()
        preds["LSTM"][test_idx] = sc_y.inverse_transform(pred_lstm_sc.reshape(-1, 1)).ravel()

        for nome in modelos:
            mae_fold = mean_absolute_error(y_test, preds[nome][test_idx])
            print(f"  {nome:13}: MAE fold = {mae_fold:.3f}")

    return preds


def calcular_metricas(y, preds, alvo_nome):
    linhas_por_marco = []
    linhas_ranking = []

    for modelo, y_pred in preds.items():
        valid = np.isfinite(y_pred)
        erro = np.abs(y[valid] - y_pred[valid])
        alvo_valid = alvo_nome[valid]

        for alvo in sorted(np.unique(alvo_valid), key=ordem_alvo):
            mask = alvo_valid == alvo
            erros_alvo = erro[mask]
            linhas_por_marco.append(
                {
                    "Modelo": modelo,
                    "Marco": alvo,
                    "N": int(mask.sum()),
                    "MAE": float(np.mean(erros_alvo)),
                    "Taxa Acerto (%)": float((erros_alvo <= INTERVALO_ACERTO).mean() * 100),
                }
            )

        mask_clinico = np.isin(alvo_valid, list(TARGETS_CLINICOS))
        erro_clinico = erro[mask_clinico]

        linhas_ranking.append(
            {
                "Modelo": modelo,
                "MAE Global <=60m": float(np.mean(erro_clinico)),
                "Taxa Acerto <=60m (%)": float((erro_clinico <= INTERVALO_ACERTO).mean() * 100),
                "N <=60m": int(mask_clinico.sum()),
                "MAE Global <=84m": float(np.mean(erro)),
                "Taxa Acerto <=84m (%)": float((erro <= INTERVALO_ACERTO).mean() * 100),
                "N <=84m": int(valid.sum()),
            }
        )

    por_marco_df = pd.DataFrame(linhas_por_marco)
    ranking_df = pd.DataFrame(linhas_ranking).sort_values("MAE Global <=60m", ascending=True)
    ranking_df.index = range(1, len(ranking_df) + 1)

    return por_marco_df, ranking_df


def ordem_alvo(alvo):
    return {"3m": 3, "6m": 6, "12m": 12, "24m": 24, "36m": 36, "48m": 48, "60m": 60, "72m": 72, "84m": 84}.get(
        alvo, 999
    )


def gerar_grafico(por_marco_df, ranking_df):
    modelos = ranking_df["Modelo"].tolist()
    marcos = sorted(por_marco_df["Marco"].unique(), key=ordem_alvo)
    cores = {
        "Random Forest": "#2ca02c",
        "XGBoost": "#1f77b4",
        "LightGBM": "#17becf",
        "CatBoost": "#ff7f0e",
        "MLP": "#d62728",
        "LSTM": "#9467bd",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("BariCurve 2.0 - Comparacao temporal com dias reais", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    x = np.arange(len(marcos))
    width = 0.12
    offsets = np.linspace(-2.5, 2.5, len(modelos)) * width

    for offset, modelo in zip(offsets, modelos):
        valores = []
        for marco in marcos:
            row = por_marco_df[(por_marco_df["Modelo"] == modelo) & (por_marco_df["Marco"] == marco)]
            valores.append(float(row["MAE"].iloc[0]) if not row.empty else 0.0)
        ax1.bar(x + offset, valores, width, label=modelo, color=cores.get(modelo), alpha=0.9)

    ax1.set_xlabel("Marco temporal")
    ax1.set_ylabel("MAE (IMC)")
    ax1.set_title("MAE por marco")
    ax1.set_xticks(x)
    ax1.set_xticklabels(marcos)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    ranking_plot = ranking_df.sort_values("MAE Global <=60m", ascending=True)
    ax2.barh(
        ranking_plot["Modelo"][::-1],
        ranking_plot["MAE Global <=60m"][::-1],
        color=[cores.get(m, "#999999") for m in ranking_plot["Modelo"][::-1]],
        alpha=0.9,
    )
    ax2.set_xlabel("MAE global ate 60m (IMC)")
    ax2.set_title("Ranking clinico ate 5 anos")
    ax2.grid(axis="x", alpha=0.3)

    for i, (_, row) in enumerate(ranking_plot[::-1].iterrows()):
        ax2.text(row["MAE Global <=60m"] + 0.03, i, f"{row['MAE Global <=60m']:.2f}", va="center")

    plt.tight_layout()
    caminho = os.path.join(BASE_DIR, "baricurve_comparacao_temporal_modelos.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"\nGrafico guardado em: {caminho}")


def guardar_resultados(por_marco_df, ranking_df):
    caminho_por_marco = os.path.join(BASE_DIR, "baricurve_comparacao_temporal_por_marco.csv")
    caminho_ranking = os.path.join(BASE_DIR, "baricurve_comparacao_temporal_ranking.csv")

    por_marco_df.to_csv(caminho_por_marco, index=False, encoding="utf-8-sig")
    ranking_df.to_csv(caminho_ranking, index=False, encoding="utf-8-sig")

    print(f"Resultados por marco guardados em: {caminho_por_marco}")
    print(f"Ranking guardado em: {caminho_ranking}")


def main():
    print("A iniciar comparacao temporal BariCurve 2.0...")
    print(f"Dados: {CAMINHO_CSV}")

    df = carregar_dados(CAMINHO_CSV)
    X_seq, y, alvo_nome, grupos = criar_amostras_temporais(df)

    print(f"Doentes carregados: {len(df)}")
    print(f"Amostras temporais: {len(y)}")
    print("Distribuicao dos alvos:")
    print(pd.Series(alvo_nome).value_counts().sort_index().to_string())
    print(f"Formato LSTM: {X_seq.shape}")
    print(f"Formato tabular: {(X_seq.shape[0], X_seq.shape[1] * X_seq.shape[2])}")

    preds = avaliar_modelos(X_seq, y, alvo_nome, grupos)
    por_marco_df, ranking_df = calcular_metricas(y, preds, alvo_nome)

    print("\nRanking final - alvo clinico ate 60m:")
    print(ranking_df.round(3).to_string())

    print("\nMAE por marco:")
    print(
        por_marco_df.pivot(index="Marco", columns="Modelo", values="MAE")
        .reindex(sorted(por_marco_df["Marco"].unique(), key=ordem_alvo))
        .round(3)
        .to_string()
    )

    guardar_resultados(por_marco_df, ranking_df)
    gerar_grafico(por_marco_df, ranking_df)


if __name__ == "__main__":
    main()
