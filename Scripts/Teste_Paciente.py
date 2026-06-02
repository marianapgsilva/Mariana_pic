from datetime import date

import matplotlib.pyplot as plt
import numpy as np

from LSTM_Curva import (
    MARCOS_INTERFACE,
    calcular_imc,
    carregar_modelo_temporal,
    prever_trajetoria_temporal,
)


def gerar_previsao_clinica(
    genero,
    idade,
    peso_0,
    altura,
    dislipidemia,
    data_cirurgia=None,
    data_peso_3m=None,
    peso_3m_real=None,
):
    modelo, sc_x, sc_y, _ = carregar_modelo_temporal()

    imc_0 = calcular_imc(peso_0, altura)
    static_values = {
        "genero_num": float(genero),
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

    pontos_reais = [{"dia": 0.0, "imc": float(imc_0)}]

    if data_cirurgia and data_peso_3m and peso_3m_real is not None:
        dia_3m_real = (data_peso_3m - data_cirurgia).days
        if dia_3m_real <= 0:
            raise ValueError("A data da medicao tem de ser posterior a data da cirurgia.")

        imc_3m_real = calcular_imc(peso_3m_real, altura)
        medicoes_iniciais.append(
            {
                "nome": "3m_real",
                "dia": float(dia_3m_real),
                "imc": float(imc_3m_real),
                "dia_estimado": 0.0,
                "origem": "real",
            }
        )
        pontos_reais.append({"dia": float(dia_3m_real), "imc": float(imc_3m_real)})

    resultados = prever_trajetoria_temporal(
        modelo, sc_x, sc_y, static_values, medicoes_iniciais
    )

    print("\n--- BariCurve temporal: previsao ate 5 anos ---")
    for r in resultados:
        print(f"{r['nome']:>3} | D{int(round(r['dia'])):>4} | {r['imc']:.2f} IMC | {r['origem']}")

    meses_prev = np.array([r["dia"] for r in resultados]) / 30.4375
    imcs_prev = np.array([r["imc"] for r in resultados])

    plt.figure(figsize=(10, 6))
    plt.plot(meses_prev, imcs_prev, "sr--", label="Projecao BariCurve", linewidth=2)
    plt.fill_between(
        meses_prev,
        imcs_prev - 2,
        imcs_prev + 2,
        color="red",
        alpha=0.1,
        label="Margem +/-2 IMC",
    )

    pontos_reais = sorted(pontos_reais, key=lambda item: item["dia"])
    plt.plot(
        [p["dia"] / 30.4375 for p in pontos_reais],
        [p["imc"] for p in pontos_reais],
        "ok-",
        label="Dados reais inseridos",
        markersize=7,
    )

    plt.title("BariCurve temporal: trajetoria ate 5 anos")
    plt.xlabel("Tempo apos cirurgia")
    plt.ylabel("IMC (kg/m2)")
    plt.xticks(
        [m["dia_nominal"] / 30.4375 for m in MARCOS_INTERFACE],
        [m["nome"] for m in MARCOS_INTERFACE],
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return resultados


if __name__ == "__main__":
    gerar_previsao_clinica(
        genero=0,
        idade=32,
        peso_0=168.0,
        altura=1.57,
        dislipidemia=1,
        data_cirurgia=date(2025, 1, 1),
        data_peso_3m=date(2025, 3, 17),
        peso_3m_real=134.6,
    )
