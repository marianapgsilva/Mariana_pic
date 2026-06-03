from tensorflow.keras.models import load_model

model = load_model(
    "../models/modelo_lstm_curva4_pesoproposta_pesomax.keras",
    compile=False
)

model.save("../models/modelo_final.h5")

print("Conversão concluída!")