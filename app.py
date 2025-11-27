import requests
import json
import os
import time
import joblib
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier


API_KEY = "185b46c637e9436e80525120250811"
CIUDAD = "20.5888,-100.3899"
MODEL_FILE = "modelo_lluvia.pkl"
DATA_FILE = "historico_clima.json"

FIREBASE_URL = "https://weatheriadx-default-rtdb.firebaseio.com/"



def generar_historico(ciudad):
    print("Generando histórico climático...\n")
    historico = []

    for i in range(1, 61):
        fecha = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        url = f"https://api.weatherapi.com/v1/history.json?key={API_KEY}&q={ciudad}&dt={fecha}"
        resp = requests.get(url)

        if resp.status_code == 200:
            data = resp.json()
            dia = data["forecast"]["forecastday"][0]["day"]
            historico.append({
                "fecha": fecha,
                "temp": dia["avgtemp_c"],
                "viento": dia["maxwind_kph"],
                "humedad": dia["avghumidity"],
                "presion": dia.get("pressure_mb", 1013),
                "nubosidad": dia.get("daily_chance_of_rain", 0),
                "lluvia_total": dia.get("totalprecip_mm", 0)
            })

    df = pd.DataFrame(historico)
    df.to_json(DATA_FILE, orient="records", indent=4)
    return df


def obtener_datos_climaticos(ciudad):
    print("Obteniendo datos actuales...\n")
    url = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={ciudad}&days=5&aqi=no&alerts=no"
    data = requests.get(url).json()

    weather_data = {}
    for dia in data["forecast"]["forecastday"]:
        fecha = dia["date"]
        day_info = dia["day"]
        weather_data[fecha] = {
            "temp": day_info["avgtemp_c"],
            "viento": day_info["maxwind_kph"],
            "humedad": day_info["avghumidity"],
            "presion": day_info.get("pressure_mb", 1013),
            "nubosidad": day_info.get("daily_chance_of_rain", 0),
            "lluvia_total": day_info.get("totalprecip_mm", 0),
            "prob_lluvia": day_info.get("daily_chance_of_rain", 0),
            "condicion": day_info["condition"]["text"]
        }

    return weather_data


def entrenar_modelo(df):
    features = ["temp", "viento", "humedad", "presion", "nubosidad", "lluvia_total"]
    df["llovera"] = (df["lluvia_total"] > 1).astype(int)

    X = df[features]
    y = df["llovera"]

    if len(y.unique()) < 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        modelo = DummyClassifier(strategy="constant", constant=0)
        modelo.fit(X_scaled, y)
        joblib.dump({"modelo": modelo, "scaler": scaler}, MODEL_FILE)
        return modelo, scaler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = LogisticRegression()
    modelo.fit(X_train_scaled, y_train)

    joblib.dump({"modelo": modelo, "scaler": scaler}, MODEL_FILE)
    return modelo, scaler


def cargar_modelo():
    if os.path.exists(MODEL_FILE):
        data = joblib.load(MODEL_FILE)
        return data["modelo"], data["scaler"]
    else:
        if os.path.exists(DATA_FILE):
            df = pd.read_json(DATA_FILE)
        else:
            df = generar_historico(CIUDAD)
        return entrenar_modelo(df)


def guardar_en_firebase(predicciones):
    payload = {
        "ciudad": CIUDAD,
        "fecha_actualizacion": datetime.now().isoformat(),
        "predicciones": predicciones
    }

    url = f"{FIREBASE_URL}/pronostico_queretaro.json"

    r = requests.put(url, json=payload)

    if r.status_code == 200:
        print("Pronóstico guardado en Firebase correctamente.")
    else:
        print("Error enviando a Firebase:", r.text)



def generar_pronostico():
    print("\n============================")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("============================\n")

    weather_data = obtener_datos_climaticos(CIUDAD)
    modelo, scaler = cargar_modelo()

    predicciones = []

    for fecha, datos_dia in weather_data.items():
        X_df = pd.DataFrame([[ 
            datos_dia["temp"],
            datos_dia["viento"],
            datos_dia["humedad"],
            datos_dia["presion"],
            datos_dia["nubosidad"],
            datos_dia["lluvia_total"]
        ]], columns=["temp","viento","humedad","presion","nubosidad","lluvia_total"])

        x_scaled = scaler.transform(X_df)
        prediccion = modelo.predict(x_scaled)[0]

        resultado = {
            "fecha": fecha,
            "condicion": datos_dia["condicion"],
            "prob_lluvia_api": datos_dia["prob_lluvia"],
            "llovera_modelo": bool(prediccion)
        }

        predicciones.append(resultado)

        estado = "Lloverá" if prediccion else "No lloverá"
        print(f"{fecha}: {estado}")

    guardar_en_firebase(predicciones)



if __name__ == "__main__":
    while True:
        generar_pronostico()
        print("⏳ Esperando 12 horas...\n")
        time.sleep(43200)
