from flask import Flask, jsonify, send_file
from flask_cors import CORS
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
import threading

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("WEATHER_API_KEY", "185b46c637e9436e80525120250811")
CIUDAD = "20.5888,-100.3899"  # Coordenadas de Querétaro
MODEL_FILE = "modelo_lluvia.pkl"
DATA_FILE = "historico_clima.json"
OUTPUT_FILE = "pronostico_lluvia_queretaro.json"

# --- FUNCIONES AUXILIARES (SIN CAMBIOS) ---

def generar_historico(ciudad):
    print("Generando histórico climático de los últimos 60 días...\n")
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
        else:
            print(f"No se pudo obtener datos para {fecha}")

    df = pd.DataFrame(historico)
    df.to_json(DATA_FILE, orient="records", indent=4)
    print(f"Archivo '{DATA_FILE}' generado con {len(historico)} días de datos.\n")
    return df


def obtener_datos_climaticos(ciudad):
    print(f"Obteniendo datos actuales de {ciudad}...\n")
    url = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={ciudad}&days=5&aqi=no&alerts=no"
    resp = requests.get(url)
    data = resp.json()

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
    print("Entrenando modelo de predicción de lluvia...\n")
    features = ["temp", "viento", "humedad", "presion", "nubosidad", "lluvia_total"]
    df["llovera"] = (df["lluvia_total"] > 1).astype(int)

    X = df[features]
    y = df["llovera"]

    if len(y.unique()) < 2:
        print("Solo hay una clase (ej. no llovió). Se generará modelo constante.\n")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        modelo = DummyClassifier(strategy="constant", constant=0)
        modelo.fit(X_scaled, y)

        joblib.dump({"modelo": modelo, "scaler": scaler}, MODEL_FILE)
        return modelo, scaler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = LogisticRegression()
    modelo.fit(X_train_scaled, y_train)

    pred = modelo.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    print(f"Precisión del modelo: {acc*100:.2f}%\n")

    joblib.dump({"modelo": modelo, "scaler": scaler}, MODEL_FILE)
    return modelo, scaler


def cargar_modelo():
    if os.path.exists(MODEL_FILE):
        print("Cargando modelo existente...\n")
        data = joblib.load(MODEL_FILE)
        return data["modelo"], data["scaler"]
    else:
        print("No se encontró modelo. Entrenando uno nuevo...\n")
        if not os.path.exists(DATA_FILE):
            df = generar_historico(CIUDAD)
        else:
            df = pd.read_json(DATA_FILE)
        return entrenar_modelo(df)


def generar_pronostico():
    print("\n============================")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("============================\n")

    weather_data = obtener_datos_climaticos(CIUDAD)
    modelo, scaler = cargar_modelo()

    print("\nObteniendo pronóstico de los próximos 5 días...\n")
    predicciones = []

    for fecha, datos_dia in weather_data.items():
        try:
            X_df = pd.DataFrame([[datos_dia["temp"], datos_dia["viento"], datos_dia["humedad"],
                                  datos_dia["presion"], datos_dia["nubosidad"], datos_dia["lluvia_total"]]],
                                columns=["temp", "viento", "humedad", "presion", "nubosidad", "lluvia_total"])

            x_scaled = scaler.transform(X_df)
            prediccion = modelo.predict(x_scaled)[0]

            resultado = {
                "fecha": fecha,
                "condicion": datos_dia["condicion"],
                "prob_lluvia_api": datos_dia["prob_lluvia"],
                "llovera_modelo": bool(prediccion)
            }
            predicciones.append(resultado)

            estado = "🌧️ Lloverá" if prediccion else "☀️ No lloverá"
            print(f"{fecha}: {estado} (Condición: {datos_dia['condicion']}, Prob. API: {datos_dia['prob_lluvia']}%)")

        except Exception as e:
            print(f"Error procesando {fecha}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(predicciones, f, indent=4, ensure_ascii=False)

    print(f"\nArchivo '{OUTPUT_FILE}' generado con éxito.\n")


# --- ENDPOINTS PARA QUE FUNCIONE COMO WEB SERVICE ---

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'API de predicción meteorológica funcionando',
        'ultima_actualizacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/pronostico', methods=['GET'])
def obtener_pronostico():
    """Obtener el pronóstico actual (el archivo JSON generado)"""
    try:
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({
                'status': 'success',
                'data': data
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No hay pronóstico disponible. El servicio se está iniciando...'
            }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/actualizar', methods=['GET'])
def forzar_actualizacion():
    """Endpoint para forzar una actualización manual del pronóstico"""
    try:
        generar_pronostico()
        return jsonify({
            'status': 'success',
            'message': 'Pronóstico actualizado correctamente'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# --- GENERAR PRONÓSTICO AL INICIAR ---
def inicializar():
    """Se ejecuta una vez al iniciar el servidor"""
    print("🚀 Iniciando servidor de pronóstico meteorológico...")
    try:
        generar_pronostico()
        print("✅ Pronóstico inicial generado correctamente")
    except Exception as e:
        print(f"❌ Error al generar pronóstico inicial: {e}")


if __name__ == '__main__':
    # Generar el pronóstico inicial en un hilo separado
    threading.Thread(target=inicializar, daemon=True).start()
    
    # Iniciar el servidor Flask
    app.run(host='0.0.0.0', port=5002, debug=False)
