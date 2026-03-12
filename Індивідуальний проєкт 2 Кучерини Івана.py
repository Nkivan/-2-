import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ML Прогноз погоди", layout="centered")
st.title("🌦 ML прогноз опадів")

def get_coordinates(city):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "weather-ml-app"
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if len(data) == 0:
        return None, None

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon

def get_weather_data(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min,windspeed_10m_max",
        "timezone": "auto"
    }
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df

st.sidebar.header("Налаштування")
city = st.sidebar.text_input("Введіть місто", "Kyiv")
days_ahead = st.sidebar.slider(
    "Прогноз на скільки днів вперед",
    1,
    7,
    3
)

latitude, longitude = get_coordinates(city)

if latitude is None:
    st.error(" Місто не знайдено")
    st.stop()

st.write(f"Координати {city}: {latitude:.2f}, {longitude:.2f}")

start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")

if st.button("Отримати історичні дані"):
    df = get_weather_data(latitude, longitude, start_date, end_date)

    df["rain_label"] = df["precipitation_sum"].apply(lambda x: 1 if x > 0 else 0)

    df.to_csv("weather_daily.csv", index=False)
    st.info("Дані збережено у файл weather_daily.csv")

    st.session_state["data"] = df

    st.success(f"Дані для {city} отримано")
    st.write(f"Використано {len(df)} днів історичних даних")
    st.dataframe(df.tail())

    st.subheader(" Графік історичних опадів")
    history_chart = df[["time", "precipitation_sum"]].set_index("time")
    st.line_chart(history_chart)

if "data" in st.session_state:
    df = st.session_state["data"]

    features = [
        "rain_sum",
        "temperature_2m_max",
        "temperature_2m_min",
        "windspeed_10m_max"
    ]

    X = df[features]
    y = df["rain_label"]

    if st.button("Навчити модель"):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        data_size_factor = min(len(df) / 365, 1)
        adjusted_accuracy = accuracy * (0.7 + 0.3 * data_size_factor)

        st.session_state["model"] = model
        st.session_state["accuracy"] = adjusted_accuracy

        st.success(f"Точність моделі для {city}: {adjusted_accuracy*100:.2f}%")

        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Фактор": features,
            "Важливість": importances
        }).sort_values(by="Важливість", ascending=False)

        st.subheader("Важливість факторів")
        st.table(importance_df)

        st.subheader("Графік важливості факторів")
        chart_data = importance_df.set_index("Фактор")
        st.bar_chart(chart_data)

if "model" in st.session_state:
    st.subheader("Прогноз погоди")

    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min,windspeed_10m_max",
        "forecast_days": days_ahead,
        "timezone": "auto"
    }

    r = requests.get(forecast_url, params=params)
    data = r.json()

    df_future = pd.DataFrame(data["daily"])
    df_future["time"] = pd.to_datetime(df_future["time"])

    X_future = df_future[features]
    model = st.session_state["model"]

    predictions = model.predict(X_future)
    probabilities = model.predict_proba(X_future)[:, 1]

    results = []

    for i in range(len(df_future)):
        rain = "🌧 Опади" if predictions[i] == 1 else "☀ Без опадів"
        results.append({
            "Дата": df_future["time"][i].date(),
            "Прогноз": rain,
            "Ймовірність опадів": f"{probabilities[i]*100:.1f}%",
            "Макс температура °C": df_future["temperature_2m_max"][i],
            "Мін температура °C": df_future["temperature_2m_min"][i]
        })

    forecast_df = pd.DataFrame(results)
    st.table(forecast_df)

    st.subheader("🌧 Ймовірність опадів")
    for i in range(len(forecast_df)):
        date = forecast_df.iloc[i]["Дата"]
        prob = probabilities[i]
        st.write(f"{date} — {prob*100:.1f}%")
        st.progress(prob)

    st.info(f"Точність моделі для {city}: {st.session_state['accuracy']*100:.2f}%")