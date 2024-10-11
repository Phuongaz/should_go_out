from datetime import datetime
import pandas as pd
import requests
import model

lat = 0;
lon = 0;

response = requests.get('http://ip-api.com/json/')
data = response.json()

lat = data['lat']
lon = data['lon']

def setup_csv():
    data = {
        'temperature': [20, 15, 22, 10, 25],
        'humidity': [60, 80, 50, 90, 40],
        'weather': ['Clear', 'Rain', 'Clear', 'Snow', 'Clear'],
        'wind_speed': [3, 5, 2, 4, 1],
        'pressure': [1012, 1008, 1015, 1005, 1013],
        'go out (1=yes, 0=No)': [1, 0, 1, 0, 1]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('weather_dataset.csv', index=False)
    print(df)

def show_weather(weather):
    print(f"Time: {weather['time']}")
    print(f"Temperature: {weather['temperature']}°C")
    print(f"Humidity: {weather['humidity']}%")
    print(f"Weather: {weather['weather']}")
    print(f"Wind speed: {weather['wind_speed']}m/s")
    print(f"Pressure: {weather['pressure']}hPa")

def get_weather_forecast(city, api_key):
    url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=vi'
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

api_key = 'fe8d8c65cf345889139d8e545f57819a'
city = 'Long An'

def find_latest_weather(data):
    latest_entry = max(data['list'], key=lambda x: x['dt'])
    latest_time = datetime.fromtimestamp(latest_entry['dt'])
    
    return {
        "time": latest_time,
        "temperature": latest_entry['main']['temp'],
        "humidity": latest_entry['main']['humidity'],
        "weather": latest_entry['weather'][0]['main'],
        "wind_speed": latest_entry['wind']['speed'],
        "pressure": latest_entry['main']['pressure']
    }

def get_weather_features(weather, df):
    weather_dummies = df.columns[df.columns.str.startswith('weather_')]
    features = [0] * len(weather_dummies)
    if weather in weather_dummies:
        index = list(weather_dummies).index(f'weather_{weather}')
        features[index] = 1 
    return features

def generate_response(predictions, weather):
    if predictions == 1:
        return "Today is a good day to go out"
    else:
        response = "Stay at home today because it's not good to go out"
        if weather['temperature'] < 10:
            response += " and it's cold"
        if weather['weather'] == 'Rain':
            response += " and it's raining"
        if weather["wind_speed"] > 5:
            response += " and it's windy"
        return response

weather_data = get_weather_forecast(city, api_key)

setup_csv()


if weather_data:
    latest_weather = find_latest_weather(weather_data)
    
    weather_model = model.Model()
    weather_model.load_data()
    weather_model.train()
    
    weather_features = get_weather_features(latest_weather['weather'], weather_model.df)
    
    input_data = pd.DataFrame([[
        latest_weather['temperature'],
        latest_weather['humidity'],
        latest_weather['wind_speed'],
        latest_weather['pressure']
    ] + weather_features], columns=[
        'temperature', 'humidity', 'wind_speed', 'pressure'
    ] + list(weather_model.df.columns[weather_model.df.columns.str.startswith('weather_')]))
    
    print("------------ Latest weather ------------")
    show_weather(latest_weather)
    print("------------ Model prediction ------------")
    print("Should go out?")
    print(generate_response(weather_model.predict(input_data)[0], latest_weather))
    print("Score:")
    print(weather_model.score())
    
    weather_model.plot()
    
else:
    print("Can't get weather data")