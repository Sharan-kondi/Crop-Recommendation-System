from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from geopy.geocoders import ArcGIS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import datetime as dt

app = Flask(__name__)

BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
API_KEY = 'a31b5b3bcf6f180b7cc3eb5936ee4515'

def kelvin_to_celsius_fahrenheit(kelvin):
    celsius = kelvin - 273.15
    fahrenheit = celsius * (9/5) + 32
    return celsius, fahrenheit

def get_weather_data(city):
    url = BASE_URL + "appid=" + API_KEY + "&q=" + city
    response = requests.get(url).json()
    if 'main' not in response:
        return None
    else:
        temp_kelvin = response['main']['temp']
        temp_celsius, temp_fahrenheit = kelvin_to_celsius_fahrenheit(temp_kelvin)
        feels_like_kelvin = response['main']['feels_like']
        feels_like_celsius, feels_like_fahrenheit = kelvin_to_celsius_fahrenheit(feels_like_kelvin)
        weather_data = {
            'Temperature': temp_celsius,
            'Feels Like': feels_like_celsius,
            'Humidity': response['main']['humidity'],
            'Wind Speed': response['wind']['speed'],
            'Description': response['weather'][0]['description'],
            'Sunrise': dt.datetime.utcfromtimestamp(response['sys']['sunrise'] + response['timezone']),
            'Sunset': dt.datetime.utcfromtimestamp(response['sys']['sunset'] + response['timezone'])
        }
        return weather_data

def get_location_from_address(address):
    geolocator = ArcGIS()
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude), location.address
    else:
        return None, None

def find_district_knn(lat, lon, coordinates_df):
    knn = KNeighborsClassifier(n_neighbors=1)
    X = coordinates_df[['Latitude', 'Longitude']].values
    y = coordinates_df['District']
    knn.fit(X, y)
    district = knn.predict(np.array([[lat, lon]]))
    return district[0]

def train_crop_prediction_model():
    data = pd.read_csv('CROPP.csv')
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')
    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    address = request.form['address']
    crop_model = train_crop_prediction_model()
    soil_data_df = pd.read_csv('soil_data1.csv')
    coordinates_df = pd.read_csv('district_coordinates.csv')

    location, full_address = get_location_from_address(address)

    if location:
        latitude, longitude = location
        district = find_district_knn(latitude, longitude, coordinates_df)

        if district:
            soil_data = soil_data_df[soil_data_df['District'] == district]
            weather_data = get_weather_data(district)

            if not soil_data.empty:
                data = soil_data.iloc[0]
                
                features = pd.DataFrame({
                    'N': [data['N']],
                    'P': [data['P']],
                    'K': [data['K']],
                    'temperature': [weather_data['Temperature']],
                    'humidity': [weather_data['Humidity']],
                    'ph': [data['pH']],
                    'rainfall': [data['Rainfall']]
                })

                # Predict top 3 crops
                predictions = crop_model.predict_proba(features)[0]
                top_indices = np.argsort(predictions)[-3:][::-1]
                top_3_crops = [(crop_model.classes_[i], f'{crop_model.classes_[i].lower()}.jpg') for i in top_indices]
                
                crop_description = {
                    'Crop1': 'Most Suitable Crop',
                    'Crop2': 'Can Go For This As Well',
                    'Crop3': 'You Always Have This Option',
                    # Add descriptions for actual crops
                }

                return render_template('result.html', 
                                       address=full_address, 
                                       district=district, 
                                       ph=data['pH'],
                                       nitrogen=data['N'],
                                       phosphorus=data['P'],
                                       potassium=data['K'],
                                       rainfall=data['Rainfall'],
                                       temperature=weather_data['Temperature'],
                                       feels_like=weather_data['Feels Like'],
                                       humidity=weather_data['Humidity'],
                                       wind_speed=weather_data['Wind Speed'],
                                       description=weather_data['Description'],
                                       sunrise=weather_data['Sunrise'],
                                       sunset=weather_data['Sunset'],
                                       top_3_crops=dict(top_3_crops),
                                       crop_description=crop_description)
            else:
                return "No soil data available for your district."
        else:
            return "District not found."
    else:
        return "Location not found."
    
if __name__ == "__main__":
    app.run(debug=True)