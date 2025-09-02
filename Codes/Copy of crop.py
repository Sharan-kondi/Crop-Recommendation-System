import pandas as pd
import numpy as np
from geopy.geocoders import ArcGIS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
import datetime as dt

BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
API_KEY = 'a31b5b3bcf6f180b7cc3eb5936ee4515'

def kelvin_to_celsius_fahrenheit(kelvin):
    """Convert temperature from Kelvin to Celsius and Fahrenheit."""
    celsius = kelvin - 273.15
    fahrenheit = celsius * (9/5) + 32
    return celsius, fahrenheit

def get_weather_data(city):
    """Fetch weather data from OpenWeatherMap API."""
    url = BASE_URL + "appid=" + API_KEY + "&q=" + city
    response = requests.get(url).json()

    if 'main' not in response:
        print(f"Error: Unable to fetch weather data for {city}")
        print(f"API Error Message: {response.get('message', 'Unknown error')}")
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
    """Get latitude and longitude from address using geopy."""
    geolocator = ArcGIS()
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude), location.address
    else:
        return None, None

def find_district_knn(lat, lon, coordinates_df):
    """Find the nearest district using K-Nearest Neighbors."""
    knn = KNeighborsClassifier(n_neighbors=1)
    X = coordinates_df[['Latitude', 'Longitude']].values
    y = coordinates_df['District']
    knn.fit(X, y)
    district = knn.predict(np.array([[lat, lon]]))
    return district[0]

def train_crop_prediction_model():
    """Train the crop prediction model."""
    # Load the dataset
    data = pd.read_csv('CROPP.csv')

    # Separate features and target
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print('Confusion Matrix:')
    print(conf_matrix)

    return model

def main():
    # Train the crop prediction model
    crop_model = train_crop_prediction_model()

    # Read the CSV files
    soil_data_df = pd.read_csv('/Users/510msqkm/Documents/4th sem/AIML/soil_data1.csv')
    coordinates_df = pd.read_csv('/Users/510msqkm/Documents/4th sem/AIML/district_coordinates.csv')

    address = input("Enter your address: ")
    location, full_address = get_location_from_address(address)

    if location:
        latitude, longitude = location
        district = find_district_knn(latitude, longitude, coordinates_df)

        if district:
            soil_data = soil_data_df[soil_data_df['District'] == district]
            weather_data = get_weather_data(district)

            if not soil_data.empty:
                data = soil_data.iloc[0]
                print(f"Your Location: {full_address}")
                print(f"District: {district}")
                print(f"ph: {data['pH']}")
                print(f"Nitrogen (N): {data['N']}")
                print(f"Phosphorus (P): {data['P']}")
                print(f"Potassium (K): {data['K']}")
                print(f"Rainfall: {data['Rainfall']} mm")

                if weather_data:
                    print(f"Temperature: {weather_data['Temperature']:.2f}°C")
                    print(f"Feels like: {weather_data['Feels Like']:.2f}°C")
                    print(f"Humidity: {weather_data['Humidity']}%")
                    print(f"Wind Speed: {weather_data['Wind Speed']} m/s")
                    print(f"Weather: {weather_data['Description']}")
                    print(f"Sunrise time: {weather_data['Sunrise']}")
                    print(f"Sunset time: {weather_data['Sunset']}")
                else:
                    print("Weather data not available.")

                # Prepare data for crop prediction
                features = pd.DataFrame({
                    'N': [data['N']],
                    'P': [data['P']],
                    'K': [data['K']],
                    'temperature': [weather_data['Temperature']],
                    'humidity': [weather_data['Humidity']],
                    'ph': [data['pH']],
                    'rainfall': [data['Rainfall']]
                })

                # Predict the crop
                new_prediction = crop_model.predict(features)
                print(f'The predicted crop for the input values is: {new_prediction[0]}')
            else:
                print("No soil data available for your district.")
        else:
            print("District not found.")
    else:
        print("Location not found.")

if __name__ == "__main__":
    main()