from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import requests  # To call APIs
import pandas
import numpy
from datetime import datetime
import joblib
from geopy.geocoders import ArcGIS
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.secret_key = '9448857746@Rvs'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'rvs'
app.config['MYSQL_PASSWORD'] = '9448857746@Rvs'
app.config['MYSQL_DB'] = 'agricore'

mysql = MySQL(app)

WEATHER_API_KEY = '11ed7f0401532c536b55b657c3873adc607ca988c69aaaa1896b8eea129ef026'
WEATHER_BASE_URL = "https://api.ambeedata.com/weather/latest/by-lat-lng?"

def get_weather_by_coordinates(lat, lon):
    """
    Fetch weather data using the Ambee Weather API based on coordinates.
    """
    headers = {
        'x-api-key': WEATHER_API_KEY
    }
    url = f"{WEATHER_BASE_URL}lat={lat}&lng={lon}"
    response = requests.get(url, headers=headers).json()

    if 'data' not in response:
        return None

    weather_data = response['data']
    
    # Convert ISO 8601 timestamp to MySQL DATETIME format
    updated_at_iso = weather_data.get('updatedAt')
    updated_at = None
    if updated_at_iso:
        updated_at = datetime.strptime(updated_at_iso, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d %H:%M:%S')

    return {
        'Timezone': weather_data.get('timezone'),
        'Country Code': weather_data.get('country_code'),
        'Apparent Temperature': weather_data.get('apparentTemperature'),
        'Cloud Cover': weather_data.get('cloudCover'),
        'Dew Point': weather_data.get('dewPoint'),
        'Humidity': weather_data.get('humidity'),
        'Pressure': weather_data.get('pressure'),
        'Precipitation Intensity': weather_data.get('precipIntensity'),
        'Temperature': weather_data.get('temperature'),
        'Visibility': weather_data.get('visibility'),
        'Wind Gust': weather_data.get('windGust'),
        'Ozone': weather_data.get('ozone'),
        'UV Index': weather_data.get('uvIndex'),
        'Wind Speed': weather_data.get('windSpeed'),
        'Wind Bearing': weather_data.get('windBearing'),
        'Icon': weather_data.get('icon'),
        'Summary': weather_data.get('summary'),
        'Updated At': updated_at
    }

def save_weather_data_to_db(location_name, latitude, longitude, weather_data):
    """
    Save weather data into the database.
    """
    cursor = mysql.connection.cursor()

    query = """
        INSERT INTO weather (
            location_name, latitude, longitude, timezone, country_code, 
            apparent_temperature, cloud_cover, dew_point, humidity, pressure, 
            precipitation_intensity, temperature, visibility, wind_gust, ozone, 
            uv_index, wind_speed, wind_bearing, icon, summary, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
        location_name, latitude, longitude,
        weather_data['Timezone'], weather_data['Country Code'],
        weather_data['Apparent Temperature'], weather_data['Cloud Cover'],
        weather_data['Dew Point'], weather_data['Humidity'], weather_data['Pressure'],
        weather_data['Precipitation Intensity'], weather_data['Temperature'],
        weather_data['Visibility'], weather_data['Wind Gust'], weather_data['Ozone'],
        weather_data['UV Index'], weather_data['Wind Speed'], weather_data['Wind Bearing'],
        weather_data['Icon'], weather_data['Summary'], weather_data['Updated At']
    ))

    mysql.connection.commit()
    cursor.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username exists
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()

        if not user:
            flash("User is not registered. Please register.", "danger")
            return redirect(url_for('login'))

        # Access the stored hashed password (assuming password is the 6th column)
        stored_password = user[6]  # Adjust index if necessary

        # Verify password
        if not check_password_hash(stored_password, password):
            flash("Password is incorrect. Please retry.", "danger")
            return redirect(url_for('login'))

        # Successful login
        session['user_id'] = user[0]  # Assuming 'id' is the 1st column
        session['username'] = user[2]  # Assuming 'username' is the 3rd column
        session['role'] = user[7]  # Assuming 'role' is the 8th column

        if user[7] == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('farmer_dashboard'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        phone = request.form['phone']
        gender = request.form['gender']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        role = 'farmer'

        # Check if username already exists
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Username is already taken. Please choose a different one.', 'danger')
            return redirect(url_for('register'))

        # If username is available, insert new user
        cursor.execute("INSERT INTO user (name, username, phone_number, gender, email, password, role) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                       (name, username, phone, gender, email, password, role))
        mysql.connection.commit()
        cursor.close()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'username' in session and session['role'] == 'admin':
        # Fetch the admin's name from the database
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT name FROM user WHERE id = %s", (session['user_id'],))
        admin_name = cursor.fetchone()[0]  # Assuming 'name' is the first column
        cursor.close()

        # Pass the admin's name to the template
        return render_template('admin_dashboard.html', admin_name=admin_name)
    else:
        return redirect(url_for('login'))

@app.route('/farmer_dashboard')
def farmer_dashboard():
    if 'username' in session and session['role'] == 'farmer':
        return render_template('farmer_dashboard.html')
    else:
        return redirect(url_for('login'))

@app.route('/add_price', methods=['GET', 'POST'])
def add_price():
    if 'username' in session and session['role'] == 'admin':
        if request.method == 'POST':
            crop_name = request.form['crop_name']
            quantity = request.form['quantity']
            price = request.form['price']
            
            try:
                cursor = mysql.connection.cursor()
                cursor.execute("INSERT INTO market_price (crop_name, quantity, price) VALUES (%s, %s, %s)",
                               (crop_name, quantity, price))
                mysql.connection.commit()
                flash('Price added successfully!', 'success')
            except Exception as e:
                mysql.connection.rollback()
                flash('Failed to add price. Please try again.', 'error')
        
        return render_template('add_price.html')
    else:
        return redirect(url_for('login'))

@app.route('/update_crop', methods=['GET', 'POST'])
def update_crop():
    if 'username' in session and session['role'] == 'admin':
        if request.method == 'POST':
            crop_name = request.form['crop_name']
            n = request.form['n']
            p = request.form['p']
            k = request.form['k']
            temperature = request.form['temperature']
            humidity = request.form['humidity']
            rainfall = request.form['rainfall']
            soil_pH = request.form['soil_ph']
            soil_type = request.form['soil_type']

            cursor = mysql.connection.cursor()

            # Check if crop already exists
            cursor.execute("SELECT * FROM crop WHERE crop_name = %s", (crop_name,))
            existing_crop = cursor.fetchone()

            if existing_crop:
                # Update existing crop
                cursor.execute(""" 
                    UPDATE crop 
                    SET N=%s, P=%s, K=%s, temperature=%s, humidity=%s, rainfall=%s, soil_pH=%s, soil_type=%s 
                    WHERE crop_name=%s
                """, (n, p, k, temperature, humidity, rainfall, soil_pH, soil_type, crop_name))
                flash('Crop information updated successfully!', 'success')
            else:
                # Insert new crop
                cursor.execute(""" 
                    INSERT INTO crop (crop_name, N, P, K, temperature, humidity, rainfall, soil_pH, soil_type) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (crop_name, n, p, k, temperature, humidity, rainfall, soil_pH, soil_type))
                flash('New crop added successfully!', 'success')

            mysql.connection.commit()
            cursor.close()

        return render_template('update_crop.html')

    return redirect(url_for('login'))

@app.route('/view_farmers')
def view_farmers():
    if 'username' in session and session['role'] == 'admin':
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id, name, username, phone_number, gender, email FROM user WHERE role='farmer'")
        farmers = cursor.fetchall()
        cursor.close()

        if farmers:
            return render_template('view_farmers.html', farmers=farmers)
        else:
            flash('No farmers found.', 'info')
            return render_template('view_farmers.html', farmers=None)

    return redirect(url_for('login'))

@app.route('/crop_info', methods=['GET', 'POST'])
def crop_info():
    if 'username' in session and session['role'] == 'farmer':
        if request.method == 'POST':
            crop_name = request.form['crop_name']
            
            # Query the database for crop details based on the entered crop name
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT * FROM crop WHERE crop_name = %s", [crop_name])
            crop = cursor.fetchone()
            
            # If no crop is found, set an empty crop details (None)
            if not crop:
                crop_details = None
            else:
                # Map the fetched data to meaningful variable names based on your table's structure
                crop_details = {
                    'name': crop[1],  # crop_name is at index 1
                    'nitrogen': crop[2],  # N is at index 2
                    'phosphorus': crop[3],  # P is at index 3
                    'potassium': crop[4],  # K is at index 4
                    'temperature': crop[5],  # temperature is at index 5
                    'humidity': crop[6],  # humidity is at index 6
                    'rainfall': crop[7],  # rainfall is at index 7
                    'soil_ph': crop[8],  # soil_pH is at index 8
                    'soil_type': crop[9]  # soil_type is at index 9
                }

            return render_template('crop_info.html', crop_details=crop_details)

        # If the method is GET, just render the form without any data
        return render_template('crop_info.html')

    else:
        return redirect(url_for('login'))


@app.route('/crop_prices', methods=['GET', 'POST'])
def crop_prices():
    if 'username' in session and session['role'] == 'farmer':
        if request.method == 'POST':
            crop_name = request.form['crop_name']
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT * FROM market_price WHERE crop_name = %s", [crop_name])
            prices = cursor.fetchall()
            return render_template('crop_prices.html', prices=prices, crop_name=crop_name)
        return render_template('crop_prices.html')

    else:
        return redirect(url_for('login'))

@app.route('/location_info', methods=['GET', 'POST'])
def location_info():
    if 'username' not in session or session['role'] != 'farmer':
        return redirect(url_for('login'))

    error_message = None  # Initialize error_message as None

    if request.method == 'POST':
        location_name = request.form['location_name']

        # Fetch district data from the database
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT latitude, longitude FROM district WHERE name = %s", (location_name,))
        district_data = cursor.fetchone()
        cursor.close()

        if district_data:
            latitude, longitude = district_data

            # Fetch weather data
            weather_data = get_weather_by_coordinates(latitude, longitude)

            if weather_data:
                # Save weather data to the database
                save_weather_data_to_db(location_name, latitude, longitude, weather_data)

                return render_template(
                    'location_info.html',
                    location_name=location_name,
                    weather_data=weather_data
                )
            else:
                flash("Unable to fetch weather data. Please try again later.", "danger")
        else:
            error_message = "District not found in the database."  # Set error message if district is not found

    return render_template('location_info.html', error_message=error_message)  # Pass error_message to the template

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

# Load the crop recommendation model
crop_model = joblib.load('crop_recommendation_model.pkl')

# Weather API configuration
BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
API_KEY = 'a31b5b3bcf6f180b7cc3eb5936ee4515'

# Function to convert Kelvin to Celsius
def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

# Function to fetch weather data
def get_weather_data(city):
    url = BASE_URL + "appid=" + API_KEY + "&q=" + city
    response = requests.get(url).json()

    if 'main' not in response:
        return None
    else:
        temp_kelvin = response['main']['temp']
        temp_celsius = kelvin_to_celsius(temp_kelvin)

        weather_data = {
            'Temperature': temp_celsius,
            'Humidity': response['main']['humidity'],
            'Description': response['weather'][0]['description']
        }
        return weather_data

# Function to get latitude and longitude from location
def get_location_from_address(address):
    geolocator = ArcGIS()
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude), location.address
    else:
        return None, None

# Function to find district using KNN (if needed)
def find_district_knn(lat, lon, coordinates_df):
    knn = KNeighborsClassifier(n_neighbors=1)
    X = coordinates_df[['Latitude', 'Longitude']].values
    y = coordinates_df['District']
    knn.fit(X, y)
    district = knn.predict(numpy.array([[lat, lon]]))
    return district[0]

# Route for crop recommendation
@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if 'username' not in session or session['role'] != 'farmer':
        flash("You need to be logged in as a farmer to access this page.", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        location_name = request.form['location_name']

        # Fetch latitude and longitude from location
        location, full_address = get_location_from_address(location_name)
        if not location:
            flash("Location not found. Please enter a valid location.", "danger")
            return render_template('crop_recommendation.html')

        latitude, longitude = location

        # Fetch weather data
        weather_data = get_weather_data(location_name)
        if not weather_data:
            flash("Unable to fetch weather data. Please try again later.", "danger")
            return render_template('crop_recommendation.html')

        # Fetch soil data (assuming soil data is stored in a CSV file)
        soil_data_df = pandas.read_csv('DBMYes/soil_data1.csv')
        district = find_district_knn(latitude, longitude, pandas.read_csv('DBMYes/district_coordinates.csv'))
        soil_data = soil_data_df[soil_data_df['District'] == district]

        if soil_data.empty:
            flash("No soil data available for this location.", "danger")
            return render_template('crop_recommendation.html')

        soil_data = soil_data.iloc[0]

        # Prepare data for prediction
        features = pandas.DataFrame({
            'N': [soil_data['N']],
            'P': [soil_data['P']],
            'K': [soil_data['K']],
            'temperature': [weather_data['Temperature']],
            'humidity': [weather_data['Humidity']],
            'ph': [soil_data['pH']],
            'rainfall': [soil_data['Rainfall']]
        })

        # Predict the top three crops
        probabilities = crop_model.predict_proba(features)
        top_three_indices = probabilities[0].argsort()[-3:][::-1]
        top_three_crops = [crop_model.classes_[i] for i in top_three_indices]

        return render_template('crop_recommendation.html', crops=top_three_crops, location=full_address)

    return render_template('crop_recommendation.html')

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'username' not in session:
        flash("You need to log in to access this page.", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_id = session['user_id']
        name = request.form['name']
        username = request.form['username']
        phone = request.form['phone']
        gender = request.form['gender']
        email = request.form['email']
        current_password = request.form['current_password']
        new_password = request.form['new_password']

        cursor = mysql.connection.cursor()

        # Fetch the current user data
        cursor.execute("SELECT * FROM user WHERE id = %s", (user_id,))
        user = cursor.fetchone()

        if not user:
            flash("User not found.", "danger")
            return redirect(url_for('edit_profile'))

        # Verify current password
        if not check_password_hash(user[6], current_password):  # Assuming password is the 6th column
            flash("Current password is incorrect.", "danger")
            return redirect(url_for('edit_profile'))

        # Check if the new username is already taken by another user
        cursor.execute("SELECT * FROM user WHERE username = %s AND id != %s", (username, user_id))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Username is already taken. Please choose a different one.", "danger")
            return redirect(url_for('edit_profile'))

        # Update user data
        update_query = """
            UPDATE user 
            SET name = %s, username = %s, phone_number = %s, gender = %s, email = %s 
            WHERE id = %s
        """
        cursor.execute(update_query, (name, username, phone, gender, email, user_id))

        # Update password if a new one is provided
        if new_password:
            hashed_password = generate_password_hash(new_password)
            cursor.execute("UPDATE user SET password = %s WHERE id = %s", (hashed_password, user_id))

        mysql.connection.commit()
        cursor.close()

        # Flash a success message and redirect to the login page
        flash("Profile updated successfully! Please log in again.", "success")
        return redirect(url_for('login'))

    # Fetch current user data for pre-filling the form
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM user WHERE id = %s", (session['user_id'],))
    user = cursor.fetchone()
    cursor.close()

    # Pass the user's role to the template
    return render_template('edit_profile.html', user=user, role=session.get('role'))

if __name__ == '__main__':
    app.run(debug=True)