
AgriCore: An AI-Powered Crop and Weather Recommendation System 🌾
Project Overview
AgriCore is a comprehensive, full-stack web application designed to empower farmers with data-driven insights for improved agricultural productivity. The system uses a machine learning model to recommend the most suitable crops based on a farmer's location, soil type, and real-time weather conditions. Additionally, it provides a secure user management system for both farmers and administrators, offering features such as market price updates, crop information, and profile management. This project serves as a practical demonstration of how AI and web technologies can be integrated to create valuable solutions in the agriculture sector.

🚀 Features
AI-Powered Crop Recommendation: The core feature is a RandomForestClassifier model that predicts the top three most suitable crops by analyzing environmental factors like Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall.

Real-time Weather & Location Data: The application fetches and displays up-to-date weather data from the OpenWeatherMap API. It uses geopy and a KNeighborsClassifier to determine the user's district based on their address.

Secure User Management: A robust system with separate dashboards for farmers and administrators, allowing for secure registration, login, and profile management using werkzeug.security for password hashing.

Database Integration: Utilizes a MySQL database to store critical information such as user details, market prices, and crop data.

Market Price Updates: The admin panel provides functionality to add and update market prices for various crops, which farmers can then view.

Model Persistence: The trained machine learning model is saved using joblib to ensure fast and efficient deployment within the web application.

🛠️ Technologies Used
Backend Framework: Flask (Python)

Machine Learning: scikit-learn, RandomForestClassifier, KNeighborsClassifier, joblib

Data Handling: pandas, numpy

APIs: OpenWeatherMap API, Ambee Weather API, geopy library for location services, requests for API calls

Database: MySQL, Flask-MySQLdb

Frontend: HTML, CSS, Bootstrap

⚙️ Setup and Installation
Follow these steps to get the AgriCore application up and running on your local machine.

Prerequisites
Python 3.x

pip (Python package installer)

A MySQL database instance

Dataset Information 📚
This project uses CSV files to provide the necessary data for the crop prediction model and location-based lookups. The relevant data files are located in the DBMYes folder and include CROPP.csv, district_coordinates.csv, and soil_data1.csv.

1. Install Dependencies 📦
It is highly recommended to use a virtual environment to avoid conflicts.

Bash

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux
source venv/bin/activate
# On Windows
.\venv\Scripts\activate

# Install the required Python packages
pip install flask Flask-MySQLdb pandas numpy geopy scikit-learn requests joblib werkzeug
2. Run the Application ▶️
Once you have the dependencies installed and have set up your MySQL database with the credentials specified in app.py, you can launch the Flask application.

Bash

python app.py
The application will start, and you can access it by opening your web browser and navigating to http://127.0.0.1:5000/.

💬 Usage
Registration and Login: Access the application's login page to register as a new user or log in with existing credentials.

Farmer Dashboard: From the farmer's dashboard, you can view crop recommendations by entering your location, look up detailed information about a specific crop, and check market prices.

Admin Dashboard: As an admin, you have the ability to manage users (view farmer accounts) and update crop information and market prices.

Crop Recommendation: Navigate to the "Crop Recommendation" page, input your location, and the system will provide a list of the top three crops suitable for your area based on soil and weather conditions.

