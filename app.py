from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import joblib
import numpy as np

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management and flash messages

# Load the trained model
try:
    model = joblib.load('xgboost_model.pkl')  # Ensure this file is in the same directory
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Fixed admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"

@app.route('/')
def home_page():
    return render_template('home.html')  # Render the home.html page

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate credentials
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            flash("Login successful!", "success")
            return redirect(url_for('predict'))  # Redirect to prediction page after login
        else:
            flash("Invalid username or password. Please try again.", "danger")
    
    return render_template('login.html')  # Render the login.html page

@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Remove user session
    flash("You have been logged out.", "info")
    return render_template('logout.html')  # Render the logout page

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('logged_in'):
        flash("You must log in to access the prediction page.", "warning")
        return redirect(url_for('login_page'))  # Redirect to login if not logged in

    if request.method == 'POST':
        try:
            # Extract input values from the form
            feature1 = float(request.form.get('Age', 0))
            feature2 = float(request.form.get('Insulin', 0))
            feature3 = float(request.form.get('BMI', 0))
            feature4 = float(request.form.get('Diabetes', 0))
            feature5 = float(request.form.get('Blood_Pressure', 0))

            # Prepare features for prediction
            features = np.array([[feature1, feature2, feature3, feature4, feature5]])

            # Check if the model is loaded
            if model is None:
                raise Exception("Model not loaded")

            # Make the prediction
            prediction = model.predict(features)

            # Return prediction result
            return render_template(
                'result.html',
                prediction_text=f"Predicted Blood Glucose Level: {prediction[0]}"
            )

        except ValueError:
            return jsonify({"error": "Invalid input. Please enter numeric values."}), 400
        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500

    return render_template('index.html')  # Render the prediction input form

if __name__ == "__main__":
    app.run(port=5001 , debug=True)
