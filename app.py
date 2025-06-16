import joblib
import numpy as np
import re
from datetime import datetime
from textblob import TextBlob
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
import contextlib
import sqlite3
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from create_database import setup_database
from utils1 import login_required, set_session
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from flask import (
    Flask, render_template, 
    request, session, redirect
)

app = Flask(__name__)



database = "users.db"
setup_database(name=database)

app.secret_key = 'xpSm7p5bgJY8rNoBjGWiz5yjxM-NEBlW6SIBI62OkLc='

train_df = pd.read_csv('train.csv')
train_df.fillna(0, inplace=True)

# Convert binary columns
binary_cols = ['profile pic', 'private', 'name==username']
for col in binary_cols:
    train_df[col] = train_df[col].astype(int)

# Drop irrelevant column
if 'external URL' in train_df.columns:
    train_df.drop('external URL', axis=1, inplace=True)

# Features and target
X = train_df.drop('fake', axis=1)
y = train_df['fake']

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ====================
# Train models
# ====================
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Dictionary for easy model selection
model_map = {
    'xgb': xgb_model,
    'rf': rf_model,
    'knn': knn_model
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    # Set data to variables
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Attempt to query associated user data
    query = 'select username, password, email from users where username = :username'

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            account = conn.execute(query, {'username': username}).fetchone()

    if not account: 
        return render_template('login.html', error='Username does not exist')

    # Verify password
    try:
        ph = PasswordHasher()
        ph.verify(account[1], password)
    except VerifyMismatchError:
        return render_template('login.html', error='Incorrect password')

    # Check if password hash needs to be updated
    if ph.check_needs_rehash(account[1]):
        query = 'update set password = :password where username = :username'
        params = {'password': ph.hash(password), 'username': account[0]}

        with contextlib.closing(sqlite3.connect(database)) as conn:
            with conn:
                conn.execute(query, params)

    # Set cookie for user session
    set_session(
        username=account[0], 
        email=account[2], 
        remember_me='remember-me' in request.form
    )
    
    return redirect('/predict_page')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    # Store data to variables 
    password = request.form.get('password')
    confirm_password = request.form.get('confirm-password')
    username = request.form.get('username')
    email = request.form.get('email')

    # Verify data
    if len(password) < 8:
        return render_template('register.html', error='Your password must be 8 or more characters')
    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match')
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return render_template('register.html', error='Username must only be letters and numbers')
    if not 3 < len(username) < 26:
        return render_template('register.html', error='Username must be between 4 and 25 characters')

    query = 'select username from users where username = :username;'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, {'username': username}).fetchone()
    if result:
        return render_template('register.html', error='Username already exists')

    # Create password hash
    pw = PasswordHasher()
    hashed_password = pw.hash(password)

    query = 'insert into users(username, password, email) values (:username, :password, :email);'
    params = {
        'username': username,
        'password': hashed_password,
        'email': email
    }

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, params)

    # We can log the user in right away since no email verification
    set_session( username=username, email=email)
    return redirect('/')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input values
        input_data = [
            int(request.form['profile_pic']),
            float(request.form['nums_length_username']),
            int(request.form['fullname_words']),
            float(request.form['nums_length_fullname']),
            int(request.form['name_equals_username']),
            int(request.form['description_length']),
            int(request.form['private']),
            int(request.form['posts']),
            int(request.form['followers']),
            int(request.form['follows'])
        ]

        model_key = request.form.get('model', 'xgb')
        model = model_map.get(model_key, xgb_model)

        # Scale the input
        input_scaled = scaler.transform([input_data])

        # Prediction & probabilities
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        fake_prob = round(proba[1] * 100, 2)
        real_prob = round(proba[0] * 100, 2)

        result_label = "Fake Account ❌" if prediction == 1 else "Genuine Account ✅"

        return render_template('result.html',
                               prediction=result_label,
                               fake_prob=fake_prob,
                               real_prob=real_prob,
                               model_name=model_key.upper())

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
