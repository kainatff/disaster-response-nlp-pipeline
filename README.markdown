# Disaster Response NLP Pipeline Project

A **Natural Language Processing (NLP)** web app that classifies real disaster messages into multiple categories such as water, food, medical aid, etc., helping emergency teams prioritize and respond more effectively.

## Overview

This project trains a machine learning model to classify disaster response messages. It uses data from real-world events and is designed to support disaster relief efforts by automating the categorization of incoming text messages.

## How It Works

- Preprocesses raw messages and categories from CSV files.
- Builds a machine learning pipeline using NLP (tokenization, lemmatization) + a `RandomForestClassifier` wrapped in `MultiOutputClassifier`.
- Saves the trained model as `classifier.pkl`.
- Loads the model in a Flask web app where users can enter messages and get instant predictions for 36+ categories.

## 🛠️ Instructions

1. **Run the ETL pipeline**

   This step cleans the data and stores it in a SQLite database:

   ```bash
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```

2. **Train the model**

   This trains the model and saves it as `classifier.pkl` in the `models` directory ..

   ```bash
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

3. **Launch the web app**

   Go to the `app` folder and run:

   ```bash
   python run.py
   ```

   Then open your browser and visit:
   👉 **[http://0.0.0.0:3001/](http://0.0.0.0:3001/)**

## Web App Features

- Input any disaster-related message.
- Get predicted categories like:
  - `request`, `food`, `water`, `shelter`, `medical_help`, etc.
- Visualizes message genre distribution (news, direct, social).
- Shows category distribution per genre.

## ML Model Details

- **Pipeline:** `CountVectorizer` → `TfidfTransformer` → `MultiOutputClassifier(RandomForest)`
- **GridSearchCV** used for tuning hyperparameters
- **Custom Tokenizer** with NLTK lemmatization
- **36 output labels** predicted in a single pass

## Requirements

- Python 3.12+
- Pandas, NumPy, Scikit-learn, NLTK, SQLAlchemy, Flask, Plotly, joblib

<img src="./images/2.png" width="90%">

<img src="./images/1.png" width="90%">
