import os
import pickle
import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

MODEL_FILE = 'model.pkl'
VEC_FILE = 'vectorizer.pkl'


# Load or train model
def train_model():
    df = pd.read_csv('IMDB Dataset.csv')
    X = df['review']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    # Save for faster restarts
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(VEC_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    return model, vectorizer


# Load saved model if exists, otherwise train
if os.path.exists(MODEL_FILE) and os.path.exists(VEC_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(VEC_FILE, 'rb') as f:
        vectorizer = pickle.load(f)
else:
    model, vectorizer = train_model()


@app.route('/')
def home():
    return render_template('index.html', review='', prediction_text=None)


@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    if not review.strip():
        return render_template('index.html', review=review, prediction_text='Please enter a review.')

    # Vectorize and predict
    vec = vectorizer.transform([review])
    pred = model.predict(vec)[0]
    result = 'positive' if pred == 1 else 'negative'
    return render_template('index.html', review=review, prediction_text=f'The review is {result}.')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))