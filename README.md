# 🎬 Sentiment Analysis using Bi-Directional RNN (Deep Learning)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


An end-to-end Deep Learning system for binary sentiment classification of movie reviews using a Bi-Directional Recurrent Neural Network (RNN), with complete text preprocessing, model training, evaluation, and real-time deployment through a Flask web application.

---

##  Project Overview

This project implements a complete end-to-end Sentiment Analysis system using Natural Language Processing (NLP) and Deep Learning.

The system classifies movie reviews as **Positive** or **Negative** using a **Bi-Directional Recurrent Neural Network (RNN)** built with TensorFlow and Keras.

The project covers the entire pipeline including:

- Data preprocessing
- Text cleaning and normalization
- Word embedding generation
- Deep learning model training
- Model evaluation
- Model saving
- Real-time deployment using Flask

---

##  Problem Statement

Understanding customer sentiment is critical for businesses and platforms that collect reviews.

The objective of this project is to:

- Analyze movie review text
- Convert raw text into numerical representations
- Train a deep learning model to classify sentiment
- Deploy the trained model for real-time predictions

---

##  Dataset

- **Dataset Used:** IMDB Movie Reviews Dataset
- **Source:** Public IMDB dataset (CSV format)
- **Total Used:** First 10,000 reviews
- **Target Labels:**
  - Positive → 1
  - Negative → 0

The dataset contains:

- Review text
- Sentiment label

---

##  Deep Learning Architecture

The model is built using a **Bi-Directional RNN** architecture.

### Model Layers:

- Embedding Layer
- Masking Layer
- Bi-Directional SimpleRNN (Layer 1)
- Bi-Directional SimpleRNN (Layer 2)
- Bi-Directional SimpleRNN (Layer 3)
- Dense Output Layer (Sigmoid activation)

### Why Bi-Directional RNN?

It processes text:

- Forward direction (left → right)
- Backward direction (right → left)

This helps capture contextual meaning more effectively.

---

##  Machine Learning Pipeline

### 1️. Data Preprocessing

- Load CSV dataset
- Select first 10,000 rows
- Map sentiment labels to binary (0/1)

### 2️. Text Cleaning

- Convert text to lowercase
- Remove punctuation
- Remove stopwords
- Apply lemmatization (WordNetLemmatizer)

### 3️. Text Vectorization

- One-hot encoding
- Vocabulary size: 10,000
- Padding sequences to fixed length
- Convert text into numerical tensors

### 4️. Data Splitting

- Training set: 7000 samples
- Validation set: 2000 samples
- Test set: 1000 samples

### 5️. Model Training

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 20
- Batch Size: 100
- Metric: Accuracy

### 6️. Model Saving

The trained model is saved as:
```
review.pkl
```

---

##  Deployment (Flask Web Application)

The trained model is deployed using Flask.

### Features:

- User enters a review
- Text is cleaned and processed
- Model predicts sentiment
- Displays:
  - Sentiment (Positive / Negative)
  - Confidence score

---

##  Project Structure
```
Sentiment_Analysis/
│
├── log_files/
├── templates/
│   └── index.html
├── main.py
├── RNN.py
├── embedding.py
├── lemmatization.py
├── log_code.py
├── app.py
├── test_model.py
├── review.pkl
├── requirements.txt
└── README.md
```

---

##  Tech Stack

### Programming Language

- Python

### Libraries & Tools

- NumPy
- Pandas
- NLTK
- TensorFlow / Keras
- Scikit-learn
- Flask
- Matplotlib

---

##  How to Run the Project

### 1️. Clone Repository
```bash
git clone https://github.com/your-username/Sentiment_Analysis.git
cd Sentiment_Analysis
```

### 2️. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️. Download NLTK Resources
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 4️. Train the Model
```bash
python main.py
```

### 5️. Run Flask App
```bash
python app.py
```

Open in browser:
```
http://127.0.0.1:5000/
```

---

##  Sample Prediction

**Input:**
```
The movie was absolutely amazing and emotionally powerful.
```

**Output:**
```
Positive (confidence: 0.92)
```

---

##  Applications

This system can be extended for:

- Movie review classification
- Product review analysis
- Customer feedback monitoring
- Opinion mining
- Social media sentiment analysis

---

##  Key Highlights

✔ End-to-end Deep Learning pipeline

✔ Custom text preprocessing implementation

✔ Bi-Directional RNN architecture

✔ Model serialization using Pickle

✔ Real-time web deployment with Flask

✔ Structured logging system

---

##  Business Impact

- Helps companies understand user sentiment
- Automates feedback classification
- Enables data-driven decision-making
- Reduces manual review analysis effort

---

## Author

**Hema Malini Gangumalla**
Aspiring Data Scientist | NLP & Deep Learning Enthusiast

📧 hemamalinig07@gmail.com

---

##  License

MIT License
