# Bank Complaint Classification System

## Overview
This project is a Django-based Machine Learning web application that classifies banking customer complaints into predefined categories and provides recommended next steps.

The system uses Machine Learning and Natural Language Processing (NLP) techniques to analyze complaint text and predict the appropriate complaint category.

## Features

- Automatic complaint classification
- Machine Learning based prediction
- TF-IDF text vectorization
- Stacking Classifier model
- Rule-based prediction enhancement
- Recommended next-step suggestions
- User-friendly web interface
- Prediction logging

## Tech Stack

- Python
- Django
- Scikit-learn
- NumPy
- SciPy
- Joblib
- HTML/CSS

## Project Structure

```text
BankComplaintClassifier/
│
├── bankml/
│   ├── views.py
│   ├── urls.py
│   ├── models.py
│   └── templates/
│
├── mlbank/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── models/
│   ├── stacking_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── logs/
│   └── predictions.csv
│
├── manage.py
├── requirements.txt
└── README.md
```

## How It Works

1. User enters a banking complaint.
2. Complaint text is preprocessed.
3. TF-IDF converts text into numerical features.
4. Stacking Classifier predicts the complaint category.
5. Rule-based logic improves prediction accuracy.
6. Suggested actions are displayed to the user.

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Server

```bash
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Machine Learning Pipeline

- Text Cleaning
- Tokenization
- TF-IDF Vectorization
- Stacking Classifier Prediction
- Rule-Based Optimization
- Category Recommendation

## Future Improvements

- REST API Integration
- User Authentication
- Admin Dashboard
- Analytics and Reporting
- Cloud Deployment
- Real-Time Complaint Tracking

## Author

**Mithun Karthik Baskar**

Final Year CSE Student  
Interested in Machine Learning, Backend Development, and Problem Solving.

## License

This project is for educational and research purposes.
