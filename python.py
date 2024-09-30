import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import json

# Load data from JSON file
def load_data(disease_data):
    with open('disease_data.json', 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data['diseases'])

# Prepare the data
def prepare_data(df):
    df['symptoms'] = df['symptoms'].apply(lambda x: ' '.join(x))
    return df

# Load and prepare data
df = load_data('disease_data.json')
df = prepare_data(df)

# Features and labels
X = df['symptoms']
y = df['name']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create pipelines for different classifiers
def create_pipeline(model):
    return Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', model)
    ])

# Initialize classifiers
models = {
    'Naive Bayes': create_pipeline(MultinomialNB()),
    'Random Forest': create_pipeline(RandomForestClassifier(n_estimators=100)),
    'SVM': create_pipeline(SVC(kernel='linear'))
}

# Train and test each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"{name} predictions: {list(predictions)}")

# Example of predicting a new set of symptoms
example_symptoms = ["cold", "cough"]
for name, model in models.items():
    prediction = model.predict([' '.join(example_symptoms)])
    print(f"Prediction using {name} for symptoms {example_symptoms}: {prediction[0]}")
