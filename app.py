import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import csv

def clean_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            try:
                writer.writerow(row)
            except Exception as e:
                st.write(f"Skipping line due to error: {e}")

# Load and clean data
clean_csv("Train.csv", "data/Cleaned_Train.csv")
clean_csv("Valid.csv", "data/Cleaned_Valid.csv")
clean_csv("Test.csv", "data/Cleaned_Test.csv")

# Load pre-cleaned datasets or allow users to upload their datasets
def load_data():
    train_df = pd.read_csv("data/Cleaned_Train.csv")
    valid_df = pd.read_csv("data/Cleaned_Valid.csv")
    test_df = pd.read_csv("data/Cleaned_Test.csv")
    
    return train_df, valid_df, test_df

def preprocess_data(df):
    # Drop missing labels and ensure label types are integers
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    return df

@st.cache(allow_output_mutation=True)
def train_model(train_data, valid_data):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    
    # Set up the pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', MultinomialNB())
    ])
    
    # Set up Grid Search for hyperparameter tuning
    param_grid = {
        'vectorizer__max_df': [0.5, 0.7],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__alpha': [0.1, 1]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(train_data['text'], train_data['label'])
    
    return grid_search.best_estimator_

# App title
st.title("Sentiment Analysis using Naive Bayes")

# Allow users to upload their own data or use preloaded data
train_df, valid_df, test_df = load_data()
train_df = preprocess_data(train_df)
valid_df = preprocess_data(valid_df)

# Train the model and get the best estimator
model = train_model(train_df.sample(frac=0.1, random_state=42), valid_df.sample(frac=0.1, random_state=42))

# Input for user to test the model
input_text = st.text_area('Enter text to analyze:')

if st.button('Analyze'):
    if input_text:
        # Vectorize and predict sentiment
        prediction = model.predict([input_text])[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        st.write(f'Sentiment: **{sentiment}**')
    else:
        st.write("Please enter some text.")

# Display model performance on validation set
st.subheader("Validation Accuracy")
valid_predictions = model.predict(valid_df['text'])
valid_accuracy = (valid_predictions == valid_df['label']).mean()
st.write(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")
