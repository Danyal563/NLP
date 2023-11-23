import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset without caching
def load_data():
    url = "tweet_emotions.csv"
    df = pd.read_csv(url)  # replace with the actual link to your dataset
    return df


# Train the emotion detection model
def train_model(df):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['sentiment'], test_size=0.2, random_state=42
    )

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train a Logistic Regression classifier
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    return model, vectorizer


# Streamlit UI
def main():
    st.title("Emotion Detection Model Training App")

    # Load the dataset without caching
    df = load_data()


    # Train the model
    st.subheader("Training the Model:")
    model, vectorizer = train_model(df)
    st.success("Model trained successfully!")

    # User input for prediction
    st.subheader("Make a Prediction:")
    user_input = st.text_input("Enter a tweet:")

    if user_input:
        # Vectorize user input using the same vectorizer
        user_input_vectorized = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(user_input_vectorized)[0]

        st.success(f"Predicted Emotion: {prediction}")


if __name__ == "__main__":
    main()
