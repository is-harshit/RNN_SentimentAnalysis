import streamlit as st
from keras.src.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation,SimpleRNN, Bidirectional, BatchNormalization,LSTM
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.callbacks import EarlyStopping
import json


maxlen=29

with open('tokenizer_weight.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def compatibilate(text):
    a = [text]

    # Convert the text to sequences of integers using the tokenizer
    a = tokenizer.texts_to_sequences(a)

    # Pad the sequences to ensure they have the same length
    a = pad_sequences(a, padding='post', maxlen=maxlen)

    # Reshape the array to fit the model input shape
    a = np.array(a)
    a = a.reshape((a.shape[0], a.shape[1], 1))  # Reshape to (1, maxlen, 1)

    return a

def lstm_rnn_model():
    model = Sequential()
    # Using LSTM for improved sequence modeling
    model.add(LSTM(100, input_shape=(maxlen, 1), return_sequences=False))  # LSTM with 100 units
    model.add(Dense(50, activation='relu'))  # Intermediate layer with ReLU activation
    model.add(Dense(25, activation='relu'))  # Another ReLU layer for additional processing
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

    # Configure the optimizer with a commonly used learning rate
    adam = optimizers.Adam(learning_rate=0.001)  # Optimizer with a learning rate of 0.001
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

# Initialize the model
def initialise():

    model = lstm_rnn_model()
    model.load_weights("./sentiment_weights_lstmFin.h5")
    return model

threshold= 0.6014326


def predict_sentiment(text):
    model=initialise()
    # This function takes in text and returns the sentiment
    # Here you need to adapt the function depending on how your model processes the input
    prediction = model.predict([text])[0]  # Adjust this line based on how your model expects input
    print(prediction)
    return "Positive" if prediction > threshold else "Negative"

# Streamlit interface
def main():
    st.title("Sentiment Analysis App")
    user_input = st.text_area("Enter Text Here:", "Type here...")
    user_input=compatibilate(user_input)
    if st.button("Analyze"):
        sentiment = predict_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")

if __name__ == '__main__':
    main()