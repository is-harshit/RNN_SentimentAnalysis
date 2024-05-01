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
from groq import Groq


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


def llama_response(msg,mode=0):
    client = Groq(
        api_key="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",
    )

    if mode==0:
       te="give the response of this sentence as positive or negative only: " + msg
    else:
        te= "what is ur take on my statement? " +msg
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": te,
            }
        ],
        model="llama3-70b-8192",
    )

    sent=chat_completion.choices[0].message.content

    return sent
    # if sent.upper().contains("POS"):
    #    return "Positive"
    # else:
    #     return "Negative"

def predict_sentiment(text):
    # model=initialise()


    # prediction = model.predict([text],verbose=0)[0]  # Adjust this line based on how your model expects input
    # return "Negative" if prediction > threshold else "Positive"
    return llama_response(text)
# Streamlit interface
def main():
    st.title("Sentiment Analysis App")

    # Initialize the user_input in session state if it does not exist
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""

    # Text area that uses session state for its value
    user_input = st.text_area("Enter Text Here:", value=st.session_state['user_input'], key="user_input_area")

    sentiment = ""
    b2 = "Get Llama Take on this"  # Default button text

    if st.button("Analyze"):
        sentiment = predict_sentiment(user_input)
        st.session_state['sentiment'] = sentiment  # Save sentiment to session state
        if "NEG" in sentiment.upper():
            b2 = "Get Llama Therapy"
        st.write(f"Sentiment: {sentiment}")

    if 'sentiment' in st.session_state and sentiment:
        if st.button(b2):  # Show button with dynamic text
            response = llama_response(user_input, 1)
            st.write(f"Llama 3 says: {response}")

    if st.button("Clear"):
        # Clear the text area by setting its value in session state to an empty string
        st.session_state['user_input'] = ""
        # Optionally, you can also clear the sentiment and any other data if needed
        if 'sentiment' in st.session_state:
            del st.session_state['sentiment']

if __name__ == '__main__':
    main()
    