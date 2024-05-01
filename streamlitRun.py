import streamlit as st
from keras.src.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.models import Model
from keras.layers import Dense, Dropout,Activation,SimpleRNN, Bidirectional, BatchNormalization,LSTM,Embedding, GRU,Input,GlobalMaxPooling1D,Dropout,Bidirectional
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.callbacks import EarlyStopping
import json
from groq import Groq
import random

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

def simple_RNN_model():
    model = Sequential()
    model.add(SimpleRNN(100, input_shape=(maxlen, 1), return_sequences=False))  # Reduced units to prevent overfitting
    model.add(Dense(50, activation='relu'))  # Added ReLU activation
    model.add(Dense(25, activation='relu'))  # Simplified architecture
    model.add(Dense(1, activation='sigmoid'))

    # Correctly use the custom optimizer with a specified learning rate
    adam = optimizers.Adam(learning_rate=0.001)  # More typical learning rate
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def simple_RNN():
    model = simple_RNN_model()
    model.load_weights("./sentiment_weights_Simple.h5")
    return model
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

def simple_LSTM_model():
    model = Sequential()
    model.add(Embedding(3000, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model


def Final_LSTM():
    model = lstm_rnn_model()
    model.load_weights("./sentiment_weights_lstmFin.h5")
    return model
def Simple_LSTM():
    model = simple_LSTM_model()
    model.load_weights("./Simple_LSTM1.h5")
    return model


def GRU_model():
    learning_rate = 0.0001  # Set your desired learning rate here
    v=9594
    inputt=Input(shape=(maxlen,))
    D=100
    x = Embedding(v + 1, D)(inputt)
    x = Dropout(0.5)(x)
    x = Bidirectional(GRU(200))(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputt, x)
    return model

def GRU():
    model = GRU_model()
    model.load_weights("./GRU.h5")
    return model


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

def predict_sentiment(text):


    model_pred=[]
    model_pred.append(simple_RNN())
    model_pred.append(Simple_LSTM())
    model_pred.append(Final_LSTM())
    model_pred.append(GRU_model())

    prediction=[]
    for i in model_pred:
        prediction.append(i.predict([text],verbose=0)[0])

    threshold= [0.5035825,0.6317706,0.6014326,0.5776334]

    preds=[]

    sentiment = llama_response(text)

    for i,val in enumerate(prediction):
        if val>threshold[i]:
            preds.append(1)
        else:
            preds.append(0)


    preds.sort()
    preds.append(sentiment)

    return preds[-1]

# Streamlit interface
def main():
    st.title("Sentiment Analysis App")
    user_input = st.text_area("Enter Text Here:")

    # Initialize button label in session state if not already present
    if 'sentiment_available' not in st.session_state:
        st.session_state['sentiment_available'] = False  # Track if sentiment has been analyzed
        st.session_state['button_label'] = "Get Llama Take on this"  # Default label

    if st.button("Analyze"):
        sentiment = predict_sentiment(user_input)
        st.session_state['sentiment'] = sentiment  # Save sentiment to session state
        st.session_state['sentiment_available'] = True  # Indicate that sentiment has been analyzed
        if "NEG" in sentiment.upper():
            st.session_state['button_label'] = "Get Llama Therapy"
        else:
            st.session_state['button_label'] = "Get Llama Take on this"
        st.write(f"Sentiment: {sentiment}")

    # Display the button for llama response only if the sentiment has been analyzed
    if st.session_state['sentiment_available']:
        if st.button(st.session_state['button_label']):
            response = llama_response(user_input, 1)
            st.write(f"Llama 3 says: {response}")

    if st.button("Clear"):
        # Clear the session state for user input and sentiment
        st.session_state['user_input'] = ""
        st.session_state['sentiment'] = ""
        st.session_state['sentiment_available'] = False  # Reset the availability of sentiment analysis

if __name__ == '__main__':
    main()
    